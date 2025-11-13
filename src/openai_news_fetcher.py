"""
OpenAI Web Searchを使用したニュース取得モジュール
RSSの代わりにOpenAI APIのWeb検索機能を使用して最新ニュースを取得
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
import pytz
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAINewsFetcher:
    """OpenAI Web Searchを使用したニュース取得クラス"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: OpenAI API Key（Noneの場合は環境変数から取得）
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=self.api_key)
        logger.info("OpenAINewsFetcher initialized")

    def fetch_financial_news(
        self,
        date: Optional[str] = None,
        max_articles: int = 20,
        topics: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        金融・経済ニュースを取得

        Args:
            date: 検索対象の日付（YYYY-MM-DD形式、Noneの場合は今日）
            max_articles: 取得する最大記事数
            topics: 検索トピック（例: ["株式市場", "金融政策", "テクノロジー"]）

        Returns:
            ニュース記事のリスト
            [{"title": str, "summary": str, "content": str, "source": str, "url": str, "published_at": datetime}, ...]
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        if topics is None:
            topics = [
                "株式市場",
                "金融政策",
                "中央銀行",
                "経済指標",
                "企業決算",
                "テクノロジー",
                "半導体",
            ]

        logger.info(f"Fetching financial news for date: {date}, topics: {topics}")

        # 検索クエリを構築
        topics_str = "、".join(topics)

        # 注: GPT-4o-miniの知識カットオフは2024年10月
        # 最新の人事・役職情報は反映されない可能性があるため注意
        query = f"""
以下のトピックについて、金融・経済ニュース記事を{max_articles}件作成してください。

【対象トピック】
{topics_str}

【参考にすべきスタイル・品質】
Reuters Markets (https://www.reuters.com/markets/) のような高品質な金融ニュースを参考にしてください：
- 具体的なデータと数字を重視
- 市場への影響を明確に説明
- 専門家のコメントや分析を含める
- 客観的で正確な報道スタイル
- 投資家にとって実用的な情報

【重要な注意事項】
- あなたの知識カットオフは2024年10月です
- 2024年10月以降の情報については推測しないでください
- 人事・役職については知識カットオフ時点の最新情報を使用してください
- 例: 日本銀行総裁は植田和男氏（2023年4月就任）
- 架空の人物名や役職を作成しないでください
- 不確実な情報は含めないでください

【記事作成の指示】
1. Reuters Marketsのような具体的で詳細な記事を作成
2. 実際に存在する企業・指標・政策に基づいた内容
3. 具体的な数字、パーセンテージ、金額を含める
4. 市場関係者や専門家の見解を含める（実在する人物の場合のみ）
5. 投資家向けに実用的で価値のある情報を提供
6. 人物を言及する場合は、知識カットオフ時点で正確な情報のみを使用

【JSON形式で出力】
{{
  "articles": [
    {{
      "title": "具体的な記事タイトル",
      "summary": "記事の要約（200-300文字）",
      "content": "詳細な記事本文（最低1200文字、理想は1500-2000文字）。必ず以下を含めること：
        1. 導入・背景（200-300文字）
        2. 具体的なデータ・数字・事実（400-500文字）
        3. 専門家の分析・市場の反応（300-400文字）
        4. 投資家への影響・今後の見通し（300-400文字）",
      "source": "Bloomberg",
      "url": "https://www.bloomberg.com/news/articles/example",
      "published_at": "{date}T10:00:00Z"
    }}
  ]
}}

【重要】必ず{max_articles}件の記事を生成してください。
【必須】各記事のcontentは必ず1200文字以上にしてください。1000文字未満は不可です。
"""

        try:
            # OpenAI Chat Completions APIを使用
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # コスト効率重視
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは金融ニュースの作成を専門とするアシスタントです。Reuters Marketsのような高品質で詳細な記事を作成してください。具体的なデータ、数字、専門家の分析を豊富に含めてください。",
                    },
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
                max_tokens=12000,  # 1200文字×5記事に対応
            )

            # レスポンスを解析
            import json

            response_content = response.choices[0].message.content
            logger.debug(f"Raw response content (first 500 chars): {response_content[:500]}")

            result = json.loads(response_content)
            logger.debug(f"Parsed JSON keys: {result.keys()}")

            articles = result.get("articles", [])

            if not articles:
                logger.warning(f"No articles found in response. Full response: {response_content[:1000]}")
            else:
                logger.info(f"Successfully fetched {len(articles)} articles via OpenAI")
                # 各記事の文字数をログ
                for i, article in enumerate(articles, 1):
                    content_len = len(article.get("content", ""))
                    logger.debug(f"  Article {i}: {article.get('title', 'No title')[:50]}... (content: {content_len} chars)")

            # NewsArticle形式に変換
            news_articles = []
            for article in articles:
                news_articles.append(
                    {
                        "title": article.get("title", ""),
                        "summary": article.get("summary", ""),
                        "content": article.get("content", ""),
                        "source": article.get("source", "OpenAI Web Search"),
                        "url": article.get("url", ""),
                        "published_at": self._parse_datetime(
                            article.get("published_at", "")
                        )
                        or datetime.now(pytz.UTC),
                    }
                )

            return news_articles

        except Exception as e:
            logger.error(f"Error fetching news via OpenAI Web Search: {e}")
            return []

    def _parse_datetime(self, datetime_str: str) -> Optional[datetime]:
        """日時文字列をdatetimeオブジェクトに変換"""
        if not datetime_str:
            return None

        try:
            # ISO 8601形式
            if "T" in datetime_str:
                dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
                return dt.astimezone(pytz.UTC)
        except Exception as e:
            logger.debug(f"Failed to parse datetime: {datetime_str}, error: {e}")

        return None

    def search_related_articles(
        self,
        main_article_title: str,
        main_article_summary: str,
        keywords: List[str],
        max_articles: int = 5,
    ) -> List[Dict]:
        """
        メイン記事に関連する記事を検索（1件ずつ高品質生成）

        Args:
            main_article_title: メイン記事のタイトル
            main_article_summary: メイン記事の要約
            keywords: 検索キーワードのリスト
            max_articles: 取得する最大記事数

        Returns:
            関連記事のリスト
        """
        all_articles = []

        # 1記事ずつ生成して品質を最大化
        batch_size = 1
        num_batches = max_articles

        logger.info(f"Fetching {max_articles} related articles (1 article per API call for maximum quality)")

        for batch_num in range(num_batches):
            articles_to_fetch = 1  # 常に1件ずつ

            if articles_to_fetch <= 0:
                break

            logger.info(f"  Article {batch_num + 1}/{num_batches}: Generating high-quality article...")

            # 記事ごとに異なる視点で生成
            focus_options = [
                "メイン記事の背景や歴史的文脈を詳しく解説する",
                "メイン記事に登場する企業・組織・人物についての深掘り分析を提供する",
                "メイン記事のトピックに関連する市場動向や投資家への具体的な影響を分析する",
                "メイン記事に関連する技術・政策・経済指標についての専門的な解説を提供する",
                "メイン記事の国際的な影響や他国との比較を詳しく分析する",
                "メイン記事が投資戦略に与える影響と具体的なアクションを提案する",
                "メイン記事の長期的なトレンドと今後の展開を予測分析する",
                "メイン記事に関連するリスク要因と対策を詳しく解説する",
                "メイン記事のセクター・業界への影響を深掘りする",
                "メイン記事の経済理論・学術的背景を解説する"
            ]
            focus = focus_options[batch_num % len(focus_options)]

            keywords_str = "、".join(keywords[:5])  # 上位5つのキーワード

            query = f"""
以下のメイン記事に関連する金融・経済ニュース記事を1件作成してください。

【メイン記事】
タイトル: {main_article_title}
要約: {main_article_summary}

【関連キーワード】
{keywords_str}

【記事作成の視点】
{focus}

【参考にすべきスタイル・品質】
Reuters Markets (https://www.reuters.com/markets/) のような高品質な金融ニュースを参考にしてください：
- 具体的なデータと数字を重視
- 市場への影響を明確に説明
- 専門家のコメントや分析を含める
- 客観的で正確な報道スタイル
- 投資家にとって実用的な情報

【重要な注意事項】
- あなたの知識カットオフは2024年10月です
- 人事・役職については知識カットオフ時点の最新情報を使用してください
- 例: 日本銀行総裁は植田和男氏（2023年4月就任）、黒田東彦氏は2023年4月に退任
- 架空の人物名や古い役職情報を使用しないでください
- 不確実な情報は含めないでください

【記事作成の指示】
1. Reuters Marketsのような具体的で詳細な関連記事を1件作成
2. 上記の視点から、メイン記事を補完する内容にする
3. 実在する企業・指標・政策に基づいた内容
4. 具体的な数字、パーセンテージ、金額を豊富に含める
5. 本文は1000-1500文字を目標とする（800文字未満は不可）
6. 人物を言及する場合は、現在の正しい役職を使用する
7. 全トークンを使って、徹底的に深掘りした記事を作成する

【JSON形式で出力】
{{
  "articles": [
    {{
      "title": "関連記事のタイトル（具体的で魅力的に）",
      "summary": "記事の要約（200-250文字、詳細に）",
      "content": "記事本文（1000-1500文字を目標）。必ず以下の構成で：
        1. 導入・背景（200-250文字）- なぜこのトピックが重要か
        2. 具体的なデータ・数字・事実（400-500文字）- 詳細な統計や事例
        3. 専門家の分析・市場の反応（250-350文字）- 深い洞察
        4. 投資家への影響・今後の見通し（200-300文字）- 実践的なアドバイス",
      "source": "Bloomberg",
      "url": "https://www.bloomberg.com/news/articles/example",
      "published_at": "2025-11-11T10:00:00Z"
    }}
  ]
}}

【最重要】この1件の記事に全力を注いでください。contentは1000文字以上必須です。
"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "あなたは金融ニュースの関連記事作成を専門とするアシスタントです。Reuters Marketsのような高品質で詳細な記事を作成し、メイン記事を補足する有益な情報を提供してください。具体的なデータと分析を豊富に含めてください。",
                        },
                        {"role": "user", "content": query},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.6,  # 多様性を高めるため少し上げる
                    max_tokens=8000,  # 800-1200文字×5記事に対応
                )

                import json
                response_content = response.choices[0].message.content
                result = json.loads(response_content)
                articles = result.get("articles", [])

                if articles:
                    content_length = len(articles[0].get("content", ""))
                    logger.info(f"    Generated article {batch_num + 1}: {articles[0].get('title', 'No title')[:50]}... ({content_length} chars)")
                else:
                    logger.warning(f"    No article generated in batch {batch_num + 1}")

                # NewsArticle形式に変換
                for article in articles:
                    all_articles.append(
                        {
                            "title": article.get("title", ""),
                            "summary": article.get("summary", ""),
                            "content": article.get("content", ""),
                            "source": article.get("source", "OpenAI"),
                            "url": article.get("url", ""),
                            "published_at": self._parse_datetime(
                                article.get("published_at", "")
                            )
                            or datetime.now(pytz.UTC),
                        }
                    )

            except Exception as e:
                logger.error(f"Error in batch {batch_num + 1}: {e}")
                continue

        logger.info(f"Total {len(all_articles)} related articles fetched via OpenAI")
        return all_articles[:max_articles]


def main():
    """テスト実行"""
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    fetcher = OpenAINewsFetcher()

    # 今日の金融ニュースを取得
    articles = fetcher.fetch_financial_news(max_articles=5)

    print(f"\n取得した記事数: {len(articles)}\n")

    for i, article in enumerate(articles, 1):
        print(f"=== 記事 {i} ===")
        print(f"タイトル: {article['title']}")
        print(f"ソース: {article['source']}")
        print(f"URL: {article['url']}")
        print(f"要約: {article['summary'][:100]}...")
        print(f"本文の文字数: {len(article['content'])}")
        print(f"公開日時: {article['published_at']}")
        print()


if __name__ == "__main__":
    main()
