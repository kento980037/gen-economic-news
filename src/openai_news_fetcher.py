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

        # 注: GPT-4oの知識カットオフは2024年10月
        # リアルタイムWeb検索が必要な場合は、Perplexity/Tavily/Serper APIの統合を検討
        query = f"""
以下のトピックについて、最近の金融・経済ニュース記事を{max_articles}件作成してください。

【対象トピック】
{topics_str}

【指示】
1. 各トピックについて、実際に存在する企業・指標・政策に基づいた記事を作成
2. タイトル、要約、詳細な本文を含める
3. 架空の情報や不確実な予測は避ける
4. 投資家向けに有益な内容にする

【JSON形式で出力】
{{
  "articles": [
    {{
      "title": "具体的な記事タイトル",
      "summary": "記事の要約（200-300文字）",
      "content": "詳細な記事本文（1000-2000文字、背景・影響・見通しを含む）",
      "source": "Bloomberg",
      "url": "https://www.bloomberg.com/news/articles/example",
      "published_at": "{date}T10:00:00Z"
    }}
  ]
}}

必ず{max_articles}件の記事を生成してください。各記事のcontentは1000文字以上にしてください。
"""

        try:
            # OpenAI Chat Completions APIを使用
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # コスト効率重視
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは金融ニュースの作成を専門とするアシスタントです。正確で詳細な情報を提供してください。",
                    },
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
                max_tokens=8000,  # 十分な出力トークンを確保
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
        メイン記事に関連する記事を検索（複数回API呼び出しで多数取得可能）

        Args:
            main_article_title: メイン記事のタイトル
            main_article_summary: メイン記事の要約
            keywords: 検索キーワードのリスト
            max_articles: 取得する最大記事数

        Returns:
            関連記事のリスト
        """
        all_articles = []

        # 1回のAPI呼び出しで取得する記事数（5件が最適）
        batch_size = 5
        num_batches = (max_articles + batch_size - 1) // batch_size  # 切り上げ

        logger.info(f"Fetching {max_articles} related articles in {num_batches} batches (batch_size={batch_size})")

        for batch_num in range(num_batches):
            articles_to_fetch = min(batch_size, max_articles - len(all_articles))

            if articles_to_fetch <= 0:
                break

            logger.info(f"  Batch {batch_num + 1}/{num_batches}: Fetching {articles_to_fetch} articles...")

            # バッチごとに異なる視点で記事を生成
            if batch_num == 0:
                focus = "メイン記事の背景や関連情報を補足する"
            elif batch_num == 1:
                focus = "メイン記事に登場する企業・組織・人物についての詳細情報を提供する"
            elif batch_num == 2:
                focus = "メイン記事のトピックに関連する市場動向や投資家への影響を分析する"
            else:
                focus = "メイン記事に関連する技術・政策・経済指標についての解説を提供する"

            keywords_str = "、".join(keywords[:5])  # 上位5つのキーワード

            query = f"""
以下のメイン記事に関連する金融・経済ニュース記事を{articles_to_fetch}件作成してください。

【メイン記事】
タイトル: {main_article_title}
要約: {main_article_summary}

【関連キーワード】
{keywords_str}

【記事作成の視点】
{focus}

【指示】
1. 上記の視点から、メイン記事を補完する記事を作成
2. 実在する企業・指標・政策に基づいた内容にする
3. 各記事は500-800文字程度の本文を含める
4. 既に作成された記事とは異なる視点・内容にする

【JSON形式で出力】
{{
  "articles": [
    {{
      "title": "関連記事のタイトル",
      "summary": "記事の要約（150-200文字）",
      "content": "記事本文（500-800文字）",
      "source": "Bloomberg",
      "url": "https://www.bloomberg.com/news/articles/example",
      "published_at": "2025-11-11T10:00:00Z"
    }}
  ]
}}

必ず{articles_to_fetch}件の記事を生成してください。
"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "あなたは金融ニュースの関連記事作成を専門とするアシスタントです。メイン記事を補足する有益な情報を提供してください。",
                        },
                        {"role": "user", "content": query},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.6,  # 多様性を高めるため少し上げる
                    max_tokens=6000,
                )

                import json
                response_content = response.choices[0].message.content
                result = json.loads(response_content)
                articles = result.get("articles", [])

                logger.info(f"    Retrieved {len(articles)} articles in batch {batch_num + 1}")

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
