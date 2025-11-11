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
        query = f"""
{date}の金融・経済に関する重要なニュース記事を{max_articles}件取得してください。

【検索条件】
- 日付: {date}
- トピック: {topics_str}
- 情報源: Bloomberg, Reuters, CNBC, Financial Times, Wall Street Journal, 日経新聞などの信頼できる金融メディア

【出力形式】
各記事について以下の情報を JSON 形式で出力してください：
{{
  "articles": [
    {{
      "title": "記事のタイトル",
      "summary": "記事の要約（200-300文字）",
      "content": "記事の本文（可能な限り詳細に、最低1000文字以上）",
      "source": "情報源（例: Bloomberg, Reuters）",
      "url": "記事のURL（取得できた場合）",
      "published_at": "公開日時（ISO 8601形式）"
    }}
  ]
}}

【重要な指示】
1. content（本文）は必ず1000文字以上含めてください
2. 実際の記事から取得した内容のみを出力してください（架空の記事は含めない）
3. 各記事のURLも取得してください
4. 金融・経済・投資家に関連性の高い記事を優先してください
5. 最新で信頼できる情報源から取得してください
"""

        try:
            # OpenAI Responses APIを使用（Web検索付き）
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Web検索対応モデル
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは金融ニュースの検索と要約を専門とするアシスタントです。Web検索を使用して最新の金融ニュースを取得し、正確な情報を提供してください。",
                    },
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            # レスポンスを解析
            import json

            result = json.loads(response.choices[0].message.content)
            articles = result.get("articles", [])

            logger.info(f"Successfully fetched {len(articles)} articles via OpenAI Web Search")

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


def main():
    """テスト実行"""
    import logging

    logging.basicConfig(level=logging.INFO)

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
