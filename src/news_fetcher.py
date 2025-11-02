"""
経済ニュース取得モジュール
RSSフィードとNewsAPIから最新の経済ニュースを取得
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import feedparser
import requests
from bs4 import BeautifulSoup
import pytz

logger = logging.getLogger(__name__)


class NewsArticle:
    """ニュース記事を表す基本クラス"""

    def __init__(
        self,
        title: str,
        summary: str,
        url: str,
        published_at: datetime,
        source: str,
        content: Optional[str] = None,
    ):
        self.title = title
        self.summary = summary
        self.url = url
        self.published_at = published_at
        self.source = source
        self.content = content or summary

    def __repr__(self):
        return f"NewsArticle(title='{self.title[:30]}...', source='{self.source}')"

    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "source": self.source,
            "content": self.content,
        }


class NewsFetcher:
    """ニュース取得クラス"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書（config.yamlから読み込んだnews設定）
        """
        self.config = config
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.max_articles = config.get("max_articles", 5)
        self.max_age_hours = config.get("max_age_hours", 24)
        self.categories = config.get("categories", ["経済", "ビジネス"])
        self.rss_feeds = config.get("rss_feeds", [])

    def fetch_news(self) -> List[NewsArticle]:
        """
        複数のソースからニュースを取得

        Returns:
            NewsArticleオブジェクトのリスト
        """
        articles = []
        sources = self.config.get("sources", ["rss"])

        if "rss" in sources:
            logger.info("Fetching news from RSS feeds...")
            rss_articles = self._fetch_from_rss()
            articles.extend(rss_articles)

        if "newsapi" in sources and self.news_api_key:
            logger.info("Fetching news from NewsAPI...")
            newsapi_articles = self._fetch_from_newsapi()
            articles.extend(newsapi_articles)

        # 日付でソート（新しい順）
        articles.sort(key=lambda x: x.published_at, reverse=True)

        # 重複を削除（タイトルが類似しているものを除外）
        unique_articles = self._remove_duplicates(articles)

        # 指定された数に制限
        return unique_articles[: self.max_articles]

    def _fetch_from_rss(self) -> List[NewsArticle]:
        """RSSフィードからニュースを取得"""
        articles = []
        cutoff_time = datetime.now(pytz.UTC) - timedelta(hours=self.max_age_hours)

        for feed_url in self.rss_feeds:
            try:
                logger.debug(f"Parsing RSS feed: {feed_url}")
                feed = feedparser.parse(feed_url)

                for entry in feed.entries:
                    # 公開日時を取得
                    published_at = self._parse_datetime(entry.get("published"))
                    if not published_at or published_at < cutoff_time:
                        continue

                    # 記事を作成
                    article = NewsArticle(
                        title=entry.get("title", ""),
                        summary=self._clean_html(entry.get("summary", "")),
                        url=entry.get("link", ""),
                        published_at=published_at,
                        source=feed.feed.get("title", "RSS"),
                    )
                    articles.append(article)
                    logger.debug(f"Found article: {article.title[:50]}...")

            except Exception as e:
                logger.error(f"Error fetching RSS feed {feed_url}: {e}")

        logger.info(f"Fetched {len(articles)} articles from RSS feeds")
        return articles

    def _fetch_from_newsapi(self) -> List[NewsArticle]:
        """NewsAPIからニュースを取得"""
        articles = []

        if not self.news_api_key:
            logger.warning("NEWS_API_KEY not set, skipping NewsAPI")
            return articles

        try:
            # NewsAPI エンドポイント
            url = "https://newsapi.org/v2/everything"

            # クエリパラメータ
            params = {
                "apiKey": self.news_api_key,
                "q": "経済 OR ビジネス OR 金融",
                "language": "ja",
                "sortBy": "publishedAt",
                "pageSize": self.max_articles * 2,  # 多めに取得してフィルタ
                "from": (
                    datetime.now() - timedelta(hours=self.max_age_hours)
                ).isoformat(),
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                logger.error(f"NewsAPI error: {data.get('message')}")
                return articles

            for item in data.get("articles", []):
                # 記事を作成
                published_at = self._parse_datetime(item.get("publishedAt"))
                if not published_at:
                    continue

                article = NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("description", ""),
                    url=item.get("url", ""),
                    published_at=published_at,
                    source=item.get("source", {}).get("name", "NewsAPI"),
                    content=item.get("content", ""),
                )
                articles.append(article)

            logger.info(f"Fetched {len(articles)} articles from NewsAPI")

        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")

        return articles

    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """日時文字列をdatetimeオブジェクトに変換"""
        if not date_str:
            return None

        try:
            # ISO 8601形式
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = pytz.UTC.localize(dt)
            return dt
        except:
            pass

        try:
            # RFC 2822形式（RSS）
            from email.utils import parsedate_to_datetime

            return parsedate_to_datetime(date_str)
        except:
            pass

        logger.warning(f"Could not parse date: {date_str}")
        return None

    def _clean_html(self, html: str) -> str:
        """HTMLタグを除去"""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text().strip()

    def _remove_duplicates(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """類似タイトルの記事を除去"""
        unique_articles = []
        seen_titles = set()

        for article in articles:
            # タイトルを正規化（空白削除、小文字化）
            normalized_title = "".join(article.title.split()).lower()

            # 類似タイトルをチェック（最初の30文字で判定）
            title_prefix = normalized_title[:30]

            if title_prefix not in seen_titles:
                seen_titles.add(title_prefix)
                unique_articles.append(article)

        logger.info(
            f"Removed {len(articles) - len(unique_articles)} duplicate articles"
        )
        return unique_articles

    def get_top_news(self) -> Optional[NewsArticle]:
        """
        最も重要なニュースを1つ取得

        Returns:
            最新のニュース記事、または None
        """
        articles = self.fetch_news()
        return articles[0] if articles else None


def main():
    """テスト実行用"""
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    # 設定読み込み
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)

    # ニュース取得
    fetcher = NewsFetcher(config["news"])
    articles = fetcher.fetch_news()

    print(f"\n取得したニュース: {len(articles)}件\n")
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article.title}")
        print(f"   ソース: {article.source}")
        print(f"   公開日時: {article.published_at}")
        print(f"   URL: {article.url}")
        print(f"   要約: {article.summary[:100]}...")
        print()


if __name__ == "__main__":
    main()
