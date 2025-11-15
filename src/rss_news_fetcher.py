"""
RSSフィードを使用した実際のニュース取得モジュール
信頼できるニュースソースからRSSフィードを取得し、記事をスクレイピング
OpenAI APIで記事内容を拡充
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
import pytz
import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
from openai import OpenAI

logger = logging.getLogger(__name__)


class RSSNewsFetcher:
    """RSSフィードを使用したニュース取得クラス"""

    # 主要な金融ニュースのRSSフィード
    RSS_FEEDS = {
        "reuters_business": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
        "bloomberg_markets": "https://feeds.bloomberg.com/markets/news.rss",
        "cnbc_world": "https://www.cnbc.com/id/100727036/device/rss/rss.html",
        "marketwatch": "https://www.marketwatch.com/rss/topstories",
        "financial_times": "https://www.ft.com/?format=rss",
    }

    def __init__(self, api_key: Optional[str] = None, use_openai_enhancement: bool = True):
        """
        初期化

        Args:
            api_key: OpenAI API Key（Noneの場合は環境変数から取得）
            use_openai_enhancement: OpenAI APIで記事内容を拡充するか
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

        self.use_openai_enhancement = use_openai_enhancement
        if use_openai_enhancement:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                logger.warning("OPENAI_API_KEY not set, OpenAI enhancement disabled")
                self.use_openai_enhancement = False
            else:
                self.client = OpenAI(api_key=self.api_key)

        logger.info(f"RSSNewsFetcher initialized (OpenAI enhancement: {self.use_openai_enhancement})")

    def fetch_rss_articles(
        self,
        max_articles: int = 20,
        feeds: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        RSSフィードから記事を取得

        Args:
            max_articles: 取得する最大記事数
            feeds: 使用するフィード名のリスト（Noneの場合は全て）

        Returns:
            記事情報のリスト
        """
        if feeds is None:
            feeds = list(self.RSS_FEEDS.keys())

        logger.info(f"Fetching articles from {len(feeds)} RSS feeds...")

        all_articles = []

        for feed_name in feeds:
            if len(all_articles) >= max_articles:
                break

            feed_url = self.RSS_FEEDS.get(feed_name)
            if not feed_url:
                logger.warning(f"Unknown feed: {feed_name}")
                continue

            try:
                logger.info(f"  Fetching from {feed_name}: {feed_url}")
                feed = feedparser.parse(feed_url)

                if not feed.entries:
                    logger.warning(f"    No entries found in feed: {feed_name}")
                    continue

                logger.info(f"    Found {len(feed.entries)} entries")

                for entry in feed.entries:
                    if len(all_articles) >= max_articles:
                        break

                    # RSSエントリから記事情報を抽出
                    article = self._parse_rss_entry(entry, feed_name)

                    if article:
                        # OpenAI APIで記事内容を拡充
                        if self.use_openai_enhancement:
                            article = self.enhance_article_with_openai(article)

                        all_articles.append(article)
                        logger.info(f"      ✓ Added: {article['title'][:50]}...")

                time.sleep(1)  # レート制限対策

            except Exception as e:
                logger.error(f"Error fetching RSS feed {feed_name}: {e}")
                continue

        logger.info(f"Successfully fetched {len(all_articles)} articles from RSS feeds")
        return all_articles[:max_articles]

    def _parse_rss_entry(self, entry, feed_name: str) -> Optional[Dict]:
        """
        RSSエントリから記事情報を抽出

        Args:
            entry: feedparserのエントリオブジェクト
            feed_name: フィード名

        Returns:
            記事情報の辞書、失敗時はNone
        """
        try:
            # タイトル
            title = entry.get("title", "")

            # URL
            url = entry.get("link", "")

            # 要約（RSSのdescription or summary）
            summary = entry.get("summary", entry.get("description", ""))

            # HTMLタグを除去
            if summary:
                soup = BeautifulSoup(summary, "html.parser")
                summary = soup.get_text(strip=True)

            # 公開日時
            published_at = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_at = datetime(*entry.published_parsed[:6], tzinfo=pytz.UTC)
            else:
                published_at = datetime.now(pytz.UTC)

            # ソース名を決定
            source = self._get_source_name(feed_name)

            # 記事本文を取得（スクレイピング）
            content = self._fetch_article_content(url)

            if not title or not url:
                logger.warning(f"Missing title or URL in RSS entry")
                return None

            # contentが短すぎる場合はsummaryを使用
            if len(content) < 200:
                content = summary

            return {
                "title": title,
                "summary": summary[:500] if summary else "",
                "content": content,
                "source": source,
                "url": url,
                "published_at": published_at,
            }

        except Exception as e:
            logger.error(f"Error parsing RSS entry: {e}")
            return None

    def _get_source_name(self, feed_name: str) -> str:
        """フィード名からソース名を取得"""
        source_map = {
            "reuters_business": "Reuters",
            "bloomberg_markets": "Bloomberg",
            "cnbc_world": "CNBC",
            "marketwatch": "MarketWatch",
            "financial_times": "Financial Times",
        }
        return source_map.get(feed_name, "Unknown Source")

    def _fetch_article_content(self, url: str, timeout: int = 10) -> str:
        """
        記事URLから本文をスクレイピング

        Args:
            url: 記事URL
            timeout: タイムアウト時間（秒）

        Returns:
            記事本文
        """
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # 一般的な記事本文のタグを探す
            content_paragraphs = []

            # 複数のセレクタを試す
            selectors = [
                "article p",
                ".article-body p",
                ".article-content p",
                ".story-body p",
                "p",
            ]

            for selector in selectors:
                elements = soup.select(selector)
                if elements and len(elements) > 3:
                    for p in elements:
                        text = p.get_text(strip=True)
                        # 短すぎる段落や広告を除外
                        if len(text) > 50 and not any(
                            exclude in text.lower()
                            for exclude in ["subscribe", "sign up", "advertisement", "cookie"]
                        ):
                            content_paragraphs.append(text)
                    break

            if content_paragraphs:
                return "\n\n".join(content_paragraphs[:20])  # 最初の20段落

            return ""

        except Exception as e:
            logger.debug(f"Could not fetch article content from {url}: {e}")
            return ""

    def enhance_article_with_openai(self, article: Dict) -> Dict:
        """
        OpenAI APIで記事内容を拡充

        Args:
            article: 記事情報の辞書

        Returns:
            拡充された記事情報
        """
        if not self.use_openai_enhancement:
            return article

        try:
            title = article.get("title", "")
            summary = article.get("summary", "")
            url = article.get("url", "")
            source = article.get("source", "")

            prompt = f"""以下の実際のニュース記事について、より詳細な解説記事を作成してください。

【元記事情報】
タイトル: {title}
ソース: {source}
URL: {url}
要約: {summary}

【指示】
1. 元記事の内容を基に、より詳細で分かりやすい解説記事（1200-1500文字）を作成してください
2. 以下の構成で記述してください：
   - 導入・背景（200-300文字）: ニュースの背景や重要性
   - 詳細分析（500-600文字）: 具体的なデータや影響の分析
   - 市場への影響（300-400文字）: 投資家や市場への影響
   - 今後の見通し（200-300文字）: 将来的な展望
3. 具体的な数字やデータがあれば含めてください
4. 投資家にとって実用的な情報を提供してください
5. 元記事の内容を尊重し、事実に基づいた解説にしてください

【出力形式】
解説記事の本文のみを出力してください（見出しや装飾は不要）。
"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは金融ニュースの解説を専門とするアシスタントです。実際のニュース記事を基に、詳細で分かりやすい解説記事を作成してください。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2500,
            )

            enhanced_content = response.choices[0].message.content.strip()

            # 拡充された内容で記事を更新
            article["content"] = enhanced_content
            logger.info(f"  Enhanced article with OpenAI ({len(enhanced_content)} chars)")

            return article

        except Exception as e:
            logger.error(f"Error enhancing article with OpenAI: {e}")
            return article


def main():
    """テスト実行"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    fetcher = RSSNewsFetcher()

    # RSSフィードから記事を取得
    articles = fetcher.fetch_rss_articles(max_articles=5)

    print(f"\n取得した記事数: {len(articles)}\n")

    for i, article in enumerate(articles, 1):
        print(f"=== 記事 {i} ===")
        print(f"タイトル: {article['title']}")
        print(f"ソース: {article['source']}")
        print(f"URL: {article['url']}")
        print(f"要約: {article['summary'][:150]}...")
        print(f"本文の文字数: {len(article['content'])}")
        print(f"公開日時: {article['published_at']}")
        print()


if __name__ == "__main__":
    main()
