"""
経済ニュースサイトから記事をスクレイピングするモジュール
Yahoo!ファイナンス、CNBCから取得
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime
import pytz
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import random
from openai import OpenAI

logger = logging.getLogger(__name__)


class NewsScraper:
    """ニュースサイトスクレイピングクラス"""

    def __init__(self, api_key: Optional[str] = None, use_openai_enhancement: bool = True):
        """
        初期化

        Args:
            api_key: OpenAI API Key（記事内容拡充用）
            use_openai_enhancement: OpenAI APIで記事内容を拡充するか
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        })

        self.use_openai_enhancement = use_openai_enhancement
        if use_openai_enhancement:
            self.api_key = api_key
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                logger.warning("OpenAI API key not provided, enhancement disabled")
                self.use_openai_enhancement = False

        logger.info(f"NewsScraper initialized (OpenAI enhancement: {self.use_openai_enhancement})")

    def scrape_cnbc(self, max_articles: int = 10) -> List[Dict]:
        """
        CNBCから記事をスクレイピング（軽量版：タイトルと要約のみ取得）

        Args:
            max_articles: 取得する最大記事数

        Returns:
            記事情報のリスト（url, title, summary, source, published_at）
        """
        articles = []
        base_url = "https://www.cnbc.com"
        url = base_url + "/world/"

        logger.info(f"Scraping CNBC for {max_articles} articles (lightweight)...")

        try:
            logger.info(f"  Scraping: {url}")

            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # 記事リンクを探す
            article_links = soup.find_all("a", href=True)

            for link in article_links:
                if len(articles) >= max_articles:
                    break

                href = link.get("href", "")

                # 記事URLのパターン
                if href.startswith(base_url) and "/2025/" in href:
                    full_url = href

                    # 重複チェック
                    if any(a["url"] == full_url for a in articles):
                        continue

                    # 記事ページから正確なタイトルと要約を取得（全文は取得しない）
                    article = self._fetch_cnbc_article_lightweight(full_url)
                    if article:
                        articles.append(article)
                        logger.info(f"    ✓ Found: {article['title'][:50]}...")
                        time.sleep(random.uniform(1.0, 2.0))  # レート制限対策

        except Exception as e:
            logger.error(f"Error scraping CNBC: {e}")

        logger.info(f"Found {len(articles)} articles from CNBC")
        return articles

    def _fetch_cnbc_article_lightweight(self, url: str) -> Optional[Dict]:
        """CNBCの記事タイトルと要約のみを取得（全文は取得しない）"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # タイトル
            title = ""
            title_tag = soup.find("h1")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # 要約（最初の2段落のみ）
            content_paragraphs = []
            article_body = soup.find("div", class_="ArticleBody-articleBody")
            if article_body:
                paragraphs = article_body.find_all("p")
                for i, p in enumerate(paragraphs):
                    if i >= 2:  # 最初の2段落だけ
                        break
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        content_paragraphs.append(text)

            summary = "\n".join(content_paragraphs) if content_paragraphs else ""

            if not title or len(summary) < 100:
                return None

            logger.debug(f"    Lightweight fetch - Title: {title[:50]}..., Summary length: {len(summary)} chars")

            return {
                "title": title,
                "summary": summary[:500],
                "content": summary,  # 軽量版では要約をcontentとして使用
                "source": "CNBC",
                "url": url,
                "published_at": datetime.now(pytz.UTC),
            }

        except Exception as e:
            logger.debug(f"Error fetching CNBC article {url}: {e}")
            return None

    def fetch_full_article(self, url: str, source: str = "CNBC") -> Optional[Dict]:
        """
        指定されたURLから記事の全文を取得（OpenAI拡充含む）

        Args:
            url: 記事URL
            source: ソース名（"CNBC" or "Yahoo!ファイナンス"）

        Returns:
            記事情報の辞書（title, summary, content, source, url, published_at）
        """
        if source == "CNBC":
            return self._fetch_cnbc_article(url)
        elif source == "Yahoo!ファイナンス":
            return self._fetch_yahoo_article(url)
        else:
            logger.error(f"Unknown source: {source}")
            return None

    def _fetch_cnbc_article(self, url: str) -> Optional[Dict]:
        """CNBCの記事詳細を取得"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # タイトル
            title = ""
            title_tag = soup.find("h1")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # 本文
            content_paragraphs = []
            article_body = soup.find("div", class_="ArticleBody-articleBody")
            if article_body:
                paragraphs = article_body.find_all("p")
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 50:
                        content_paragraphs.append(text)

            content = "\n\n".join(content_paragraphs)

            if not title or len(content) < 300:
                return None

            summary = "\n".join(content_paragraphs[:2]) if len(content_paragraphs) >= 2 else content[:300]

            logger.info(f"    Full article scraped - {len(content)} chars")
            logger.debug(f"    Article preview: {content[:200]}...")

            # OpenAI APIで記事内容を拡充（全文を渡す）
            if self.use_openai_enhancement:
                logger.info(f"    Enhancing with OpenAI...")
                content = self._enhance_content_with_openai(title, content, url, "CNBC")
                logger.info(f"    Enhanced content: {len(content)} chars")
                logger.debug(f"    Enhanced preview: {content[:200]}...")

            return {
                "title": title,
                "summary": summary[:500],
                "content": content,
                "source": "CNBC",
                "url": url,
                "published_at": datetime.now(pytz.UTC),
            }

        except Exception as e:
            logger.debug(f"Error fetching CNBC article {url}: {e}")
            return None

    def scrape_yahoo_finance(self, max_articles: int = 10) -> List[Dict]:
        """
        Yahoo!ファイナンスから記事をスクレイピング（軽量版：タイトルと要約のみ取得）

        Args:
            max_articles: 取得する最大記事数

        Returns:
            記事情報のリスト（url, title, summary, source, published_at）
        """
        articles = []
        base_url = "https://finance.yahoo.co.jp"
        url = base_url + "/news"

        logger.info(f"Scraping Yahoo! Finance for {max_articles} articles (lightweight)...")

        try:
            logger.info(f"  Scraping: {url}")

            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # 記事リンクを探す
            article_links = soup.find_all("a", href=True)

            for link in article_links:
                if len(articles) >= max_articles:
                    break

                href = link.get("href", "")

                # 記事URLのパターン
                if "/news/detail/" in href:
                    full_url = urljoin(base_url, href)

                    # 重複チェック
                    if any(a["url"] == full_url for a in articles):
                        continue

                    # 記事ページから正確なタイトルと要約を取得（全文は取得しない）
                    article = self._fetch_yahoo_article_lightweight(full_url)
                    if article:
                        articles.append(article)
                        logger.info(f"    ✓ Found: {article['title'][:50]}...")
                        time.sleep(random.uniform(1.0, 2.0))  # レート制限対策

        except Exception as e:
            logger.error(f"Error scraping Yahoo! Finance: {e}")

        logger.info(f"Found {len(articles)} articles from Yahoo! Finance")
        return articles

    def _fetch_yahoo_article_lightweight(self, url: str) -> Optional[Dict]:
        """Yahoo!ファイナンスの記事タイトルと要約のみを取得（全文は取得しない）"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # タイトル
            title = ""
            title_tag = soup.find("h1")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # 要約（最初の2段落のみ）
            content_paragraphs = []
            article_body = soup.find("div", class_="article-body")
            if not article_body:
                article_body = soup.find("article")

            if article_body:
                paragraphs = article_body.find_all("p")
                for i, p in enumerate(paragraphs):
                    if i >= 2:  # 最初の2段落だけ
                        break
                    text = p.get_text(strip=True)
                    if len(text) > 30:
                        content_paragraphs.append(text)

            summary = "\n".join(content_paragraphs) if content_paragraphs else ""

            if not title or len(summary) < 100:
                return None

            logger.debug(f"    Lightweight fetch - Title: {title[:50]}..., Summary length: {len(summary)} chars")

            return {
                "title": title,
                "summary": summary[:500],
                "content": summary,  # 軽量版では要約をcontentとして使用
                "source": "Yahoo!ファイナンス",
                "url": url,
                "published_at": datetime.now(pytz.UTC),
            }

        except Exception as e:
            logger.debug(f"Error fetching Yahoo! Finance article {url}: {e}")
            return None

    def _fetch_yahoo_article(self, url: str) -> Optional[Dict]:
        """Yahoo!ファイナンスの記事詳細を取得"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # タイトル
            title = ""
            title_tag = soup.find("h1")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # 本文
            content_paragraphs = []
            article_body = soup.find("div", class_="article-body")
            if not article_body:
                article_body = soup.find("article")

            if article_body:
                paragraphs = article_body.find_all("p")
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 30:
                        content_paragraphs.append(text)

            content = "\n\n".join(content_paragraphs)

            if not title or len(content) < 300:
                return None

            summary = "\n".join(content_paragraphs[:2]) if len(content_paragraphs) >= 2 else content[:300]

            logger.info(f"    Full article scraped - {len(content)} chars")
            logger.debug(f"    Article preview: {content[:200]}...")

            # OpenAI APIで記事内容を拡充（全文を渡す）
            if self.use_openai_enhancement:
                logger.info(f"    Enhancing with OpenAI...")
                content = self._enhance_content_with_openai(title, content, url, "Yahoo!ファイナンス")
                logger.info(f"    Enhanced content: {len(content)} chars")
                logger.debug(f"    Enhanced preview: {content[:200]}...")

            return {
                "title": title,
                "summary": summary[:500],
                "content": content,
                "source": "Yahoo!ファイナンス",
                "url": url,
                "published_at": datetime.now(pytz.UTC),
            }

        except Exception as e:
            logger.debug(f"Error fetching Yahoo! Finance article {url}: {e}")
            return None

    def _enhance_content_with_openai(self, title: str, content: str, url: str, source: str) -> str:
        """OpenAI APIで記事内容を拡充"""
        if not self.use_openai_enhancement:
            return content

        try:
            prompt = f"""以下の実際のニュース記事について、より詳細で分かりやすい解説記事（1200-1500文字）を日本語で作成してください。

【元記事情報】
タイトル: {title}
ソース: {source}
URL: {url}
本文: {content}

【指示】
1. 元記事の内容を基に、より詳細で分かりやすい解説記事を作成してください
2. 以下の構成で記述してください：
   - 導入・背景（200-300文字）: ニュースの背景や重要性
   - 詳細分析（500-600文字）: 具体的なデータや影響の分析
   - 市場への影響（300-400文字）: 投資家や経済への影響
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
                        "content": "あなたは経済ニュースの解説を専門とするアシスタントです。実際のニュース記事を基に、詳細で分かりやすい解説記事を日本語で作成してください。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2500,
            )

            enhanced_content = response.choices[0].message.content.strip()
            logger.info(f"  Enhanced content with OpenAI ({len(enhanced_content)} chars)")
            return enhanced_content

        except Exception as e:
            logger.error(f"Error enhancing content with OpenAI: {e}")
            return content

    def search_cnbc_articles(self, query: str, max_articles: int = 10) -> List[Dict]:
        """
        CNBCの検索API（Queryly）を使って記事を取得

        注意: このAPIはCNBCのWebサイトで使用されている公開APIです。
        - APIキーは公開情報（Webサイトに埋め込まれている）
        - いつでも変更・廃止される可能性があります
        - 過度なリクエストは避けてください

        Args:
            query: 検索クエリ（キーワード）
            max_articles: 取得する最大記事数

        Returns:
            記事情報のリスト
        """
        articles = []

        logger.info(f"Searching CNBC via Queryly API for '{query}' (max {max_articles} articles)...")

        try:
            # Queryly API を使用（CNBCのWebサイトで使用されている公開API）
            api_url = "https://api.queryly.com/cnbc/json.aspx"
            params = {
                "queryly_key": "31a35d40a9a64ab3",  # CNBCの公開APIキー（Webサイトから取得）
                "query": query,
                "endindex": min(max_articles * 2, 40),  # 上限設定（過度なリクエスト防止）
                "batchsize": min(max_articles * 2, 40),
                "showfaceted": "false"
            }

            # レート制限を考慮して少し待機
            time.sleep(random.uniform(0.5, 1.0))

            response = self.session.get(api_url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            if 'results' not in data:
                logger.warning(f"No results in API response for query: {query}")
                return articles

            results = data['results']
            logger.info(f"  API returned {len(results)} results")

            # 各結果から記事情報を抽出
            for result in results:
                if len(articles) >= max_articles:
                    break

                url = result.get('url', '')
                title = result.get('cn:title', '') or result.get('title', '')
                summary = result.get('description', '') or result.get('summary', '')

                # 2025年の記事のみ
                if '/2025/' not in url:
                    continue

                # 重複チェック
                if any(a["url"] == url for a in articles):
                    continue

                # 軽量版として要約をそのまま使用（詳細取得はしない）
                if title and summary and len(summary) > 50:
                    articles.append({
                        "title": title,
                        "summary": summary[:500],
                        "content": summary,  # 軽量版では要約をcontentとして使用
                        "source": "CNBC",
                        "url": url,
                        "published_at": datetime.now(pytz.UTC),
                    })
                    logger.info(f"    ✓ Found: {title[:50]}...")

        except Exception as e:
            logger.error(f"Error searching CNBC via Queryly API for '{query}': {e}")

        logger.info(f"Found {len(articles)} articles from CNBC search")
        return articles

    def fetch_articles(self, max_articles: int = 20, include_international: bool = True) -> List[Dict]:
        """
        複数のニュースサイトから記事を取得

        Args:
            max_articles: 取得する最大記事数
            include_international: 海外ニュースサイトを含めるか

        Returns:
            記事情報のリスト
        """
        all_articles = []

        # 日本のニュースサイト
        # Yahoo!ファイナンス
        yahoo_articles = self.scrape_yahoo_finance(max_articles=max_articles // 2)
        all_articles.extend(yahoo_articles)

        # 海外ニュースサイト
        if include_international and len(all_articles) < max_articles:
            # CNBC（メイン - 取得実績あり）
            cnbc_articles = self.scrape_cnbc(max_articles=max_articles - len(all_articles))
            all_articles.extend(cnbc_articles)

        logger.info(f"Total {len(all_articles)} articles scraped from all sources")
        return all_articles[:max_articles]


def main():
    """テスト実行"""
    import logging
    import os

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    api_key = os.getenv("OPENAI_API_KEY")
    scraper = NewsScraper(api_key=api_key, use_openai_enhancement=False)

    articles = scraper.fetch_articles(max_articles=10, include_international=True)

    print(f"\n取得した記事数: {len(articles)}\n")

    # ソース別に集計
    sources = {}
    for article in articles:
        source = article['source']
        sources[source] = sources.get(source, 0) + 1

    print('ソース別記事数:')
    for source, count in sources.items():
        print(f'  {source}: {count}件')

    print('\n記事一覧:')
    for i, article in enumerate(articles, 1):
        print(f'{i}. [{article["source"]}] {article["title"][:70]}...')
        print(f'   URL: {article["url"]}')
        print(f'   本文: {len(article["content"])}文字')
        print()


if __name__ == "__main__":
    main()
