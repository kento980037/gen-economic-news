"""
経済ニュース取得モジュール
RSSフィード、NewsAPI、OpenAI Web Searchから最新の経済ニュースを取得
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import feedparser
import requests
from bs4 import BeautifulSoup
import pytz
import numpy as np
from openai import OpenAI
from openai_news_fetcher import OpenAINewsFetcher
from news_scraper import NewsScraper

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
        self.filter_multi_topic = config.get("filter_multi_topic_articles", True)

        # コンテキスト情報拡張の設定
        context_config = config.get("context_enhancement", {})
        self.search_historical = context_config.get("search_historical", True)
        self.historical_range_days = context_config.get("historical_range_days", 90)
        self.historical_min_age_days = context_config.get("historical_min_age_days", 30)

        # OpenAI クライアント（埋め込みベクトル用）
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            logger.warning("OPENAI_API_KEY not set, embedding-based similarity disabled")

        # News Scraper（Yahoo! Finance / CNBC から直接取得）- 最優先
        try:
            self.news_scraper = NewsScraper(api_key=self.openai_api_key, use_openai_enhancement=True)
        except Exception as e:
            logger.warning(f"NewsScraper initialization failed: {e}")
            self.news_scraper = None

        # OpenAI Web Search用のフェッチャー（フォールバック）
        try:
            self.openai_news_fetcher = OpenAINewsFetcher(api_key=self.openai_api_key)
        except Exception as e:
            logger.warning(f"OpenAINewsFetcher initialization failed: {e}")
            self.openai_news_fetcher = None

        # 埋め込みベクトルのキャッシュ
        self.embedding_cache = {}

        # 記事キャッシュ（重複fetch_news()を防ぐ）
        self._cached_articles = None

        # 話題性判定の設定
        self.use_trending_score = config.get("use_trending_score", False)

    def fetch_news(self) -> List[NewsArticle]:
        """
        複数のソースからニュースを取得

        Returns:
            NewsArticleオブジェクトのリスト
        """
        articles = []
        sources = self.config.get("sources", ["rss"])

        # 優先順位1: News Scraper（Yahoo! Finance / CNBC から直接取得）
        if self.news_scraper and len(articles) < self.max_articles:
            logger.info("Fetching news from News Scraper (Yahoo Finance / CNBC)...")
            scraped_articles = self._fetch_from_news_scraper()
            articles.extend(scraped_articles)

        # 優先順位2: 従来のRSSフィード（フォールバック）
        if "rss" in sources and len(articles) < self.max_articles:
            logger.info("Fetching news from RSS feeds...")
            rss_articles = self._fetch_from_rss()
            articles.extend(rss_articles)

        # 優先順位3: OpenAI Web Search（最後のフォールバック）
        if "openai" in sources and self.openai_news_fetcher and len(articles) < self.max_articles:
            logger.info("Fetching news from OpenAI Web Search (fallback)...")
            openai_articles = self._fetch_from_openai()
            articles.extend(openai_articles)

        if "newsapi" in sources and self.news_api_key:
            logger.info("Fetching news from NewsAPI...")
            newsapi_articles = self._fetch_from_newsapi()
            articles.extend(newsapi_articles)

        # 日付でソート（新しい順）
        articles.sort(key=lambda x: x.published_at, reverse=True)

        # 重複を削除（タイトルが類似しているものを除外）
        unique_articles = self._remove_duplicates(articles)

        # 本文が短すぎる記事を除外（Bloombergの有料記事など）
        min_content_length = 200  # 最小200文字
        articles_with_content = []
        for article in unique_articles:
            content_length = len(article.content or "")
            if content_length >= min_content_length:
                articles_with_content.append(article)
            else:
                logger.debug(
                    f"Skipping article with insufficient content ({content_length} chars): "
                    f"{article.title[:50]}..."
                )

        logger.info(
            f"Filtered {len(unique_articles) - len(articles_with_content)} articles "
            f"with insufficient content (< {min_content_length} chars)"
        )

        # 指定された数に制限
        return articles_with_content[: self.max_articles]

    def _fetch_from_rss(self) -> List[NewsArticle]:
        """RSSフィードからニュースを取得"""
        articles = []
        cutoff_time = datetime.now(pytz.UTC) - timedelta(hours=self.max_age_hours)
        rss_timeout = self.config.get("rss_timeout", 10)  # デフォルト10秒

        for feed_url in self.rss_feeds:
            try:
                logger.debug(f"Parsing RSS feed: {feed_url} (timeout: {rss_timeout}s)")

                # タイムアウト付きでRSSを取得
                import socket
                original_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(rss_timeout)

                feed = feedparser.parse(feed_url)

                socket.setdefaulttimeout(original_timeout)

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
                # タイムアウトをリセット
                try:
                    import socket
                    socket.setdefaulttimeout(None)
                except:
                    pass

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

            # クエリパラメータ（金融・市場に特化）
            params = {
                "apiKey": self.news_api_key,
                "q": "半導体 OR AI OR NVIDIA OR テクノロジー OR 株式市場 OR 金融 OR 投資",
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

    def _fetch_from_news_scraper(self) -> List[NewsArticle]:
        """NewsScraperから日本のニュースサイトを取得"""
        articles = []

        if not self.news_scraper:
            logger.warning("NewsScraper not initialized, skipping scraper source")
            return articles

        try:
            # 日本と海外のニュースサイトからスクレイピング
            scraped_news = self.news_scraper.fetch_articles(
                max_articles=self.max_articles,
                include_international=True  # 海外ニュースサイトも含める
            )

            # NewsArticleオブジェクトに変換
            for news in scraped_news:
                article = NewsArticle(
                    title=news["title"],
                    summary=news["summary"],
                    url=news["url"],
                    published_at=news["published_at"],
                    source=news["source"],
                    content=news["content"],
                )
                articles.append(article)

            logger.info(f"Fetched {len(articles)} articles from News Scraper")

        except Exception as e:
            logger.error(f"Error fetching from News Scraper: {e}")

        return articles

    def _fetch_from_openai(self) -> List[NewsArticle]:
        """OpenAI Web Searchからニュースを取得"""
        articles = []

        if not self.openai_news_fetcher:
            logger.warning("OpenAINewsFetcher not initialized, skipping OpenAI source")
            return articles

        try:
            # OpenAI設定を取得
            openai_config = self.config.get("openai", {})
            max_articles = openai_config.get("max_articles", 20)
            topics = openai_config.get("topics", None)

            # ニュースを取得
            openai_news = self.openai_news_fetcher.fetch_financial_news(
                date=None,  # 今日
                max_articles=max_articles,
                topics=topics
            )

            # NewsArticleオブジェクトに変換
            for news in openai_news:
                article = NewsArticle(
                    title=news["title"],
                    summary=news["summary"],
                    url=news["url"],
                    published_at=news["published_at"],
                    source=news["source"],
                    content=news["content"]
                )
                articles.append(article)

            logger.info(f"Fetched {len(articles)} articles from OpenAI Web Search")

        except Exception as e:
            logger.error(f"Error fetching from OpenAI Web Search: {e}")

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

    def _check_single_article_coherence(self, article: NewsArticle) -> bool:
        """
        1つの記事内にトピックが一貫しているかチェック

        Args:
            article: チェックする記事

        Returns:
            True: トピックが一貫している（単一トピック）
            False: 複数のトピックが混在している
        """
        if not self.openai_client:
            # OpenAI APIがない場合はチェックできないのでTrueを返す
            return True

        try:
            # GPTに記事を分析させ、トピック数を判定
            prompt = f"""以下のニュース記事を分析してください。

タイトル: {article.title}
要約: {article.summary[:500]}

【質問】
この記事は単一の明確なトピックについて書かれていますか？
それとも、複数の無関係なトピック（例：タリフ問題、医療政策、市政など）が混在していますか？

【回答形式】
以下のいずれかで答えてください：
- SINGLE: 単一の明確なトピックのみ
- MULTIPLE: 複数の無関係なトピックが混在

判断理由も1行で簡潔に説明してください。

回答:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "あなたはニュース記事のトピック分析の専門家です。記事が単一トピックか複数トピックかを判定してください。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )

            result = response.choices[0].message.content.strip()
            logger.debug(f"Topic coherence check for '{article.title[:50]}...': {result}")

            # "MULTIPLE"が含まれていれば複数トピック記事と判定
            if "MULTIPLE" in result.upper():
                logger.warning(f"Multi-topic article detected: {article.title[:60]}...")
                logger.warning(f"  Reason: {result}")
                return False

            return True

        except Exception as e:
            logger.warning(f"Failed to check article coherence: {e}")
            # エラー時は安全側に倒してTrueを返す
            return True

    def _filter_coherent_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """
        複数トピックが混在している記事を除外

        Args:
            articles: 記事のリスト

        Returns:
            単一トピックの記事のみのリスト
        """
        if not articles:
            return []

        # フィルタリングが無効な場合はそのまま返す
        if not self.filter_multi_topic:
            logger.info("Multi-topic filtering is disabled, skipping coherence check")
            return articles

        logger.info(f"Filtering articles for topic coherence ({len(articles)} articles)...")

        coherent_articles = []
        for article in articles:
            if self._check_single_article_coherence(article):
                coherent_articles.append(article)
            else:
                logger.info(f"Filtered out multi-topic article: {article.title[:60]}...")

        logger.info(f"Coherence filtering: {len(articles)} -> {len(coherent_articles)} articles")
        return coherent_articles

    def select_main_article_by_clustering(self, articles: List[NewsArticle]) -> tuple[NewsArticle, List[NewsArticle]]:
        """
        記事をクラスタリングして、最も多く報道されているトピックのメイン記事と関連記事を選択

        Args:
            articles: 記事のリスト

        Returns:
            (メイン記事, 同じクラスタの関連記事リスト)
        """
        if not articles:
            return None, []

        # 0. 複数トピックが混在している記事を除外
        coherent_articles = self._filter_coherent_articles(articles)

        if not coherent_articles:
            logger.warning("No coherent single-topic articles found, using original list")
            coherent_articles = articles

        if len(coherent_articles) == 1:
            return coherent_articles[0], []

        logger.info(f"Clustering {len(coherent_articles)} coherent articles to find the most important topic...")

        # 1. 全記事の埋め込みベクトルを取得
        article_texts = [f"{a.title} {a.summary}" for a in coherent_articles]
        embeddings = self._get_embeddings_batch(article_texts)

        # 埋め込み取得に失敗した場合は最新記事を返す
        valid_embeddings = [e for e in embeddings if e is not None]
        if len(valid_embeddings) < 2:
            logger.warning("Not enough embeddings for clustering, using latest article")
            return coherent_articles[0], []

        # 2. 類似度マトリクスを計算
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            if embeddings[i] is None:
                continue
            for j in range(i + 1, n):
                if embeddings[j] is None:
                    continue
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim

        # 3. 改善されたクラスタリング（より厳格な閾値と一貫性チェック）
        # 類似度が0.75以上の記事を同じクラスタとする（0.6 → 0.75に引き上げ）
        threshold = 0.75
        clusters = []
        assigned = set()

        for i in range(n):
            if i in assigned or embeddings[i] is None:
                continue

            # 新しいクラスタを作成
            cluster = [i]
            assigned.add(i)

            # 類似した記事を同じクラスタに追加
            for j in range(n):
                if j not in assigned and embeddings[j] is not None:
                    if similarity_matrix[i][j] >= threshold:
                        cluster.append(j)
                        assigned.add(j)

            clusters.append(cluster)

        # 4. クラスタの一貫性をチェック（複数トピック混在を防ぐ）
        valid_clusters = []
        for cluster in clusters:
            if len(cluster) == 1:
                valid_clusters.append(cluster)
                continue

            # クラスタ内の全ペア間の平均類似度を計算
            similarities = []
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    similarities.append(similarity_matrix[cluster[i]][cluster[j]])

            avg_similarity = np.mean(similarities) if similarities else 0

            # 平均類似度が0.70以上なら一貫性のあるクラスタと判断
            # （個別ペアは0.75以上だが、全体平均は少し緩和）
            coherence_threshold = 0.70
            if avg_similarity >= coherence_threshold:
                valid_clusters.append(cluster)
                logger.info(f"Coherent cluster found: {len(cluster)} articles, avg similarity: {avg_similarity:.3f}")
            else:
                # 一貫性がない場合は分割（最も類似度の高い記事だけ残す）
                logger.warning(f"Incoherent cluster detected (avg sim: {avg_similarity:.3f}), splitting...")
                # 各記事を単独クラスタにする
                for idx in cluster:
                    valid_clusters.append([idx])

        # 5. 最大の一貫性のあるクラスタを選択
        # サイズが2以上のクラスタを優先（単独記事は除外）
        multi_article_clusters = [c for c in valid_clusters if len(c) >= 2]

        if multi_article_clusters:
            largest_cluster = max(multi_article_clusters, key=len)
            logger.info(f"Found {len(valid_clusters)} clusters. Selected coherent cluster with {len(largest_cluster)} articles:")
        else:
            # 一貫性のあるクラスタがない場合は最新記事を選択
            logger.warning("No coherent multi-article clusters found, using latest article")
            return coherent_articles[0], []

        for idx in largest_cluster:
            logger.info(f"  - {coherent_articles[idx].source}: {coherent_articles[idx].title[:60]}...")

        # 6. クラスタ内で最新の記事をメインに選ぶ
        cluster_articles = [coherent_articles[idx] for idx in largest_cluster]
        cluster_articles.sort(key=lambda a: a.published_at, reverse=True)

        main_article = cluster_articles[0]
        related_articles = cluster_articles[1:]  # 残りを関連記事に

        logger.info(f"Selected main article: {main_article.source} - {main_article.title[:60]}...")
        logger.info(f"Related articles in same cluster: {len(related_articles)}")

        return main_article, related_articles

    def get_top_news(self) -> Optional[NewsArticle]:
        """
        最も重要なニュースを1つ取得

        Returns:
            最新のニュース記事、または None
        """
        articles = self.fetch_news()
        return articles[0] if articles else None

    def get_related_articles(self, main_article: NewsArticle, max_related: int = 10) -> List[NewsArticle]:
        """
        メイン記事に関連する記事を取得（OpenAI生成、実際のURLを使用）

        Args:
            main_article: メイン記事
            max_related: 取得する関連記事の最大数（デフォルト10件）

        Returns:
            関連記事のリスト
        """
        logger.info(f"Finding {max_related} related articles for: {main_article.title[:50]}...")

        if not self.openai_news_fetcher:
            logger.warning("OpenAINewsFetcher not available, returning empty list")
            return []

        try:
            # OpenAI APIを使って重要キーワードを抽出
            keywords = self._extract_important_keywords(main_article, max_keywords=5)

            if not keywords:
                logger.warning("No keywords extracted from main article")
                return []

            logger.info(f"Searching with keywords: {', '.join(keywords[:3])}...")

            # OpenAI APIで関連記事を検索（実際のURLを含むプロンプト使用）
            openai_articles = self.openai_news_fetcher.search_related_articles(
                main_article_title=main_article.title,
                main_article_summary=main_article.summary,
                keywords=keywords,
                max_articles=max_related,
            )

            # NewsArticleオブジェクトに変換
            articles = []
            for article_dict in openai_articles:
                # メイン記事と同じタイトルはスキップ
                if article_dict["title"] == main_article.title:
                    continue

                # 本文が短すぎる記事を除外（200文字未満）
                content_length = len(article_dict.get("content", ""))
                if content_length < 200:
                    logger.debug(f"Skipping article with insufficient content ({content_length} chars): {article_dict['title'][:50]}...")
                    continue

                article = NewsArticle(
                    title=article_dict["title"],
                    summary=article_dict["summary"],
                    url=article_dict["url"],
                    published_at=article_dict["published_at"],
                    source=article_dict["source"],
                    content=article_dict["content"],
                )
                articles.append(article)

            # 上位max_related件のみ返す
            articles = articles[:max_related]

            logger.info(f"Found {len(articles)} related articles:")
            for article in articles:
                logger.info(f"  - {article.source}: {article.title[:60]}...")

            return articles

        except Exception as e:
            logger.error(f"Error searching related articles: {e}")
            return []


    def _extract_keywords(self, text: str) -> set:
        """
        テキストからキーワードを抽出（後方互換性のため残す・シンプル版）

        Args:
            text: 対象テキスト

        Returns:
            キーワードのセット
        """
        # 一般的なストップワードを除外
        stopwords = {
            "の", "に", "は", "を", "が", "と", "で", "も", "へ", "から", "まで",
            "より", "など", "について", "による", "により", "において",
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or"
        }

        # 単語に分割（簡易版）
        import re
        words = re.findall(r'\w+', text.lower())

        # ストップワードを除外
        keywords = {word for word in words if word not in stopwords and len(word) > 2}

        return keywords

    def _extract_important_keywords(self, main_article: NewsArticle, max_keywords: int = 5) -> List[str]:
        """
        OpenAI APIを使ってメイン記事から「情報を補足すべき重要キーワード」を抽出

        【目的】
        関連記事検索のため、メイン記事で言及されているが詳細が不足している
        重要な要素を特定し、それらについて補足情報を集める

        Args:
            main_article: メイン記事
            max_keywords: 抽出する最大キーワード数

        Returns:
            重要キーワードのリスト（重要度順）
        """
        if not self.openai_client:
            logger.warning("OpenAI client not available, falling back to simple extraction")
            # フォールバック: 簡易抽出
            text = f"{main_article.title} {main_article.summary}"
            keywords = self._extract_keywords(text)
            return sorted(keywords, key=len, reverse=True)[:max_keywords]

        try:
            logger.info("Extracting important keywords via OpenAI...")

            # 本文があれば使用、なければ要約を使用
            article_content = main_article.content if main_article.content and len(main_article.content) > 100 else main_article.summary

            prompt = f"""以下の金融ニュース記事を分析し、「情報を補足すべき重要キーワード」を{max_keywords}個抽出してください。

【記事情報】
タイトル: {main_article.title}
本文: {article_content[:1500]}

【抽出基準】
この記事を理解するために、追加の背景情報や詳細な解説が必要な要素を特定してください：

1. **企業名・組織名**
   - 例: 「NVIDIA」「トヨタ自動車」「日本銀行」「FRB」
   - 記事で言及されているが、その企業の事業内容・業績・戦略などの背景情報が不足している場合

2. **専門用語・経済概念**
   - 例: 「量的緩和」「PER」「AI半導体」「サプライチェーン」
   - 記事で使われているが、その意味や仕組みの詳しい説明がない場合

3. **経済指標・統計データ**
   - 例: 「CPI」「GDP」「失業率」「PMI」
   - 記事で数字が出ているが、その指標の意味や重要性の説明が不足している場合

4. **政策・制度・規制**
   - 例: 「ゼロ金利政策」「インボイス制度」「関税政策」
   - 記事で触れられているが、その背景や影響の詳細が不明な場合

5. **人物名（役職付き）**
   - 例: 「パウエルFRB議長」「植田日銀総裁」「イーロン・マスク」
   - 記事で名前が出ているが、その人物の経歴や立場の説明が不足している場合

6. **製品・サービス・技術**
   - 例: 「ChatGPT」「iPhone 15」「自動運転技術」
   - 記事で言及されているが、その詳細や市場への影響が不明な場合

【重要】
- 一般的すぎる単語（「市場」「株価」「経済」「企業」「投資」など）は避ける
- 記事で既に十分に説明されている要素は除外する
- 具体的で検索可能なキーワードを選ぶ
- 関連記事を検索する際に有用なキーワードを選ぶ
- 視聴者にとって「もっと詳しく知りたい」と思える要素を優先

【出力形式】
各行に1つずつキーワードを出力してください（説明は不要）。
重要度順に並べてください。

例:
NVIDIA
量的緩和政策
AI半導体市場
パウエルFRB議長
CPI（消費者物価指数）
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは金融ニュース分析の専門家です。記事から「情報を補足すべき重要キーワード」を抽出してください。"
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3,
            )

            result = response.choices[0].message.content.strip()
            keywords = [line.strip() for line in result.split('\n') if line.strip()]

            # 空行やコメントを除外
            keywords = [kw for kw in keywords if kw and not kw.startswith('#') and not kw.startswith('//')]

            logger.info(f"Extracted {len(keywords)} important keywords: {', '.join(keywords[:3])}...")

            return keywords[:max_keywords]

        except Exception as e:
            logger.error(f"Error extracting important keywords: {e}")
            # フォールバック: 簡易抽出
            text = f"{main_article.title} {main_article.summary}"
            keywords = self._extract_keywords(text)
            return sorted(keywords, key=len, reverse=True)[:max_keywords]

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        テキストの埋め込みベクトルを取得

        Args:
            text: 埋め込みを取得するテキスト

        Returns:
            埋め込みベクトル（numpy配列）、またはNone
        """
        if not self.openai_client:
            return None

        # キャッシュチェック
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        try:
            # OpenAI Embeddings APIを呼び出し
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",  # コスト効率の良いモデル
                input=text
            )

            # 埋め込みベクトルを取得
            embedding = np.array(response.data[0].embedding)

            # キャッシュに保存
            self.embedding_cache[text] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def _get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        複数テキストの埋め込みベクトルを一括取得（高速化）

        Args:
            texts: 埋め込みを取得するテキストのリスト

        Returns:
            埋め込みベクトルのリスト
        """
        if not self.openai_client or not texts:
            return [None] * len(texts)

        results = []
        uncached_texts = []
        uncached_indices = []

        # キャッシュをチェック
        for i, text in enumerate(texts):
            if text in self.embedding_cache:
                results.append(self.embedding_cache[text])
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # キャッシュにないテキストをバッチで取得
        if uncached_texts:
            try:
                logger.info(f"Fetching {len(uncached_texts)} embeddings in batch")
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=uncached_texts
                )

                # 結果を格納
                for i, data in enumerate(response.data):
                    embedding = np.array(data.embedding)
                    text = uncached_texts[i]
                    original_index = uncached_indices[i]

                    # キャッシュに保存
                    self.embedding_cache[text] = embedding
                    results[original_index] = embedding

            except Exception as e:
                logger.error(f"Error getting batch embeddings: {e}")

        return results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        2つのベクトル間のコサイン類似度を計算

        Args:
            vec1: ベクトル1
            vec2: ベクトル2

        Returns:
            コサイン類似度（-1.0～1.0、実際は0.0～1.0の範囲）
        """
        # ゼロベクトルチェック
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # コサイン類似度 = (A・B) / (||A|| * ||B||)
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)

        return float(similarity)

    def _calculate_relevance(self, main_keywords: set, article: NewsArticle) -> float:
        """
        記事の関連性スコアを計算（埋め込みベクトル使用）

        Args:
            main_keywords: メイン記事のキーワード（後方互換性のため残す）
            article: 比較対象の記事

        Returns:
            関連性スコア（0.0～1.0）
        """
        # 埋め込みベクトルが利用可能な場合はそれを使用
        if self.openai_client:
            # メイン記事のテキスト（キーワードを文字列に変換）
            main_text = " ".join(main_keywords)

            # 比較記事のテキスト
            article_text = f"{article.title} {article.summary}"

            # 埋め込みベクトルを取得
            main_embedding = self._get_embedding(main_text)
            article_embedding = self._get_embedding(article_text)

            if main_embedding is not None and article_embedding is not None:
                # コサイン類似度を計算
                similarity = self._cosine_similarity(main_embedding, article_embedding)
                logger.debug(f"Embedding similarity with '{article.title[:50]}...': {similarity:.3f}")
                return similarity

        # フォールバック: Jaccard係数（埋め込みが使えない場合）
        logger.debug("Using fallback Jaccard similarity")
        article_keywords = self._extract_keywords(article.title + " " + article.summary)

        common_keywords = main_keywords & article_keywords

        if not main_keywords or not article_keywords:
            return 0.0

        jaccard = len(common_keywords) / len(main_keywords | article_keywords)

        return jaccard


    def calculate_trending_score(self, article: NewsArticle) -> float:
        """
        OpenAI APIを使って記事の話題性スコアを計算

        Args:
            article: 評価する記事

        Returns:
            話題性スコア（0.0～10.0）
        """
        if not self.openai_client:
            logger.warning("OpenAI client not initialized, returning default score")
            return 5.0

        try:
            prompt = f"""
以下の金融ニュース記事の「話題性」を0から10のスコアで評価してください。

【評価基準】
- 市場への影響度（株価、為替、金利への影響）
- 注目度（投資家が注目しているか）
- 緊急性・速報性
- 金融市場での重要性
- グローバルな影響

【記事情報】
タイトル: {article.title}
要約: {article.summary[:300]}
ソース: {article.source}

数字のみを返してください（例: 7.5）
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "あなたは金融市場の専門家です。記事の話題性を客観的に評価してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.3,
            )

            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            logger.debug(f"Trending score for '{article.title[:50]}...': {score}")
            return max(0.0, min(10.0, score))  # 0-10の範囲に制限

        except Exception as e:
            logger.error(f"Error calculating trending score: {e}")
            return 5.0  # エラー時はデフォルトスコア

    def select_article_by_trending_score(self, articles: List[NewsArticle]) -> NewsArticle:
        """
        話題性スコアで記事を選択

        Args:
            articles: 記事リスト

        Returns:
            最も話題性の高い記事
        """
        if not articles:
            return None

        if len(articles) == 1:
            return articles[0]

        logger.info(f"Calculating trending scores for all {len(articles)} articles...")

        # 全記事の話題性スコアを計算
        scored_articles = []
        for idx, article in enumerate(articles, 1):
            score = self.calculate_trending_score(article)
            scored_articles.append((article, score))
            logger.info(f"  [{idx}/{len(articles)}] {score:.1f}/10 - {article.title[:60]}...")

        # スコアでソート（降順）
        scored_articles.sort(key=lambda x: x[1], reverse=True)

        best_article = scored_articles[0][0]
        best_score = scored_articles[0][1]
        logger.info(f"Selected article with highest score ({best_score:.1f}/10): {best_article.title}")

        return best_article

    def get_related_context_openai(self, main_article: NewsArticle) -> str:
        """
        OpenAI APIを使って関連コンテキストを取得（検索機能付き）

        Args:
            main_article: メイン記事

        Returns:
            関連コンテキスト情報（マークダウン形式）
        """
        if not self.openai_client:
            logger.warning("OpenAI client not initialized")
            return ""

        try:
            logger.info(f"Fetching related context via OpenAI for: {main_article.title[:50]}...")

            prompt = f"""
以下の金融ニュース記事について、投資家向けポッドキャストの台本作成に必要な追加情報を収集してください。

【メイン記事】
タイトル: {main_article.title}
要約: {main_article.summary}
ソース: {main_article.source}
公開日: {main_article.published_at.strftime('%Y年%m月%d日')}

【収集する情報】
1. **背景・文脈**: なぜこのニュースが重要なのか、どういう経緯でこうなったか
2. **過去の類似ケース**: 過去に似た状況があれば、その時どうなったか
3. **市場への影響**: 株価、為替、金利、投資判断への影響
4. **専門家の見解**: アナリストや経済学者の意見（あれば）
5. **関連する統計データ**: 具体的な数字（GDP、失業率、株価指数など）
6. **今後の見通し**: 予想される展開や注目ポイント

【出力形式】
マークダウン形式で、見出しごとに整理してください。
情報源がある場合は明記してください。
不明な点は無理に書かないでください。

重要: 投資家にとって実践的で、台本作成に役立つ情報を優先してください。
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # コスト効率重視（情報収集には十分）
                messages=[
                    {"role": "system", "content": "あなたは金融市場の専門家です。正確で実践的な情報を提供してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3,
            )

            context = response.choices[0].message.content
            logger.info(f"Retrieved {len(context)} characters of context from OpenAI")

            return context

        except Exception as e:
            logger.error(f"Error fetching context from OpenAI: {e}")
            return ""


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
