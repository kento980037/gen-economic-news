"""
メタデータ生成モジュール
OpenAI GPT APIを使用して動画のタイトル、説明文、タグを生成
"""

import os
import logging
from typing import Dict, List
from openai import OpenAI
from datetime import datetime
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class MetadataGenerator:
    """メタデータ生成クラス"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書（config.yamlから読み込んだmetadata設定）
        """
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        self.model = config.get("openai_model", "gpt-4o-mini")
        self.title_config = config.get("title", {})
        self.description_config = config.get("description", {})
        self.tags_config = config.get("tags", {})

    def generate_metadata(
        self, script_data: Dict, news_article: Dict, related_articles: List[Dict] = None
    ) -> Dict:
        """
        台本とニュース記事からメタデータを生成

        Args:
            script_data: 台本データ（ScriptGenerator.generate_script()の出力）
            news_article: メイン記事データ
            related_articles: 関連記事データのリスト（オプション）

        Returns:
            メタデータ辞書
            {
                "title": str,  # YouTube用タイトル
                "description": str,  # 説明文（参考記事セクションを含む）
                "tags": List[str],  # タグリスト
                "references": Dict,  # 参考記事情報（main_article, related_articles）
                "generated_at": str,  # 生成日時
            }
        """
        logger.info("Generating metadata...")

        # プロンプトを構築
        prompt = self._build_prompt(script_data, news_article, related_articles or [])

        # プロンプトをログに出力
        logger.info("=" * 80)
        logger.info("METADATA GENERATION PROMPT")
        logger.info("=" * 80)
        logger.info(f"[PROMPT]\n{prompt}")
        logger.info("=" * 80)

        try:
            # OpenAI APIを呼び出し
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.8,
            )

            # レスポンスからテキストを抽出
            response_text = response.choices[0].message.content

            # メタデータを解析（記事データを渡してURL修正を行う）
            metadata = self._parse_metadata_response(response_text, news_article, related_articles)

            # デフォルトタグを追加
            default_tags = self.tags_config.get("default_tags", [])
            for tag in default_tags:
                if tag not in metadata["tags"]:
                    metadata["tags"].append(tag)

            # タグ数を制限
            max_tags = self.tags_config.get("max_count", 15)
            metadata["tags"] = metadata["tags"][:max_tags]

            logger.info(f"Metadata generated: {metadata['title']}")
            return metadata

        except Exception as e:
            logger.error(f"Error generating metadata: {e}")
            raise

    def _build_prompt(self, script_data: Dict, news_article: Dict, related_articles: List[Dict]) -> str:
        """プロンプトを構築"""
        title_max_length = self.title_config.get("max_length", 100)
        title_style = self.title_config.get("style", "clickbait_moderate")
        include_date = self.title_config.get("include_date", True)
        include_hashtags = self.description_config.get("include_hashtags", True)
        include_chapters = self.description_config.get("include_chapters", True)

        # スタイルの説明
        style_instructions = {
            "formal": "フォーマルで落ち着いたトーン",
            "clickbait_moderate": "興味を引くが誇張しすぎない適度なキャッチーさ",
            "clickbait_strong": "強い興味喚起を重視したキャッチーなスタイル",
            "financial_podcast": "金融ポッドキャスト向け：投資家の注目を集めつつ、信頼性も保つバランス型。数字・緊急性・ベネフィットを組み合わせたクリック率重視スタイル",
        }

        # 関連記事情報を構築
        related_articles_text = ""
        if related_articles:
            related_articles_text = "\n【関連記事情報】\n"
            related_articles_text += f"この動画では、メイン記事に加えて{len(related_articles)}件の関連記事を参照しています。\n\n"
            for i, article in enumerate(related_articles, 1):
                related_articles_text += f"{i}. {article.get('title', '')}\n"
                related_articles_text += f"   ソース: {article.get('source', '')}\n"
                related_articles_text += f"   URL: {article.get('url', '')}\n\n"

        # 参考記事セクションの例を構築
        reference_section_example = f"""
📰 参考記事

【メイン記事】
{news_article.get('title', '')}
出典: {news_article.get('source', '')}
{news_article.get('url', '')}
"""

        if related_articles:
            reference_section_example += "\n【関連記事】\n"
            for i, article in enumerate(related_articles, 1):
                reference_section_example += f"{i}. {article.get('title', '')}\n"
                reference_section_example += f"   出典: {article.get('source', '')}\n"
                reference_section_example += f"   {article.get('url', '')}\n\n"

        prompt = f"""以下の経済ニュース解説動画のYouTube用メタデータを作成してください。

【動画情報】
台本タイトル: {script_data.get('title', '')}
台本内容:
{script_data.get('script', '')[:500]}...

キーワード: {', '.join(script_data.get('keywords', []))}

【メイン記事情報】
タイトル: {news_article.get('title', '')}
要約: {news_article.get('summary', '')}
ソース: {news_article.get('source', '')}
URL: {news_article.get('url', '')}
{related_articles_text}
【要件】
1. 動画タイトル（金融ニュースポッドキャスト向け・超重要）:
   - 文字数: 15〜35文字（スマホで一瞬で伝わる長さ）
   - スタイル: {style_instructions.get(title_style, '')}
   - {"日付を含める" if include_date else "日付は含めない"}
   - SEO・検索を意識した詳しめのタイトル

   **【YouTubeクリック率を最大化する必須テクニック】**
   以下の要素を必ず組み合わせること：

   ✅ **1. 具体的な数字を入れる**
   - 「3つの理由」「10%上昇」「5分でわかる」など

   ✅ **2. ターゲットを明確にする（投資家向け）**
   - 「投資家必見」「個人投資家向け」「初心者でもわかる」
   - 「株式投資家へ」「為替トレーダー注目」など

   ✅ **3. ベネフィット（得られる結果）を明示**
   - 「これで分かる〇〇」「知れば得する△△」
   - 「今すぐ確認すべき〇〇」など

   ✅ **4. 強めワード・緊急性ワードを1つ入れる**
   - 「速報」「緊急解説」「衝撃」「急騰/急落」
   - 「注目」「重要」「警戒」「転換点」
   - 「知らないと損」「見逃せない」など

   ✅ **5. 疑問形も効果的**
   - 「なぜ〇〇が急騰？」「本当に買い時？」

   ✅ **6. 金融特有のキーワードを含める**
   - 「日銀」「FRB」「金利」「株価」「為替」
   - 「ドル円」「日経平均」「利上げ/利下げ」
   - 「市場」「相場」「投資判断」など

   **【NGパターン】**
   - ❌ 専門用語だけで一般人に伝わらない
   - ❌ 長すぎて何が言いたいか不明
   - ❌ 平凡で他の動画と差別化できない

   **【タイトルテンプレート例（金融向け）】**
   - 「【速報】〇〇が△％急騰！投資家が知るべき3つの理由」
   - 「日銀の〇〇発表で市場激変！今後の展開を5分解説」
   - 「投資家必見：〇〇ショックの真相と対策」
   - 「なぜ今〇〇が注目？初心者にもわかる市場分析」
   - 「【緊急】〇〇で相場転換！？知らないと損する影響」

2. サムネイル用タイトル（超重要・クリック率に直結）:
   **2段構成で出力してください：**
   - **メインテキスト（大きい文字）**: 5〜8文字（最もインパクトのある言葉）
   - **サブテキスト（小さい文字）**: 8〜15文字（補足情報）

   **【2段構成サムネイルの作り方】**
   ✅ **メイン（大）+ サブ（小）の組み合わせ例**

   パターン1：数字 + 詳細
   - 動画タイトル：「【速報】NVIDIA株が15%急騰！投資家が知るべき3つの理由」
   - サムネイル：
     - メイン：「15%急騰」（大きく）
     - サブ：「NVIDIA株の真相」（小さく）

   パターン2：企業名 + 出来事
   - 動画タイトル：「日銀の金利引き上げで市場激変！今後の展開を5分解説」
   - サムネイル：
     - メイン：「日銀利上げ」（大きく）
     - サブ：「市場への影響は？」（小さく）

   パターン3：疑問形 + 答え
   - 動画タイトル：「なぜ今テスラが注目？初心者にもわかる市場分析」
   - サムネイル：
     - メイン：「テスラ急騰」（大きく）
     - サブ：「今買うべき？」（小さく）

   パターン4：結論 + 理由
   - 動画タイトル：「円高が止まらない！FRBの政策転換が原因か」
   - サムネイル：
     - メイン：「円高加速」（大きく）
     - サブ：「FRB政策転換」（小さく）

   **【メインテキストに入れるべき要素】**
   - 数字（「15%」「3つの理由」「10分」）
   - 企業名・通貨名（「NVIDIA」「ドル円」「日経平均」）
   - 強めワード（「急騰」「暴落」「激変」「速報」）
   - 疑問形（「なぜ？」「本当？」）

   **【サブテキストに入れるべき要素】**
   - 詳細情報（「市場の反応」「投資家への影響」）
   - 疑問形（「どうなる？」「買い時？」）
   - ターゲット（「投資家必見」「初心者向け」）
   - 期間・タイミング（「2025年」「今週の展開」）

   **【NGパターン】**
   - ❌ メインとサブが同じ内容（情報の重複）
   - ❌ 両方とも長すぎて読めない
   - ❌ 抽象的すぎて何の話か分からない
   - ❌ メインが弱くサブが強い（視覚的に逆転）

3. 説明文:
   - 動画の内容を簡潔に説明
   - 重要なポイントを箇条書きで
   - {"ハッシュタグを含める" if include_hashtags else ""}
   - 視聴者にとっての価値を明確に
   - **【注意】**: 免責事項と参考記事セクションは自動で追加されるため、説明文には含めないでください
   - 説明文は動画の内容説明のみに集中してください
   - 説明文は1つのセクションとして完結させ、途中で##見出しを使わないこと

4. タグ:
   - 関連性の高いタグを10-15個
   - 一般的なタグと具体的なタグのバランス

【出力形式】
以下の形式で厳密に出力してください:

## タイトル
[YouTube動画タイトル（15-35文字、詳しめ）]

## サムネイル_メイン
[メインテキスト（5-8文字、大きい文字で表示される）]

## サムネイル_サブ
[サブテキスト（8-15文字、小さい文字で表示される）]

## 説明文
[YouTube動画説明文の本文のみ]
※免責事項と参考記事セクションは自動で追加されます

## タグ
[タグ1, タグ2, タグ3, ...]

【重要】
- サムネイルはメインとサブの2つに分けて出力してください
- メインテキストは最もインパクトのある5-8文字
- サブテキストは補足情報の8-15文字
- メインとサブで情報が重複しないようにしてください
- 説明文には動画の内容説明のみを記載してください（免責事項と参考記事は自動追加されます）
"""

        return prompt

    def _validate_url(self, url: str, timeout: int = 5) -> bool:
        """
        URLが有効かどうかを検証

        Args:
            url: 検証するURL
            timeout: タイムアウト時間（秒）

        Returns:
            URLが有効な場合True、無効な場合False
        """
        if not url or not url.startswith("http"):
            return False

        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return False

            response = requests.head(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )

            return 200 <= response.status_code < 400

        except Exception:
            return False

    def _fix_reference_urls(self, description: str, news_article: Dict = None, related_articles: List[Dict] = None) -> str:
        """
        説明文内の参考記事セクションのURLを実際のURLで置き換え

        OpenAI APIがプレースホルダーURLを生成してしまう問題を修正するため、
        参考記事セクションを完全に再構築する

        Args:
            description: 元の説明文
            news_article: メイン記事データ
            related_articles: 関連記事データのリスト

        Returns:
            修正された説明文
        """
        # 参考記事セクションを探す
        ref_section_markers = ["📰 参考記事", "参考記事", "## 参考記事"]
        ref_start_idx = -1

        for marker in ref_section_markers:
            idx = description.find(marker)
            if idx != -1:
                ref_start_idx = idx
                break

        # 参考記事セクションが見つからない場合は、説明文の末尾に追加
        if ref_start_idx == -1:
            logger.warning("Reference section not found in description, appending it")
            # 説明文の本文部分を保持
            base_description = description.strip()
        else:
            # 参考記事セクションより前の部分を保持
            base_description = description[:ref_start_idx].strip()

        # 注意文を追加
        disclaimer = "\n\n⚠️ 免責事項\n"
        disclaimer += "本動画は情報提供のみを目的としており、特定の銘柄や投資行動を推奨するものではありません。投資に関する判断はご自身の責任でお願いします。\n"
        disclaimer += "また、本動画で扱う情報は信頼できるデータに基づいていますが、その正確性および完全性を保証するものではありません。\n"

        # 参考記事セクションを再構築
        ref_section = "\n\n📰 参考記事\n\n"

        # メイン記事の情報を追加
        if news_article:
            ref_section += "【メイン記事】\n"
            ref_section += f"{news_article.get('title', '')}\n"
            ref_section += f"出典: {news_article.get('source', '')}\n"
            ref_section += f"{news_article.get('url', '')}\n"

        # 関連記事の情報を追加
        if related_articles and len(related_articles) > 0:
            ref_section += "\n【関連記事】\n"
            for i, article in enumerate(related_articles, 1):
                ref_section += f"{i}. {article.get('title', '')}\n"
                ref_section += f"   出典: {article.get('source', '')}\n"
                ref_section += f"   {article.get('url', '')}\n\n"

            logger.info(f"Included {len(related_articles)} related articles in metadata")

        # 説明文、注意文、参考記事を結合
        return base_description + disclaimer + ref_section

    def _parse_metadata_response(self, response_text: str, news_article: Dict = None, related_articles: List[Dict] = None) -> Dict:
        """
        Claude APIのレスポンスを解析

        Args:
            response_text: Claude APIからの生成テキスト
            news_article: メイン記事データ（URL修正用）
            related_articles: 関連記事データ（URL修正用）

        Returns:
            メタデータ辞書
        """
        lines = response_text.split("\n")

        title = ""
        thumbnail_main = ""
        thumbnail_sub = ""
        description = ""
        tags = []
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith("## タイトル") or line.startswith("##タイトル"):
                current_section = "title"
                continue
            elif line.startswith("## サムネイル_メイン") or line.startswith("##サムネイル_メイン") or line.startswith("## サムネイルメイン"):
                current_section = "thumbnail_main"
                continue
            elif line.startswith("## サムネイル_サブ") or line.startswith("##サムネイル_サブ") or line.startswith("## サムネイルサブ"):
                current_section = "thumbnail_sub"
                continue
            elif line.startswith("## 説明文") or line.startswith("##説明文"):
                current_section = "description"
                continue
            elif (
                line.startswith("## タグ")
                or line.startswith("##タグ")
                or line.startswith("## tag")
            ):
                current_section = "tags"
                continue
            elif line.startswith("##"):
                current_section = None
                continue

            if not line or line.startswith("#"):
                continue

            if current_section == "title":
                title = line
                current_section = None
            elif current_section == "thumbnail_main":
                thumbnail_main = line
                current_section = None
            elif current_section == "thumbnail_sub":
                thumbnail_sub = line
                current_section = None
            elif current_section == "description":
                description += line + "\n"
            elif current_section == "tags":
                # カンマ区切りのタグを抽出
                tags = [t.strip() for t in line.split(",") if t.strip()]
                current_section = None

        # サムネイル用テキストがない場合は動画タイトルから生成
        if not thumbnail_main and title:
            # 簡易的に最初の8文字を抽出
            thumbnail_main = title[:8]
            logger.warning(f"Thumbnail main text not found, using truncated title: {thumbnail_main}")

        if not thumbnail_sub and title:
            # 簡易的に8文字目以降を抽出
            thumbnail_sub = title[8:20]
            logger.warning(f"Thumbnail sub text not found, using truncated title: {thumbnail_sub}")

        # 説明文の参考記事セクションを実際のURLで置き換え
        if news_article or related_articles:
            description = self._fix_reference_urls(description, news_article, related_articles)

        # 参考記事情報を別フィールドに保存
        references = {
            "main_article": news_article if news_article else None,
            "related_articles": related_articles if related_articles else []
        }

        return {
            "title": title.strip(),
            "thumbnail_main": thumbnail_main.strip(),
            "thumbnail_sub": thumbnail_sub.strip(),
            "description": description.strip(),
            "tags": tags,
            "references": references,
            "generated_at": datetime.now().isoformat(),
        }

    def optimize_title_for_seo(self, title: str, keywords: List[str]) -> str:
        """
        タイトルをSEO最適化

        Args:
            title: 元のタイトル
            keywords: 重要なキーワードリスト

        Returns:
            最適化されたタイトル
        """
        logger.info("Optimizing title for SEO")

        prompt = f"""以下のYouTube動画タイトルをSEO最適化してください。

【現在のタイトル】
{title}

【重要なキーワード】
{', '.join(keywords)}

【要件】
- 最大{self.title_config.get('max_length', 100)}文字
- 重要なキーワードを含める
- クリック率を高める魅力的な表現
- 検索されやすい言葉選び

【出力】
最適化されたタイトルのみを出力してください。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.8,
            )

            optimized_title = response.choices[0].message.content.strip()
            logger.info(f"Optimized title: {optimized_title}")
            return optimized_title

        except Exception as e:
            logger.error(f"Error optimizing title: {e}")
            return title  # エラー時は元のタイトルを返す

    def generate_hashtags(self, script_data: Dict, count: int = 10) -> List[str]:
        """
        ハッシュタグを生成

        Args:
            script_data: 台本データ
            count: 生成するハッシュタグ数

        Returns:
            ハッシュタグのリスト
        """
        logger.info(f"Generating {count} hashtags")

        prompt = f"""以下の動画台本から、SNSで使えるハッシュタグを{count}個生成してください。

【台本】
{script_data.get('script', '')[:500]}...

【要件】
- 日本語のハッシュタグ
- 動画の内容に関連性が高い
- トレンドに合ったもの
- 検索されやすいもの

【出力】
ハッシュタグをカンマ区切りで出力してください（#は不要）
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.8,
            )

            hashtags_text = response.choices[0].message.content.strip()
            hashtags = [tag.strip() for tag in hashtags_text.split(",") if tag.strip()]

            logger.info(f"Generated hashtags: {hashtags}")
            return hashtags

        except Exception as e:
            logger.error(f"Error generating hashtags: {e}")
            return []


def main():
    """テスト実行用"""
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    # 設定読み込み
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)

    # サンプルデータ
    sample_script = {
        "title": "日銀が政策金利を引き上げ、今後の経済への影響は?",
        "script": """こんにちは、経済ニュース解説チャンネルへようこそ。
今日は、日本銀行が発表した政策金利の引き上げについて詳しく解説します。
日銀は金融政策決定会合で、政策金利を0.25%から0.5%に引き上げることを決定しました。
この決定の背景には、インフレ率が目標の2%を上回る状況が続いていることがあります。""",
        "keywords": ["日銀", "政策金利", "金融政策", "インフレ", "経済"],
    }

    sample_article = {
        "title": "日銀、政策金利を0.5%に引き上げ決定",
        "summary": "日本銀行は金融政策決定会合で、政策金利を0.25%から0.5%に引き上げることを決定した。",
        "source": "経済新聞",
        "url": "https://example.com/article1"
    }

    # 関連記事のサンプル
    sample_related_articles = [
        {
            "title": "市場関係者、日銀の利上げを歓迎",
            "summary": "市場関係者は日銀の利上げ決定を歓迎している。",
            "source": "Bloomberg",
            "url": "https://example.com/article2"
        },
        {
            "title": "円相場、利上げ発表後に急伸",
            "summary": "日銀の利上げ発表を受け、円相場が対ドルで急伸した。",
            "source": "Reuters",
            "url": "https://example.com/article3"
        }
    ]

    # メタデータ生成
    generator = MetadataGenerator(config["metadata"])
    metadata = generator.generate_metadata(sample_script, sample_article, sample_related_articles)

    print("\n=== 生成されたメタデータ ===\n")
    print(f"タイトル:\n{metadata['title']}\n")
    print(f"説明文:\n{metadata['description']}\n")
    print(f"タグ: {', '.join(metadata['tags'])}")


if __name__ == "__main__":
    main()
