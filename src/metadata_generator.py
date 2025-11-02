"""
メタデータ生成モジュール
OpenAI GPT APIを使用して動画のタイトル、説明文、タグを生成
"""

import os
import logging
from typing import Dict, List
from openai import OpenAI
from datetime import datetime

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
        self, script_data: Dict, news_article: Dict
    ) -> Dict:
        """
        台本とニュース記事からメタデータを生成

        Args:
            script_data: 台本データ（ScriptGenerator.generate_script()の出力）
            news_article: ニュース記事データ

        Returns:
            メタデータ辞書
            {
                "title": str,  # YouTube用タイトル
                "description": str,  # 説明文
                "tags": List[str],  # タグリスト
                "generated_at": str,  # 生成日時
            }
        """
        logger.info("Generating metadata...")

        # プロンプトを構築
        prompt = self._build_prompt(script_data, news_article)

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

            # メタデータを解析
            metadata = self._parse_metadata_response(response_text)

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

    def _build_prompt(self, script_data: Dict, news_article: Dict) -> str:
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
        }

        prompt = f"""以下の経済ニュース解説動画のYouTube用メタデータを作成してください。

【動画情報】
台本タイトル: {script_data.get('title', '')}
台本内容:
{script_data.get('script', '')[:500]}...

キーワード: {', '.join(script_data.get('keywords', []))}

【元記事情報】
タイトル: {news_article.get('title', '')}
要約: {news_article.get('summary', '')}
ソース: {news_article.get('source', '')}
URL: {news_article.get('url', '')}

【要件】
1. タイトル:
   - 最大{title_max_length}文字
   - スタイル: {style_instructions.get(title_style, '')}
   - {"日付を含める" if include_date else "日付は含めない"}
   - 検索されやすいキーワードを含める
   - 視聴者の興味を引く内容

2. 説明文:
   - 動画の内容を簡潔に説明
   - 重要なポイントを箇条書きで
   - {"ハッシュタグを含める" if include_hashtags else ""}
   - {"チャプター情報を含める（タイムスタンプ付き）" if include_chapters else ""}
   - 視聴者にとっての価値を明確に
   - **必須**: 説明文の最後に「参考記事」セクションを設け、元記事のタイトルとURLを記載すること

3. タグ:
   - 関連性の高いタグを10-15個
   - 一般的なタグと具体的なタグのバランス

【出力形式】
以下の形式で出力してください:

## タイトル
[YouTube動画タイトル]

## 説明文
[YouTube動画説明文]

📰 参考記事
{news_article.get('title', '')}
{news_article.get('url', '')}

## タグ
[タグ1, タグ2, タグ3, ...]
"""

        return prompt

    def _parse_metadata_response(self, response_text: str) -> Dict:
        """
        Claude APIのレスポンスを解析

        Args:
            response_text: Claude APIからの生成テキスト

        Returns:
            メタデータ辞書
        """
        lines = response_text.split("\n")

        title = ""
        description = ""
        tags = []
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith("## タイトル") or line.startswith("##タイトル"):
                current_section = "title"
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
            elif current_section == "description":
                description += line + "\n"
            elif current_section == "tags":
                # カンマ区切りのタグを抽出
                tags = [t.strip() for t in line.split(",") if t.strip()]
                current_section = None

        return {
            "title": title.strip(),
            "description": description.strip(),
            "tags": tags,
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
    }

    # メタデータ生成
    generator = MetadataGenerator(config["metadata"])
    metadata = generator.generate_metadata(sample_script, sample_article)

    print("\n=== 生成されたメタデータ ===\n")
    print(f"タイトル:\n{metadata['title']}\n")
    print(f"説明文:\n{metadata['description']}\n")
    print(f"タグ: {', '.join(metadata['tags'])}")


if __name__ == "__main__":
    main()
