"""
台本生成モジュール
OpenAI GPT APIを使用してニュース記事から動画用の台本を生成
"""

import os
import logging
from typing import Dict, List, Optional
from openai import OpenAI
from datetime import datetime

logger = logging.getLogger(__name__)


class ScriptGenerator:
    """台本生成クラス"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書（config.yamlから読み込んだscript設定）
        """
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        self.model = config.get("openai_model", "gpt-4o-mini")
        self.min_length = config.get("min_length", 800)
        self.max_length = config.get("max_length", 2000)
        self.tone = config.get("tone", "professional")
        self.style = config.get("style", "narration")
        self.system_prompt = config.get("system_prompt", "")

    def generate_script(
        self,
        news_article: Dict,
        target_duration: int = 1080,
        additional_context: Optional[str] = None,
        max_retries: int = 2,
    ) -> Dict:
        """
        ニュース記事から台本を生成

        Args:
            news_article: ニュース記事の辞書（NewsArticle.to_dict()の出力）
            target_duration: 目標動画時間（秒）デフォルト1080秒=18分
            additional_context: 追加のコンテキスト情報
            max_retries: 文字数不足時の最大再試行回数

        Returns:
            生成された台本の辞書
            {
                "title": str,  # 動画タイトル
                "script": str,  # ナレーション台本
                "keywords": List[str],  # キーワードリスト
                "estimated_duration": int,  # 推定時間（秒）
                "generated_at": str,  # 生成日時
            }
        """
        logger.info(f"Generating script for: {news_article.get('title', 'Unknown')}")

        # 文字数を時間から計算（日本語: 1秒あたり約5-6文字、余裕を持って5文字）
        target_chars = target_duration * 5
        logger.info(f"Target duration: {target_duration} seconds, Target chars: {target_chars} characters")

        result = None

        for attempt in range(max_retries + 1):
            # プロンプトを構築
            user_prompt = self._build_prompt(
                news_article, target_chars, additional_context
            )

            # 初回以外は文字数不足を指摘
            if attempt > 0:
                logger.warning(f"Retry {attempt}/{max_retries}: Previous script was too short ({len(result['script'])} chars)")
                user_prompt = f"""前回の台本が短すぎました（{len(result['script'])}文字）。

**必須要件**: 必ず{target_chars}文字以上の台本を作成してください。

{user_prompt}

【重要】各セクションをもっと詳しく書いてください：
- 具体例を3つ以上追加
- 過去事例との比較を詳しく
- 数字やデータを豊富に使用
- 専門家の見解を複数引用
"""

            # プロンプトをログに出力
            if attempt == 0:
                logger.info("=" * 80)
                logger.info("SCRIPT GENERATION PROMPT")
                logger.info("=" * 80)
                if self.system_prompt:
                    logger.info(f"[SYSTEM PROMPT]\n{self.system_prompt}")
                    logger.info("-" * 80)
                logger.info(f"[USER PROMPT]\n{user_prompt}")
                logger.info("=" * 80)

            try:
                # OpenAI APIを呼び出し
                messages = [{"role": "user", "content": user_prompt}]
                if self.system_prompt:
                    messages.insert(0, {"role": "system", "content": self.system_prompt})

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=20000,  # 出力余裕を持たせる（約10,000文字分）
                    temperature=0.5,  # 指示に従いやすくする
                )

                # レスポンスからテキストを抽出
                script_text = response.choices[0].message.content

                # 台本を解析
                result = self._parse_script_response(script_text, news_article)

                script_length = len(result['script'])
                logger.info(f"Script generated. Length: {script_length} chars (target: {target_chars})")

                # 文字数チェック（目標の80%以上ならOK）
                if script_length >= target_chars * 0.8:
                    logger.info(f"✓ Script length is sufficient ({script_length}/{target_chars} chars)")
                    return result
                else:
                    logger.warning(f"✗ Script is too short ({script_length}/{target_chars} chars, {script_length/target_chars*100:.1f}%)")
                    if attempt < max_retries:
                        logger.info(f"Retrying generation (attempt {attempt + 2}/{max_retries + 1})...")
                    else:
                        logger.warning("Max retries reached. Returning script as-is.")
                        return result

            except Exception as e:
                logger.error(f"Error generating script: {e}")
                if attempt == max_retries:
                    raise
                logger.info(f"Retrying after error (attempt {attempt + 2}/{max_retries + 1})...")

        return result

    def generate_script_from_multiple_articles(
        self, news_articles: List[Dict], target_duration: int = 1080
    ) -> Dict:
        """
        複数のニュース記事から1つの台本を生成（まとめニュース形式）

        Args:
            news_articles: ニュース記事の辞書リスト
            target_duration: 目標動画時間（秒）デフォルト1080秒=18分

        Returns:
            生成された台本の辞書
        """
        logger.info(f"Generating script from {len(news_articles)} articles")

        target_chars = target_duration * 5

        # プロンプトを構築
        user_prompt = self._build_multi_article_prompt(news_articles, target_chars)

        try:
            messages = [{"role": "user", "content": user_prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=16384,  # 5400文字（日本語）の生成に必要（約8000-16000トークン）
                temperature=0.7,
            )

            script_text = response.choices[0].message.content
            result = self._parse_script_response(script_text)

            logger.info(
                f"Multi-article script generated. Length: {len(result['script'])} chars"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating multi-article script: {e}")
            raise

    def _build_prompt(
        self, news_article: Dict, target_chars: int, additional_context: Optional[str]
    ) -> str:
        """単一記事用のプロンプトを構築（簡潔版）"""

        # 本文を優先、なければ要約を使用
        article_content = news_article.get('content', '')
        if not article_content or len(article_content.strip()) < 50:
            article_content = news_article.get('summary', '')

        prompt = f"""⚠️ 最重要要件: 必ず{target_chars}文字以上の台本を作成してください ⚠️

【メイン記事】
タイトル: {news_article.get('title', '')}
本文: {article_content}
ソース: {news_article.get('source', '')}
公開日時: {news_article.get('published_at', '')}
"""

        if additional_context:
            prompt += f"\n【参考記事・コンテキスト】\n{additional_context}\n"

        prompt += f"""
【必須要件】
1. **文字数**: {target_chars}文字以上（目標時間18分）
2. **構成**: 2-4つのポイントで整理（各ポイント1200-1500文字）
3. **具体性**: 企業名、数字、日付を必ず明記（抽象表現禁止）
4. **専門用語**: 直後に説明を入れる（例: FRB（米国中央銀行））
5. **深掘り**: 背景、過去事例、専門家見解を詳しく

【構成例（各セクションの文字数を守ること）】
- オープニング: 200文字（結論先出し）
- ポイント1: 1200-1500文字（具体例・過去データ・専門家見解を豊富に）
- ポイント2: 1200-1500文字（同上）
- ポイント3: 1200-1500文字（必要に応じて）
- 今後の展開: 900-1200文字（短期・中長期予測）
- まとめ: 300文字
- CTA: 600文字（いいね・チャンネル登録）

⚠️ 各ポイントは必ず1200文字以上書いてください。
具体例が少ない場合は、過去事例、市場への影響、他国との比較などで補ってください。

【出力形式】
## タイトル
[15-35文字の投資家向けタイトル]

## 台本
[純粋なナレーション原稿のみ。セクション名は書かない。]

## キーワード
[カンマ区切りで5-10個]

⚠️ 最重要: 台本は{target_chars}文字以上必須。各ポイント1200文字以上書くこと。
"""

        return prompt

    def _build_multi_article_prompt(
        self, news_articles: List[Dict], target_chars: int
    ) -> str:
        """複数記事用のプロンプトを構築"""
        articles_text = ""
        for i, article in enumerate(news_articles, 1):
            articles_text += f"""
【記事{i}】
タイトル: {article.get('title', '')}
要約: {article.get('summary', '')}
ソース: {article.get('source', '')}

"""

        prompt = f"""以下の複数の経済ニュースを元に、今日の経済ニュースまとめ動画の台本を作成してください。

{articles_text}

【台本要件】
- 文字数: {target_chars}文字前後
- スタイル: ナレーション形式
- 構成: オープニング → 各ニュースの紹介（重要度順） → エンディング

【注意事項】
1. 各ニュースを簡潔に分かりやすく紹介
2. ニュース間の関連性があれば言及
3. 全体として統一感のある流れを作る
4. 最後に今日のポイントをまとめる

【出力形式】
以下の形式で出力してください:

## タイトル
[動画のタイトル]

## 台本
[ナレーション台本本文]

## キーワード
[重要なキーワードをカンマ区切りで]
"""

        return prompt

    def _parse_script_response(
        self, script_text: str, news_article: Optional[Dict] = None
    ) -> Dict:
        """
        Claude APIのレスポンスを解析して構造化

        Args:
            script_text: Claude APIからの生成テキスト
            news_article: 元のニュース記事（オプション）

        Returns:
            構造化された台本データ
        """
        lines = script_text.split("\n")

        title = ""
        script = ""
        keywords = []
        current_section = None

        for line in lines:
            line = line.strip()

            if line.startswith("## タイトル") or line.startswith("##タイトル"):
                current_section = "title"
                continue
            elif line.startswith("## 台本") or line.startswith("##台本"):
                current_section = "script"
                continue
            elif (
                line.startswith("## キーワード")
                or line.startswith("##キーワード")
                or line.startswith("## keyword")
            ):
                current_section = "keywords"
                continue
            elif line.startswith("##"):
                current_section = None
                continue

            if not line or line.startswith("#"):
                continue

            if current_section == "title":
                title = line
                current_section = None
            elif current_section == "script":
                script += line + "\n"
            elif current_section == "keywords":
                # カンマ区切りのキーワードを抽出
                keywords = [k.strip() for k in line.split(",") if k.strip()]
                current_section = None

        # タイトルが取得できなかった場合は元記事から
        if not title and news_article:
            title = news_article.get("title", "経済ニュース解説")

        # キーワードが取得できなかった場合は空リスト
        if not keywords:
            keywords = ["経済", "ビジネス", "ニュース"]

        # 推定時間を計算（日本語5文字/秒）
        estimated_duration = len(script.strip()) // 5

        return {
            "title": title.strip(),
            "script": script.strip(),
            "keywords": keywords,
            "estimated_duration": estimated_duration,
            "generated_at": datetime.now().isoformat(),
        }

    def refine_script(self, script_data: Dict, feedback: str) -> Dict:
        """
        既存の台本を改良

        Args:
            script_data: 既存の台本データ
            feedback: 改善のためのフィードバック

        Returns:
            改良された台本データ
        """
        logger.info("Refining script based on feedback")

        prompt = f"""以下の台本を改良してください。

【現在の台本】
タイトル: {script_data.get('title', '')}

{script_data.get('script', '')}

【改善要望】
{feedback}

【出力形式】
以下の形式で改良された台本を出力してください:

## タイトル
[改良されたタイトル]

## 台本
[改良された台本]

## キーワード
[キーワードをカンマ区切りで]
"""

        try:
            messages = [{"role": "user", "content": prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=16384,  # 5400文字（日本語）の生成に必要（約8000-16000トークン）
                temperature=0.7,
            )

            script_text = response.choices[0].message.content
            result = self._parse_script_response(script_text)

            logger.info("Script refined successfully")
            return result

        except Exception as e:
            logger.error(f"Error refining script: {e}")
            raise


def main():
    """テスト実行用"""
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    # 設定読み込み
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)

    # サンプルニュース記事
    sample_article = {
        "title": "日銀、政策金利を0.5%に引き上げ決定",
        "summary": "日本銀行は金融政策決定会合で、政策金利を0.25%から0.5%に引き上げることを決定した。",
        "content": "日本銀行は本日の金融政策決定会合で、政策金利を現行の0.25%から0.5%に引き上げることを決定しました。この決定は、インフレ率が目標の2%を上回る状況が続いていることを受けたもので、8対1の多数決で可決されました。市場では追加利上げの可能性も指摘されています。",
        "url": "https://example.com/news/123",
        "published_at": "2025-01-15T10:00:00+09:00",
        "source": "経済新聞",
    }

    # 台本生成
    generator = ScriptGenerator(config["script"])
    result = generator.generate_script(sample_article, target_duration=180)

    print("\n=== 生成された台本 ===\n")
    print(f"タイトル: {result['title']}")
    print(f"\n推定時間: {result['estimated_duration']}秒\n")
    print(f"台本:\n{result['script']}")
    print(f"\nキーワード: {', '.join(result['keywords'])}")


if __name__ == "__main__":
    main()
