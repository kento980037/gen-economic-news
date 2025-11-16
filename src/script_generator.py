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
    ) -> Dict:
        """
        ニュース記事から台本を生成

        Args:
            news_article: ニュース記事の辞書（NewsArticle.to_dict()の出力）
            target_duration: 目標動画時間（秒）デフォルト1080秒=18分
            additional_context: 追加のコンテキスト情報

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

        # プロンプトを構築
        user_prompt = self._build_prompt(
            news_article, target_chars, additional_context
        )

        # プロンプトをログに出力
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
                max_tokens=16384,  # 5400文字（日本語）の生成に必要（約8000-16000トークン）
                temperature=0.7,
            )

            # レスポンスからテキストを抽出
            script_text = response.choices[0].message.content

            # 台本を解析
            result = self._parse_script_response(script_text, news_article)

            logger.info(
                f"Script generated successfully. Length: {len(result['script'])} chars"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating script: {e}")
            raise

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
        """単一記事用のプロンプトを構築"""

        # 本文を優先、なければ要約を使用
        article_content = news_article.get('content', '')
        if not article_content or len(article_content.strip()) < 50:
            article_content = news_article.get('summary', '')

        prompt = f"""⚠️⚠️⚠️ 最重要指示 ⚠️⚠️⚠️
必ず{target_chars}文字以上の台本を作成してください。これは絶対に守るべき要件です。

以下の金融ニュース記事を元に、YouTube金融ポッドキャスト用の台本を作成してください。

【メイン記事】
タイトル: {news_article.get('title', '')}
本文: {article_content}
ソース: {news_article.get('source', '')}
公開日時: {news_article.get('published_at', '')}

【台本要件】
⚠️ **超重要**: 必ず{target_chars}文字以上の台本を作成してください
- 文字数: {target_chars}文字前後（最低でも{target_chars}文字は必須）
- 目標時間: 18分程度（1テーマあたり、より詳しく解説）
- スタイル: {self.style}（ポッドキャストナレーション形式）
- トーン: {self.tone}
- 重要: ポイント数は2-4のままで、各ポイントを深く、詳しく解説してください
- 各ポイントは最低でも180-240秒分（900-1200文字）の詳細な解説が必要です

【YouTube金融ポッドキャストの台本作成10の鉄則】

✅ **1. 導入15秒で「今日の価値」を明示（超重要）**
- 最初の15秒で結論を先に言う
- 「今日のテーマは○○。結論は△△です」
- 「これを知ると、今週の相場で〇〇を防げます」
- 視聴者の離脱を防ぐため、冒頭で得られる価値を明確に

✅ **2. 構成は「2〜4つのポイント」で柔軟に整理し、各ポイントを深く掘り下げる**
内容の複雑さに応じてポイント数を調整する：

【シンプルなニュース（2ポイント）】
例：単一企業の決算発表、シンプルな価格変動
1. 何が起きたか + なぜ重要か（120-180秒でしっかり解説）
   - 具体的な数字と事実
   - 過去のデータとの比較
   - 専門家の見方や市場の反応
2. 投資家への影響・今後の展開（120-180秒でしっかり解説）
   - 短期的な影響
   - 中長期的な見通し
   - 類似ケースとの比較

【標準的なニュース（3ポイント）】
例：政策変更、市場動向、業界トレンド
1. 結論（今日の重要ポイント）（120-180秒）
   - 何が起きたのか詳細に
   - なぜこれが重要なのか
   - 具体例を複数挙げる
2. 理由・背景（なぜそうなったか）（120-180秒）
   - 主要な要因を深掘り
   - 過去の経緯
   - 他の関連要因
3. 今後の展開・投資家への影響（120-180秒）
   - 短期的な予測
   - 中長期的な展望
   - 投資家が取るべき行動

【複雑なニュース（4ポイント）】
例：複数要因が絡む市場変動、構造的変化
1. 結論（今日の重要ポイント）（120秒）
2. 主要因（最も重要な背景）（120-180秒で詳しく）
3. 副要因（追加の背景・関連情報）（120-180秒で詳しく）
4. 今後の展開・投資家への影響（120-180秒で詳しく）

⚠️ **重要**:
- 無理やり3つに合わせない！自然な情報の区切りを優先する
- 各ポイントは120-180秒かけてしっかり深掘りする
- 具体例、過去事例、数字を豊富に使って厚みのある解説にする

✅ **3. 具体性を最優先する（最重要ルール）**

抽象的な表現は禁止。必ず具体的な固有名詞と数字を使用：
- 企業名、アナリスト名、機関名を明記
- 数字は割合、金額、日付を正確に
- 情報源に無い場合は捏造せず「詳細は不明」と述べる

例：
❌ 「テクノロジー株が急落」→ ✅ 「NVIDIAが8.2%下落、TSMCが5.7%下落」
❌ 「専門家は懸念」→ ✅ 「ゴールドマン・サックスのチーフエコノミスト、ジャン・ハッチウス氏は」

✅ **4. 深掘りと差別化**

必ず含める：
- なぜそうなったか（背景・原因を具体的に）
- 過去の類似ケース（年月日と事例）
- 投資家への具体的影響
- 他国・他市場との比較

✅ **5. 専門用語は必ず補足説明**
直後に簡単な説明を入れる。例：「FRB（アメリカの中央銀行）」「CPI（消費者物価指数）」

✅ **6. その他の重要ポイント**
- 感情を5%混ぜる（「これは意外です」など）
- 時系列でなく因果関係で構成
- 数字は「変化」を強調（「前月より0.3ポイント上昇」）
- 各ポイント120-180秒かけて深掘り

✅ **7. 最後のまとめとCTA**
締めくくり：
- 今日のポイント2〜4つ（箇条書き風、本編と同じ数）
- 投資や生活にどう活かせるか
- 視聴者へのメッセージ

✅ **8. 自然な話し言葉**
「〜です、〜ます」調で短い文、読み上げやすいリズム。
冒頭15秒で結論を先に言う。

"""

        if additional_context:
            prompt += f"\n【追加コンテキスト（複数ソース参照）】\n{additional_context}\n"
            prompt += """
【複数記事参照時の追加要件】
- 複数の情報源から得られた情報を統合的に分析してください
- 各メディアの報道内容の共通点と相違点を考慮してください
- より多角的で深い分析を提供してください
- 情報の信頼性を高めるため、複数ソースで確認された事実を優先してください

【重要：トピックの一貫性チェック】
⚠️ もし参考記事の中に明らかに異なるトピックが含まれている場合は、メイン記事と関連が深い記事のみを使用してください
⚠️ 複数の無関係なトピック（例：タリフ、医療、市政など）を1つの台本に混ぜないでください
⚠️ 1つの台本は1つの明確なテーマに集中してください
"""

        prompt += """
【出力形式】
以下の形式で厳密に出力してください:

## タイトル
[動画のタイトル（15-35文字、投資家向けキャッチー）]

## 台本
【重要】台本にはセクション名や時間表記を含めないこと。
純粋なナレーション原稿のみを出力してください。

以下の構成で台本を作成（セクション名は書かない）:

⚠️ **超重要**: 18分の動画を作るため、各セクションを十分に長く書いてください

1. オープニング（約40秒分 = 200文字）
   - 結論先出し
   - 今日の価値を明示
   - 自然な導入で始める

2-4. ポイント1-3（各240-300秒 = 1200-1500文字）
   ⚠️ 各ポイント必ず1200文字以上書く
   - 具体的な数字・企業名・事例を豊富に
   - 過去データとの比較、専門家の見解
   - 背景・要因を深掘り、市場への影響
   - 段落で区切る（改行2つ）

5. 今後の展開（180-240秒 = 900-1200文字）
   短期・中期・長期の予測、投資家の行動

6. まとめ（60秒 = 300文字）
   ポイント再確認、視聴者へのメッセージ

7. CTA（120秒 = 600文字）
   いいね・登録のお願い、締め「良い投資を！」

⚠️ **文字数チェック**:
- ポイント1-3: 各1200文字以上（合計3600文字以上）
- その他のセクション: 合計約1800文字
- 総合計: 5400文字前後を必ず達成してください

## キーワード
[重要なキーワードをカンマ区切りで5-10個]

【重要】
- セクション名・時間表記は書かない
- 純粋なナレーション原稿のみ
- 段落は改行2つで区切る

⚠️⚠️⚠️ **最重要確認** ⚠️⚠️⚠️
各ポイント1200文字以上、台本全体5400文字前後を必ず達成。
不足なら具体例・過去事例・専門家見解を追加。
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
