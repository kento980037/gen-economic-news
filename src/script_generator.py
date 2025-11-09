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
        target_duration: int = 180,
        additional_context: Optional[str] = None,
    ) -> Dict:
        """
        ニュース記事から台本を生成

        Args:
            news_article: ニュース記事の辞書（NewsArticle.to_dict()の出力）
            target_duration: 目標動画時間（秒）デフォルト180秒=3分
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
                max_tokens=4096,
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
        self, news_articles: List[Dict], target_duration: int = 300
    ) -> Dict:
        """
        複数のニュース記事から1つの台本を生成（まとめニュース形式）

        Args:
            news_articles: ニュース記事の辞書リスト
            target_duration: 目標動画時間（秒）デフォルト300秒=5分

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
                max_tokens=4096,
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
        prompt = f"""以下の金融ニュース記事を元に、YouTube金融ポッドキャスト用の台本を作成してください。

【記事情報】
タイトル: {news_article.get('title', '')}
要約: {news_article.get('summary', '')}
本文: {news_article.get('content', news_article.get('summary', ''))}
ソース: {news_article.get('source', '')}
公開日時: {news_article.get('published_at', '')}

【台本要件】
- 文字数: {target_chars}文字前後
- 目標時間: 2〜3分（1テーマあたり）
- スタイル: {self.style}（ポッドキャストナレーション形式）
- トーン: {self.tone}

【YouTube金融ポッドキャストの台本作成10の鉄則】

✅ **1. 導入15秒で「今日の価値」を明示（超重要）**
- 最初の15秒で結論を先に言う
- 「今日のテーマは○○。結論は△△です」
- 「これを知ると、今週の相場で〇〇を防げます」
- 視聴者の離脱を防ぐため、冒頭で得られる価値を明確に

✅ **2. 構成は「2〜4つのポイント」で柔軟に整理**
内容の複雑さに応じてポイント数を調整する：

【シンプルなニュース（2ポイント）】
例：単一企業の決算発表、シンプルな価格変動
1. 何が起きたか + なぜ重要か
2. 投資家への影響・今後の展開

【標準的なニュース（3ポイント）】
例：政策変更、市場動向、業界トレンド
1. 結論（今日の重要ポイント）
2. 理由・背景（なぜそうなったか）
3. 今後の展開・投資家への影響

【複雑なニュース（4ポイント）】
例：複数要因が絡む市場変動、構造的変化
1. 結論（今日の重要ポイント）
2. 主要因（最も重要な背景）
3. 副要因（追加の背景・関連情報）
4. 今後の展開・投資家への影響

⚠️ **重要**: 無理やり3つに合わせない！自然な情報の区切りを優先する

✅ **3. 一般ニュースとの差別化を入れる**
以下のいずれかを必ず含める：
- なぜそうなったか（背景解説）
- 過去の同様ケースとの比較
- 投資家目線の解釈
- 他国・他市場との比較
- 一般人の生活への影響

✅ **4. 専門用語は必ず補足説明する（超重要）**
- 専門用語を使ったら、直後に必ず簡単な説明を入れる
- 説明は「（　）」で括るか、「つまり〇〇のことです」と続ける

【補足説明のテンプレート】
- 「QE（量的緩和）」→「QE、つまり量的緩和ですが、これは中央銀行がお金を市場に大量に供給する政策のことです」
- 「テーパリング」→「テーパリング、つまり金融緩和の縮小ですが」
- 「FRB」→「FRB（アメリカの中央銀行）は」
- 「インフレ率」→「インフレ率（物価の上昇率）が」
- 「利回り」→「利回り（投資したお金に対する収益率）が」
- 「ドルインデックス」→「ドルインデックス（ドルの総合的な強さを示す指標）が」

【必ず補足すべき用語例】
金融政策系：QE、テーパリング、ZIRP、NIRP、金融緩和、利上げサイクル
市場指標系：CPI、PPI、PMI、VIX指数、イールドカーブ
投資用語：PER、PBR、配当利回り、株式分割、自社株買い
為替用語：キャリートレード、介入、スワップポイント

【補足説明の注意点】
- 説明は10秒以内に収める（長すぎない）
- 難しい言葉で説明しない（例：「金融緩和」を「流動性供給」と説明しない）
- 日常的な言葉に置き換える（例：「流動性」→「市場に出回るお金の量」）

✅ **5. 感情を5%混ぜる（冷静+少しの驚き）**
台本に自然な感情表現を入れる：
- 「正直これはかなり意外でした」
- 「これは投資家にとって朗報です」
- 「この数字はちょっと注目すべきです」
※ただし信頼性を損なわない程度に

✅ **6. 時系列ではなく"因果関係"で並べる**
構成の順序：
1. 何が起きた
2. なぜそれが起きた（原因）
3. その結果どうなる（影響）
4. 私たち（投資家/視聴者）にどう影響するか

✅ **7. 数字は「変化」にフォーカス**
- 「インフレ率は前月より0.3ポイント上昇」
- 「この伸びは市場予想を超えています」
- 単なる数字の羅列ではなく、変化を言語化

✅ **8. テンポ重視（2〜3分以内に収める）**
- 1トピックは最大2分半
- 3分を超えると離脱率が急増
- 簡潔で無駄のない表現を心がける

✅ **9. 最後に20秒のまとめを置く**
締めくくりは必ず以下の構成：
- 今日のポイント2〜4つ（箇条書き風、本編と同じ数）
- 投資や生活にどう活かせるか
- 視聴者へのメッセージ

✅ **10. 自然な話し言葉で読みやすく**
- 「〜です、〜ます」調
- 短い文で区切る
- 読み上げたときのリズムを意識

【NGパターン】
❌ 冒頭で専門用語を多用
❌ 時系列での淡々とした説明
❌ 結論が最後まで分からない
❌ 視聴者にとってのメリットが不明
❌ 長すぎる1文（読みにくい）

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

1. オープニング（約15秒分）
   - 結論先出し
   - 今日の価値を明示
   - 自然な導入で始める

2. ポイント1（約30秒分）
   - 1つ目の重要ポイント
   - 段落で区切る（改行2つ）

3. ポイント2（約30秒分）
   - 2つ目の重要ポイント
   - 段落で区切る（改行2つ）

4. ポイント3（約30秒分） ⚠️ 必要に応じて省略可
   - 3つ目の重要ポイント（複雑なニュースのみ）
   - シンプルなニュースなら次の「今後の展開」へ
   - 段落で区切る（改行2つ）

5. ポイント4（約30秒分） ⚠️ 必要に応じて追加
   - 4つ目の重要ポイント（非常に複雑なニュースのみ）
   - 通常は不要
   - 段落で区切る（改行2つ）

6. 今後の展開（約30秒分）
   - 予測・見通し
   - 段落で区切る（改行2つ）

7. まとめ（約20秒分）
   - 2〜4つのポイント再確認（本編と同じ数）
   - 視聴者へのメッセージ
   - 段落で区切る（改行2つ）

8. CTA（約60秒分）
   - いいねボタンのお願い
   - チャンネル登録の訴求
   - 価値の再確認
   - 締めの言葉「良い投資を！」

【台本作成例（3ポイントの場合）】
こんにちは。今日のテーマは日銀の利上げです。結論は、これで円高が進む可能性が高いということです。投資家の皆さんが今週注意すべき3つのポイントを解説します。

まず1つ目。なぜ今利上げなのか。実はインフレ率が前月より0.3ポイント上昇し、日銀の目標2%を超えています。これは2年ぶりの高水準です。

2つ目のポイント。発表直後、ドル円は2円急落しました。これは市場予想を超える反応です。投資家の間では円高警戒ムードが広がっています。

3つ目のポイント。今後の展開ですが、市場では追加利上げの観測も出ています。（以下続く）

【シンプルなニュースなら2ポイントの例】
こんにちは。今日はNVIDIAの決算について解説します。結論は、予想を上回る好決算で株価が急騰したということです。2つのポイントで見ていきましょう。

まず1つ目。売上高は前年比94%増の180億ドルで、市場予想を大きく上回りました。特にデータセンター向けGPUが好調です。

2つ目のポイント。この好決算を受けて株価は時間外取引で8%上昇しています。投資家の皆さんにとっては、AI関連銘柄の強さを再確認する結果となりました。

（セクション名は一切書かない）

## キーワード
[重要なキーワードをカンマ区切りで5-10個]

【絶対に守ること】
- 台本には「【オープニング（15秒）】」などのセクション名を絶対に書かない
- 時間表記も書かない
- 純粋なナレーション原稿だけを出力
- 段落の区切りは改行2つで表現
- 自然な話し言葉で流れるように書く
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
                max_tokens=4096,
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
