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

        # toneとstyleの組み合わせに応じてsystem_promptを選択
        if self.tone == "chill":
            if self.style == "dialogue":
                self.system_prompt = config.get("system_prompt_chill_dialogue", "")
            else:
                self.system_prompt = config.get("system_prompt_chill", "")
        else:
            self.system_prompt = config.get("system_prompt", "")

        # マルチターン会話設定
        multiturn_config = config.get("multiturn_dialogue", {})
        self.multiturn_enabled = multiturn_config.get("enabled", False)
        self.turns_per_section = multiturn_config.get("turns_per_section", 4)

        # キャラクター別のsystem prompt
        self.system_prompt_speaker_a = config.get("system_prompt_speaker_a", "")
        self.system_prompt_speaker_b = config.get("system_prompt_speaker_b", "")

    def generate_script_by_sections(
        self,
        news_article: Dict,
        target_duration: int = 1080,
        additional_context: Optional[str] = None,
    ) -> Dict:
        """
        セクション分割で台本を生成（確実に目標文字数を達成）

        Args:
            news_article: ニュース記事の辞書
            target_duration: 目標動画時間（秒）
            additional_context: 追加のコンテキスト情報

        Returns:
            生成された台本の辞書
        """
        logger.info(f"Generating script by sections for: {news_article.get('title', 'Unknown')}")

        target_chars = target_duration * 5
        logger.info(f"Target duration: {target_duration} seconds, Target chars: {target_chars} characters")

        # 記事情報の準備
        article_content = news_article.get('content', '')
        if not article_content or len(article_content.strip()) < 50:
            article_content = news_article.get('summary', '')

        article_info = f"""【メイン記事】
タイトル: {news_article.get('title', '')}
本文: {article_content}
ソース: {news_article.get('source', '')}
公開日時: {news_article.get('published_at', '')}
"""
        if additional_context:
            article_info += f"\n【参考記事・コンテキスト】\n{additional_context}\n"

        # まず構成を生成（各ポイントで何を書くか明確化）
        logger.info("Generating outline (structure planning)...")
        outline = self._generate_outline(article_info, target_chars)

        logger.info("Outline generated:")
        logger.info(f"  Point 1 [WHAT]: {outline['point1_summary']}")
        logger.info(f"  Point 2 [WHY]: {outline['point2_summary']}")
        logger.info(f"  Point 3 [SO WHAT]: {outline['point3_summary']}")

        # セクションごとに生成
        sections = {}

        # 1. タイトル生成
        logger.info("Generating title...")
        sections['title'] = self._generate_title(article_info, news_article)

        # 2. オープニング（200文字）
        logger.info("Generating opening (200 chars)...")
        sections['opening'] = self._generate_section(
            "オープニング",
            article_info,
            200,
            "結論を先に言い、今日の価値を明示する導入部分。視聴者の興味を引く。"
        )

        # 3. ポイント1【WHAT: 何が起きたか】（1400文字）
        logger.info("Generating point 1 [WHAT] (1400 chars)...")
        sections['point1'] = self._generate_section(
            "ポイント1【何が起きたか】",
            article_info,
            1400,
            """このニュースで「具体的に何が起きたのか」を詳しく説明してください。

【このポイントで扱う内容】（構成に基づく）
""" + outline['point1_details'] + """

【含めるべき内容】
- 具体的な出来事・発表内容（誰が、何を、いつ、どこで）
- 数字・金額・割合を正確に
- 関係する企業名・人物名
- 発表や変化の詳細
- 過去のデータとの比較（例: 前四半期比、前年比）

【禁止事項】
- 原因や背景の説明（それはポイント2で扱う）
- 市場への影響や意味の解説（それはポイント3で扱う）

上記の構成に従い、純粋に「事実」のみを詳しく報告してください。"""
        )

        # 4. ポイント2【WHY: なぜ起きたか】（1400文字）
        logger.info("Generating point 2 [WHY] (1400 chars)...")
        sections['point2'] = self._generate_section(
            "ポイント2【なぜ起きたか】",
            article_info,
            1400,
            """ポイント1で説明した出来事が「なぜ起きたのか」を詳しく分析してください。

【このポイントで扱う内容】（構成に基づく）
""" + outline['point2_details'] + """

【含めるべき内容】
- 背景・原因（経済環境、政策、市場トレンド）
- 過去の経緯（同様の事例、歴史的な流れ）
- 関連する要因（他の出来事との関係）
- 主要なステークホルダーの動機や意図

【禁止事項】
- 出来事そのものの詳細説明（既にポイント1で説明済み）
- 今後の予測や影響（それはポイント3で扱う）

上記の構成に従い、「なぜこうなったのか」の理由を深掘りしてください。"""
        )

        # 5. ポイント3【SO WHAT: 何を意味するか】（1200文字）
        logger.info("Generating point 3 [SO WHAT] (1200 chars)...")
        sections['point3'] = self._generate_section(
            "ポイント3【何を意味するか】",
            article_info,
            1200,
            """この出来事が「投資家や市場にとって何を意味するのか」を詳しく解説してください。

【このポイントで扱う内容】（構成に基づく）
""" + outline['point3_details'] + """

【含めるべき内容】
- 市場への具体的な影響（株価、為替、金利など）
- 投資家が取るべき行動や注意点
- 専門家・アナリストの見解
- 他の企業や業界への波及効果
- 類似の過去事例との比較（その時どうなったか）

【禁止事項】
- 出来事の詳細説明（既にポイント1で説明済み）
- 原因や背景の説明（既にポイント2で説明済み）

上記の構成に従い、「だから何？」「投資家はどうすべき？」に答えてください。"""
        )

        # 6. 今後の展開（700文字）
        logger.info("Generating future outlook (700 chars)...")
        sections['outlook'] = self._generate_section(
            "今後の展開",
            article_info,
            700,
            "短期・中期・長期の予測、投資家が取るべき行動を具体的に。"
        )

        # 7. まとめ（300文字）
        logger.info("Generating summary (300 chars)...")
        sections['summary'] = self._generate_section(
            "まとめ",
            article_info,
            300,
            "今日の重要ポイントを2-3つ箇条書き風に再確認。"
        )

        # 8. CTA（300文字）
        logger.info("Generating CTA (300 chars)...")
        sections['cta'] = self._generate_section(
            "CTA",
            article_info,
            300,
            "いいね・チャンネル登録のお願い。親しみやすく、押し付けがましくなく。最後は「良い投資を！」で締める。"
        )

        # セクションを結合（改行1つで連結、音声のポーズを短く）
        full_script = "\n".join([
            sections['opening'],
            sections['point1'],
            sections['point2'],
            sections['point3'],
            sections['outlook'],
            sections['summary'],
            sections['cta']
        ])

        # 添削前の文字数を記録
        original_length = len(full_script)
        logger.info(f"Original script length: {original_length} chars (before editing)")

        # OpenAI APIで添削（重複削除、流れ改善）
        logger.info("Editing script to remove redundancy and improve flow...")
        edited_script = self._edit_script(full_script, target_chars, outline)

        edited_length = len(edited_script)
        logger.info(f"Edited script length: {edited_length} chars (removed {original_length - edited_length} chars)")

        # キーワード生成
        logger.info("Generating keywords...")
        keywords = self._generate_keywords(article_info, edited_script)

        logger.info(f"✓ Script generated by sections. Final length: {edited_length} chars (target: {target_chars})")

        return {
            "title": sections['title'],
            "script": edited_script,
            "keywords": keywords,
            "estimated_duration": edited_length // 5,
            "generated_at": datetime.now().isoformat(),
        }

    def generate_script_chill(
        self,
        news_article: Dict,
        target_duration: int = 720,  # 10-15分 = 600-900秒、中央値720秒
        additional_context: Optional[str] = None,
    ) -> Dict:
        """
        ゆるチル系ポッドキャスト台本を生成（夜にお酒を飲みながら聞ける）

        Args:
            news_article: ニュース記事の辞書
            target_duration: 目標動画時間（秒）デフォルト720秒=12分
            additional_context: 追加のコンテキスト情報

        Returns:
            生成された台本の辞書
        """
        logger.info(f"Generating chill podcast script for: {news_article.get('title', 'Unknown')}")

        # マルチターン会話生成が有効かつdialogueスタイルの場合
        if self.multiturn_enabled and self.style == "dialogue":
            logger.info("Using multi-turn dialogue generation")
            return self._generate_dialogue_multiturn(
                news_article, target_duration, additional_context
            )

        # 文字数を時間から計算（日本語: 1秒あたり約5文字）
        target_chars = target_duration * 5  # 720秒 × 5 = 3600文字
        logger.info(f"Target duration: {target_duration} seconds, Target chars: {target_chars} characters")

        # 記事情報の準備
        article_content = news_article.get('content', '')
        if not article_content or len(article_content.strip()) < 50:
            article_content = news_article.get('summary', '')

        article_info = f"""【メイン記事】
タイトル: {news_article.get('title', '')}
本文: {article_content}
ソース: {news_article.get('source', '')}
公開日時: {news_article.get('published_at', '')}
"""
        if additional_context:
            article_info += f"\n【関連記事】\n{additional_context}\n"

        # ゆるチル系専用プロンプト（dialogueかnarrationで出力形式が変わる）
        if self.style == "dialogue":
            script_format = """【超重要】会話形式で、**すべての発言に必ず「A:」または「B:」を明記してください**。

**絶対に守ること：**
- すべての発言は「A:」または「B:」で始める（例外なし）
- 発言者ラベルのない行は絶対に書かない
- 各発言は改行で区切る

**正しい形式の例：**
A: はい、今夜もやってきましたー
B: どうもー。今日も一杯やりながらですね
A: 今日のニュース、なんかすごいことになってますよ
B: まあ、簡単に言うと〜ってことですね

**間違った形式（絶対NG）：**
はい、今夜もやってきましたー（発言者ラベルがない）
どうもー。今日も一杯やりながらですね（発言者ラベルがない）"""
        else:
            script_format = "[純粋なナレーション原稿のみ。セクション名は書かない。]"

        user_prompt = f"""{article_info}

上記のニュース記事をもとに、夜にお酒を飲みながら聞ける"ゆるくてチルい"雰囲気のYouTubeポッドキャスト用の台本を作ってください。

**⚠️ 最重要要件: 必ず{target_chars}文字以上の台本を作成してください ⚠️**

**台本の要件**
- **長さ: {target_chars}文字以上（10〜15分相当）← 必須！短いのはNG**
- トーン: 深夜ラジオ風、友達と話すような自然な口調
- 構成: システムプロンプトに従った7セクション構成
- スタイル: {"2人の掛け合い（ボケ＆ツッコミ）" if self.style == "dialogue" else "単独ナレーション"}
{"- **【超重要】すべての発言に必ず「A:」または「B:」のラベルを付ける（例外なし）**" if self.style == "dialogue" else ""}

**お願い**
- 記事本文の文章はそのまま読まず、要点を再構成して語り口調に
- 難しい部分はゆるい言葉に置き換える
- 硬い専門用語には軽く一言説明を入れる
- AIっぽさを消して"話してる感じ"で書く
- 具体的な企業名・数字・日付は正確に（記事に基づく）
- **会話を増やす**: やりとりを多くして、十分な長さを確保
- **深掘りする**: 一つのトピックについて、複数の角度から話す
- **具体例を複数**: 過去事例、他社比較、市場反応など
{"- **発言者ラベル（A:、B:）のない行は絶対に書かない**" if self.style == "dialogue" else ""}

【出力形式】
以下の形式で出力してください:

## タイトル
[15-35文字のカジュアルなタイトル]

## 台本
{script_format}

## キーワード
[カンマ区切りで5-10個]
"""

        try:
            # OpenAI APIを呼び出し
            messages = [{"role": "user", "content": user_prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=16384,
                temperature=0.7,  # ゆるチル系は少し高めの温度で自然さを出す
            )

            # レスポンスからテキストを抽出
            script_text = response.choices[0].message.content

            # 台本を解析
            result = self._parse_script_response(script_text, news_article)

            # ラベルなし行を修正（無条件で適用）
            logger.info("Applying speaker label correction...")
            result['script'] = self._fix_missing_speaker_labels(result['script'])
            logger.info("Speaker label correction applied")

            script_length = len(result['script'])
            logger.info(f"✓ Chill script generated. Length: {script_length} chars (target: {target_chars})")

            return result

        except Exception as e:
            logger.error(f"Error generating chill script: {e}")
            raise

    def _generate_dialogue_multiturn(
        self,
        news_article: Dict,
        target_duration: int = 720,
        additional_context: Optional[str] = None,
    ) -> Dict:
        """
        マルチターン会話生成（AとBの発言を交互に生成）

        Args:
            news_article: ニュース記事の辞書
            target_duration: 目標動画時間（秒）
            additional_context: 追加のコンテキスト情報

        Returns:
            生成された台本の辞書
        """
        logger.info("Starting multi-turn dialogue generation")

        # 記事情報の準備
        article_content = news_article.get('content', '')
        if not article_content or len(article_content.strip()) < 50:
            article_content = news_article.get('summary', '')

        article_info = f"""【メイン記事】
タイトル: {news_article.get('title', '')}
本文: {article_content}
ソース: {news_article.get('source', '')}
公開日時: {news_article.get('published_at', '')}
"""
        if additional_context:
            article_info += f"\n【関連記事】\n{additional_context}\n"

        # 会話の全体構成を定義
        sections = [
            {
                "name": "オープニング",
                "turns": 2,  # A → B
                "context": "今日のポッドキャストを始める。お酒を飲みながらのリラックスした雰囲気で、今日のニュースを簡単に紹介。"
            },
            {
                "name": "ニュース紹介",
                "turns": 3,  # A → B → A
                "context": "今日のニュースの概要を紹介。Aがざっくり、Bが正確な情報を補足。"
            },
            {
                "name": "メイン解説",
                "turns": self.turns_per_section,  # 設定値（デフォルト4）
                "context": "ニュースの詳細を深掘り。Aのボケ、Bのツッコミで掘り下げていく。数字や背景を詳しく。"
            },
            {
                "name": "関連情報・過去事例",
                "turns": 3,
                "context": "関連する情報や過去の似た事例について話す。Aが興味を示し、Bが説明。"
            },
            {
                "name": "感想と意味",
                "turns": 3,
                "context": "このニュースが何を意味するか、投資家にとってどうか。Aの感性、Bの論理で。"
            },
            {
                "name": "まとめ",
                "turns": 2,  # B → A
                "context": "今日のポイントを2-3つに整理。Bがまとめ、Aが感想。"
            },
            {
                "name": "エンディング",
                "turns": 2,  # A → B
                "context": "いいね・チャンネル登録のお願い。「良い投資を！」で締める。"
            }
        ]

        # 会話履歴を保持
        conversation_history = []
        full_dialogue = []

        # 各セクションで会話を生成
        for section in sections:
            logger.info(f"Generating section: {section['name']} ({section['turns']} turns)")

            section_context = f"""【セクション】{section['name']}
【このセクションの目的】{section['context']}
【記事情報】
{article_info}
"""

            # 各ターンで発言を生成
            for turn_idx in range(section['turns']):
                # 話者を決定（奇数ターンはA、偶数ターンはB）
                speaker = "A" if turn_idx % 2 == 0 else "B"

                # 発言を生成
                utterance = self._generate_speaker_turn(
                    speaker=speaker,
                    section_context=section_context,
                    conversation_history=conversation_history,
                    is_first_turn=(turn_idx == 0)
                )

                # 会話履歴に追加
                conversation_history.append({
                    "speaker": speaker,
                    "text": utterance
                })

                # 台本に追加
                full_dialogue.append(f"{speaker}: {utterance}")

                logger.info(f"  Turn {turn_idx + 1}/{section['turns']}: {speaker} ({len(utterance)} chars)")

        # 台本を結合
        script_text = "\n".join(full_dialogue)
        script_length = len(script_text)

        logger.info(f"✓ Multi-turn dialogue generated. Total length: {script_length} chars")

        # タイトルを生成
        title = self._generate_title(article_info, news_article)

        # キーワードを生成
        keywords = self._generate_keywords(article_info, script_text)

        return {
            "title": title,
            "script": script_text,
            "keywords": keywords,
            "estimated_duration": script_length // 5,
            "generated_at": datetime.now().isoformat(),
        }

    def _generate_speaker_turn(
        self,
        speaker: str,
        section_context: str,
        conversation_history: List[Dict],
        is_first_turn: bool = False
    ) -> str:
        """
        1ターンの発言を生成（AまたはB）

        Args:
            speaker: 話者（"A" または "B"）
            section_context: セクションのコンテキスト
            conversation_history: これまでの会話履歴
            is_first_turn: セクションの最初のターンかどうか

        Returns:
            生成された発言テキスト
        """
        # キャラクターに応じたsystem promptを選択
        if speaker == "A":
            system_prompt = self.system_prompt_speaker_a
        else:
            system_prompt = self.system_prompt_speaker_b

        # 会話履歴を整形
        history_text = ""
        if conversation_history:
            recent_history = conversation_history[-6:]  # 直近6ターンのみ
            history_text = "\n".join([
                f"{h['speaker']}: {h['text']}" for h in recent_history
            ])

        # プロンプトを構築
        if is_first_turn:
            # セクションの最初のターン
            user_prompt = f"""{section_context}

このセクションを始めてください。あなた（{speaker}）が最初に話します。

【重要】
- 短い発言（1-3文）で
- 自然な会話のように
- キャラクターらしく
"""
        else:
            # 続きのターン
            user_prompt = f"""{section_context}

【これまでの会話】
{history_text}

上記の会話を受けて、あなた（{speaker}）の発言を続けてください。

【重要】
- 短い発言（1-3文）で
- 相手の発言を受けて自然に
- キャラクターらしく
"""

        try:
            # OpenAI APIを呼び出し
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,  # 短い発言なので少なめ
                temperature=0.8,  # 自然な会話のため高め
            )

            utterance = response.choices[0].message.content.strip()

            # 「A:」や「B:」のプレフィックスを削除（もし含まれている場合）
            import re
            utterance = re.sub(r'^[AB][:：]\s*', '', utterance)

            return utterance

        except Exception as e:
            logger.error(f"Error generating turn for speaker {speaker}: {e}")
            return f"[発言生成エラー]"

    def generate_script(
        self,
        news_article: Dict,
        target_duration: int = 1080,
        additional_context: Optional[str] = None,
        max_retries: int = 2,
    ) -> Dict:
        """
        ニュース記事から台本を生成（toneに応じて適切なメソッドを呼び出す）

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

        # toneに応じて適切なメソッドを呼び出す
        if self.tone == "chill":
            # ゆるチル系の場合は専用メソッドを使用
            result = self.generate_script_chill(
                news_article=news_article,
                target_duration=target_duration if target_duration != 1080 else 720,  # デフォルトを12分に
                additional_context=additional_context
            )

            # 念のため再度ラベル修正を適用（無条件）
            logger.info("Applying speaker label correction (final check)...")
            result['script'] = self._fix_missing_speaker_labels(result['script'])
            logger.info("Speaker label correction applied (final check)")

            return result

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
            if attempt > 0 and result is not None:
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
                    max_tokens=16384,  # gpt-4o-miniの最大値
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

    def _generate_section(
        self, section_name: str, article_info: str, target_chars: int, instruction: str
    ) -> str:
        """
        個別セクションを生成

        Args:
            section_name: セクション名
            article_info: 記事情報
            target_chars: 目標文字数
            instruction: セクションの指示内容

        Returns:
            生成されたセクションのテキスト
        """
        prompt = f"""{article_info}

【セクション】{section_name}

【このセクションの役割】
{instruction}

【執筆方針】
- **簡潔に核心だけ**: 回りくどい説明は避け、重要なポイントに絞る
- **具体性重視**: 企業名、数字、日付を正確に（記事情報に基づく）
- **専門用語**: 括弧で短く補足（例: GDP（国内総生産））
- **話し言葉**: 自然な「です・ます」調
- **目標文字数**: {target_chars}文字程度（質を優先）

【ファクトチェック】
- 記事に明記されていない統計データや数値は使用しない
- 日付は記事情報に基づき、不確実な場合は「〜月中旬」「〜月頃」と表現
- 将来予測は断定を避け「〜の可能性がある」と表現
- 過去事例は記事に明記されているものに限定

【重要な禁止事項】
このセクションに書くのは上記の役割に関することだけです。
他のポイントで扱う内容は一切書かないでください。

純粋なナレーション原稿のみを出力してください（セクション名や見出しは不要）。
"""

        try:
            messages = [{"role": "user", "content": prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=16384,
                temperature=0.6,
            )

            section_text = response.choices[0].message.content.strip()
            actual_length = len(section_text)

            logger.info(f"  {section_name}: {actual_length}/{target_chars} chars")

            # 文字数が少なすぎる場合は警告
            if actual_length < target_chars * 0.7:
                logger.warning(f"  {section_name} is shorter than expected ({actual_length}/{target_chars})")

            return section_text

        except Exception as e:
            logger.error(f"Error generating section '{section_name}': {e}")
            return f"[セクション生成エラー: {section_name}]"

    def _generate_outline(self, article_info: str, target_chars: int) -> Dict:
        """
        台本の構成を生成（各ポイントで何を扱うか明確化）

        Args:
            article_info: 記事情報
            target_chars: 目標文字数

        Returns:
            構成情報の辞書
        """
        prompt = f"""{article_info}

上記の記事を元に、YouTube金融ポッドキャスト用の台本構成を作成してください。
台本は「WHAT → WHY → SO WHAT」の3ポイント構成で、各ポイントで扱う内容を明確に分けます。

【絶対厳守：事実確認】
⚠️ **記事に書かれている情報のみを使用してください**
- 推測や補足情報の追加は一切禁止
- 日付、数値、人物名、企業名は記事通りに正確に
- 記事にない過去事例や統計データを創作しない
- 不明な情報は「記事では触れられていない」と明記
- インフレ率や価格上昇率などの統計データは、記事に明記されている場合のみ使用
- 将来予測は「〜の可能性がある」「〜が見込まれる」など断定を避ける表現に
- 政治的動機や戦略については推測であることを明示（「一部アナリストは〜と分析」など）
- 過去事例を引用する場合は、記事に明記されているものに限定

【構成要件】
- ポイント1【何が起きたか】: 事実のみ（数字、出来事、発表内容）
- ポイント2【なぜ起きたか】: 背景・原因・過去の経緯
- ポイント3【何を意味するか】: 市場への影響、投資家への示唆

【重要】各ポイントの内容は重複しないように明確に分けてください。

以下の形式で出力してください：

## ポイント1【何が起きたか】
[1行要約]

**扱う内容**（箇条書き）:
-
-
-

## ポイント2【なぜ起きたか】
[1行要約]

**扱う内容**（箇条書き）:
-
-
-

## ポイント3【何を意味するか】
[1行要約]

**扱う内容**（箇条書き）:
-
-
-
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.5,
            )

            outline_text = response.choices[0].message.content.strip()

            # 構成テキストをパース
            outline = self._parse_outline(outline_text)

            logger.debug(f"Outline generated successfully")
            return outline

        except Exception as e:
            logger.error(f"Error generating outline: {e}")
            # フォールバック: 空の構成を返す
            return {
                'point1_summary': '記事の主要な出来事',
                'point1_details': '- 記事で報道された主要な事実\n- 具体的な数字と日付',
                'point2_summary': '出来事の背景と原因',
                'point2_details': '- なぜこの出来事が起きたのか\n- 過去の経緯',
                'point3_summary': '市場と投資家への影響',
                'point3_details': '- 市場への具体的な影響\n- 投資家への示唆',
            }

    def _parse_outline(self, outline_text: str) -> Dict:
        """構成テキストをパースして辞書に変換"""
        outline = {
            'point1_summary': '',
            'point1_details': '',
            'point2_summary': '',
            'point2_details': '',
            'point3_summary': '',
            'point3_details': '',
        }

        lines = outline_text.split('\n')
        current_section = None
        current_details = []

        for line in lines:
            line = line.strip()

            if 'ポイント1' in line or 'Point 1' in line:
                if current_section:
                    outline[f'{current_section}_details'] = '\n'.join(current_details)
                current_section = 'point1'
                current_details = []
            elif 'ポイント2' in line or 'Point 2' in line:
                if current_section:
                    outline[f'{current_section}_details'] = '\n'.join(current_details)
                current_section = 'point2'
                current_details = []
            elif 'ポイント3' in line or 'Point 3' in line:
                if current_section:
                    outline[f'{current_section}_details'] = '\n'.join(current_details)
                current_section = 'point3'
                current_details = []
            elif line.startswith('[') and line.endswith(']') and current_section:
                # 要約行
                outline[f'{current_section}_summary'] = line[1:-1]
            elif line.startswith('-') or line.startswith('*'):
                # 箇条書き行
                current_details.append(line)

        # 最後のセクションを保存
        if current_section:
            outline[f'{current_section}_details'] = '\n'.join(current_details)

        return outline

    def _generate_title(self, article_info: str, news_article: Dict) -> str:
        """タイトルを生成"""
        prompt = f"""{article_info}

上記の記事を元に、YouTube金融ポッドキャスト用の魅力的なタイトルを生成してください。

【要件】
- 15-35文字
- 投資家の興味を引く
- 具体的な内容が分かる
- 数字や企業名を含めると良い

タイトルのみを出力してください（説明不要）。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7,
            )
            title = response.choices[0].message.content.strip()
            logger.info(f"  Title: {title}")
            return title

        except Exception as e:
            logger.error(f"Error generating title: {e}")
            return news_article.get("title", "経済ニュース解説")[:35]

    def _edit_script(self, script: str, target_chars: int, outline: Dict) -> str:
        """
        台本を添削（重複削除、流れ改善、冗長性削除、ファクトチェック）

        Args:
            script: 添削前の台本
            target_chars: 目標文字数
            outline: 構成情報（各ポイントで扱う内容）

        Returns:
            添削後の台本
        """
        prompt = f"""以下の台本を添削してください。各セクションを個別に生成したため、重複や冗長性が多く含まれています。

【台本の構成】（各ポイントで扱うべき内容）
ポイント1【何が起きたか】: {outline['point1_summary']}
扱う内容:
{outline['point1_details']}

ポイント2【なぜ起きたか】: {outline['point2_summary']}
扱う内容:
{outline['point2_details']}

ポイント3【何を意味するか】: {outline['point3_summary']}
扱う内容:
{outline['point3_details']}

【現在の台本】（{len(script)}文字）
{script}

【添削の最重要原則】
**重複は徹底的に削除し、簡潔で情報密度の高い台本にしてください。**
文字数よりも品質を優先してください。

【ファクトチェック要件】（最重要）
添削時に以下の点を厳密にチェックし、必要に応じて修正してください：

1. **日付の正確性**
   - 発表日や報道日が記事情報と一致しているか確認
   - 不確実な日付は「〜月中旬」「〜月頃」など曖昧な表現に修正
   - 例: 「11月16日に発表」→「11月中旬に発表（報道は11月14〜15日）」

2. **統計データの検証**
   - インフレ率、価格上昇率などの数値が記事情報に基づいているか確認
   - 記事にない数値を創作していないか確認
   - 数値が過大または不正確な場合は削除または修正
   - 例: 「インフレ率が6%」→記事に根拠がなければ「インフレ圧力が高まっている」など曖昧な表現に

3. **断定的表現の回避**
   - 将来予測や効果について断定的な表現を避ける
   - 「〜が期待される」「〜が安定する」→「〜に寄与する可能性がある」「〜が見込まれる」
   - 因果関係が複合的な場合は「〜も一因となっている」と表現

4. **推測の明示**
   - 政治的動機や選挙戦略などの分析は推測であることを明示
   - 「〜は選挙戦略である」→「一部アナリストは選挙を意識した動きと分析している」
   - 「〜が目的だ」→「〜を目的としている可能性がある」

5. **根拠のない過去事例の削除**
   - 過去事例を引用する際は記事情報に基づいているか確認
   - 記事にない過去事例は削除または「過去には同様のケースで〜という事例があったとされる」と曖昧に
   - 具体的な出典がない場合は削除を優先

6. **公式統計への言及**
   - 統計データを使用する場合は、可能な限り「公式統計によると」「〜のデータでは」と付記
   - BLS（米国労働統計局）、USDA（米国農務省）などの公式機関を言及

【具体的な添削指示】

1. **重複の徹底削除**（最優先）
   - 同じ数字（例: 8200万ドル）が3回以上出てきたら1-2回に減らす
   - 同じ事実（例: 投資セクター、日付）の繰り返しを1回だけに
   - 「これは〜ということです」「つまり〜です」のような言い換えを削除
   - 例を複数挙げている場合は最も重要な1-2個に絞る

2. **冗長表現の簡潔化**
   - 「〜と考えられます」「〜の可能性があります」などの曖昧な表現を削除または明確に
   - 「このように」「さらに」「また」などの接続詞を必要最小限に
   - 長い説明を短く核心だけに
   - 同じ意味の文を統合

3. **不自然な接続詞の削除**
   - 「さて、今日は〜」「さて、ここからは〜」→ 削除
   - セクション間は自然に繋げる

4. **まとめセクションの整理**
   - まとめは箇条書き（3-4点）で簡潔に
   - 本文で既出の内容を繰り返さない
   - `---` でまとめを区切る

5. **専門用語の説明**（簡潔に）
   - 不足している場合のみ追加
   - 括弧で短く補足（例: GDP（国内総生産））

【削除すべきもの】
- 同じ内容の繰り返し（何度も同じ数字や事実を言わない）
- 回りくどい説明（核心だけ残す）
- 不必要な例示（重要なもの1-2個だけ）
- 曖昧な表現（「可能性があります」を多用しない）
- 記事にない統計データや過去事例

【保持すべきもの】
- 具体的な数字、企業名、日付（初出のみ、記事情報に基づくもの）
- 重要な事実と分析
- 構成（オープニング→本文→まとめ→CTA）

【構成に基づく重複削除】（最重要）
上記の構成を見て、各ポイントの役割を理解してください：
- **ポイント1（WHAT）**: {outline['point1_summary']}に集中
- **ポイント2（WHY）**: {outline['point2_summary']}に集中
- **ポイント3（SO WHAT）**: {outline['point3_summary']}に集中

**重複削除のルール**:
1. ポイント1で扱った内容（事実・数字）をポイント2・3で繰り返さない
2. ポイント2で扱った内容（原因・背景）をポイント3で繰り返さない
3. 同じ数字や事実は、構成で指定されたポイントでのみ詳しく説明
4. 他のポイントで言及する場合は「前述の通り」「先ほどの」などで簡潔に

**例**:
- 「Infineon 1.6%, SAP 3.2%下落」はポイント1でのみ詳しく
- ポイント3では「これらの企業」「先述の下落」と簡潔に
- 同じ数字を4回も繰り返さない

【目標】
- 最終的な文字数: 4500-5000文字程度（質を優先、長さは二の次）
- 聞いててストレスがない、簡潔で分かりやすい台本
- 事実に基づいた正確な情報提供

添削後の台本のみを出力してください（説明不要）。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
                temperature=0.3,  # 低めに設定して忠実に添削
            )

            edited_script = response.choices[0].message.content.strip()

            # 文字数が大幅に減少した場合は警告
            if len(edited_script) < target_chars * 0.7:
                logger.warning(f"Edited script is significantly shorter than target ({len(edited_script)}/{target_chars})")

            return edited_script

        except Exception as e:
            logger.error(f"Error editing script: {e}")
            logger.warning("Returning original script without editing")
            return script

    def _generate_keywords(self, article_info: str, full_script: str) -> List[str]:
        """キーワードを生成"""
        prompt = f"""{article_info}

上記の記事と以下の台本から、YouTube用のキーワードを5-10個抽出してください。

【台本抜粋】
{full_script[:500]}...

カンマ区切りで出力してください（説明不要）。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.5,
            )
            keywords_text = response.choices[0].message.content.strip()
            keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]
            logger.info(f"  Keywords: {', '.join(keywords[:3])}...")
            return keywords[:10]

        except Exception as e:
            logger.error(f"Error generating keywords: {e}")
            return ["経済", "ビジネス", "ニュース"]

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

    def _fix_missing_speaker_labels(self, script: str) -> str:
        """
        すべてのラベルを削除して、発言行にAとBを交互に割り当て直す

        このアプローチにより、AIが途中から間違ったラベルを付けた場合でも
        確実に正しい交互パターンを保証できます。

        Args:
            script: 台本テキスト

        Returns:
            修正された台本テキスト
        """
        import re

        logger.info("=" * 60)
        logger.info("STARTING SPEAKER LABEL CORRECTION")
        logger.info(f"Input script length: {len(script)} chars")
        logger.info("=" * 60)

        lines = script.split("\n")
        logger.info(f"Total lines in script: {len(lines)}")

        # ステップ1: すべてのラベルを削除して、発言のみを抽出
        utterances = []
        lines_with_label = 0
        lines_without_label = 0

        for line in lines:
            line_stripped = line.strip()

            # 空行や区切り線はスキップ
            if not line_stripped or re.match(r'^[-=*_]+$', line_stripped):
                continue

            # ラベルを削除して発言のみを抽出
            speaker_match = re.match(r'^([AB])[:：]\s*(.+)$', line_stripped)
            if speaker_match:
                # 既にラベルがある場合は、ラベルを削除
                lines_with_label += 1
                text = speaker_match.group(2).strip()
                if text and len(text) >= 3:
                    utterances.append(text)
                    logger.debug(f"Extracted (had label): {text[:50]}...")
            else:
                # ラベルがない場合はそのまま
                if len(line_stripped) >= 3:
                    lines_without_label += 1
                    utterances.append(line_stripped)
                    logger.debug(f"Extracted (no label): {line_stripped[:50]}...")

        logger.info(f"Lines with label: {lines_with_label}")
        logger.info(f"Lines without label: {lines_without_label}")
        logger.info(f"Total utterances extracted: {len(utterances)}")

        # ステップ2: AとBを交互に割り当て
        fixed_lines = []
        for i, text in enumerate(utterances):
            speaker = "A" if i % 2 == 0 else "B"
            fixed_lines.append(f"{speaker}: {text}")
            if i < 5:  # 最初の5行をログ出力
                logger.info(f"Line {i+1}: {speaker}: {text[:50]}...")

        logger.info(f"Reconstructed dialogue with {len(utterances)} utterances (alternating A/B pattern)")

        # 連続する同じ発言者をチェック（デバッグ用）
        prev_speaker = None
        consecutive_count = 0
        for line in fixed_lines:
            match = re.match(r'^([AB])[:：]', line)
            if match:
                speaker = match.group(1)
                if speaker == prev_speaker:
                    consecutive_count += 1
                prev_speaker = speaker

        if consecutive_count > 0:
            logger.warning(f"Found {consecutive_count} consecutive same-speaker lines (this should not happen)")
        else:
            logger.info("✓ All lines have alternating speakers (A/B pattern verified)")

        result = "\n".join(fixed_lines)
        logger.info(f"Output script length: {len(result)} chars")
        logger.info("=" * 60)
        logger.info("SPEAKER LABEL CORRECTION COMPLETED")
        logger.info("=" * 60)

        return result

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
