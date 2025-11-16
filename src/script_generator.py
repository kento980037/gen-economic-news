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
        edited_script = self._edit_script(full_script, target_chars)

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

【目標文字数】{target_chars}文字（必須）

【指示】
{instruction}

【重要要件】
- 必ず{target_chars}文字以上書いてください
- 具体的な企業名、数字、日付を使用
- 専門用語は直後に説明を入れる
- 自然な話し言葉で、「です・ます」調
- セクション名や見出しは書かず、純粋なナレーション原稿のみ

このセクションの内容のみを出力してください（他のセクションは含めない）。
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

    def _edit_script(self, script: str, target_chars: int) -> str:
        """
        台本を添削（重複削除、流れ改善、冗長性削除）

        Args:
            script: 添削前の台本
            target_chars: 目標文字数

        Returns:
            添削後の台本
        """
        prompt = f"""以下の台本を添削してください。各セクションを個別に生成したため、重複や冗長性が多く含まれています。

【現在の台本】（{len(script)}文字）
{script}

【添削の最重要原則】
**重複は徹底的に削除し、簡潔で情報密度の高い台本にしてください。**
文字数よりも品質を優先してください。

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

【保持すべきもの】
- 具体的な数字、企業名、日付（初出のみ）
- 重要な事実と分析
- 構成（オープニング→本文→まとめ→CTA）

【目標】
- 最終的な文字数: 4500-5000文字程度（質を優先、長さは二の次）
- 聞いててストレスがない、簡潔で分かりやすい台本

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
