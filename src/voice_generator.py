"""
音声生成モジュール
OpenAI TTS APIを使用してテキストから音声を生成
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, List
from openai import OpenAI
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class VoiceGenerator:
    """音声生成クラス"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書（config.yamlから読み込んだvoice設定）
        """
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        self.model = config.get("model", "tts-1-hd")
        self.voice = config.get("voice", "alloy")
        self.speed = config.get("speed", 1.0)
        self.format = config.get("format", "mp3")

    def generate_voice(
        self, text: str, output_path: str, voice: Optional[str] = None
    ) -> str:
        """
        テキストから音声ファイルを生成

        Args:
            text: 読み上げるテキスト
            output_path: 出力ファイルパス
            voice: 使用する声（None の場合は設定ファイルの声を使用）

        Returns:
            生成された音声ファイルのパス
        """
        if not text:
            raise ValueError("Text cannot be empty")

        # 出力ディレクトリが存在しない場合は作成
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating voice: {len(text)} characters")
        logger.info(f"Output path: {output_path}")

        try:
            # OpenAI TTS APIを呼び出し
            response = self.client.audio.speech.create(
                model=self.model,
                voice=voice or self.voice,
                input=text,
                speed=self.speed,
                response_format=self.format,
            )

            # 音声データをファイルに保存
            response.stream_to_file(output_path)

            logger.info(f"Voice file generated successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating voice: {e}")
            raise

    def generate_voice_segments(
        self, text_segments: list[str], output_dir: str, prefix: str = "segment"
    ) -> list[str]:
        """
        複数のテキストセグメントから音声ファイルを生成

        Args:
            text_segments: テキストセグメントのリスト
            output_dir: 出力ディレクトリ
            prefix: ファイル名のプレフィックス

        Returns:
            生成された音声ファイルパスのリスト
        """
        output_paths = []
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {len(text_segments)} voice segments")

        for i, text in enumerate(text_segments):
            if not text.strip():
                logger.warning(f"Skipping empty segment {i}")
                continue

            output_path = output_dir_path / f"{prefix}_{i:03d}.{self.format}"
            try:
                file_path = self.generate_voice(text, str(output_path))
                output_paths.append(file_path)
            except Exception as e:
                logger.error(f"Error generating segment {i}: {e}")
                # エラーがあっても続行

        logger.info(f"Generated {len(output_paths)} voice segments successfully")
        return output_paths

    def split_text_for_tts(
        self, text: str, max_chars: int = 4000
    ) -> list[str]:
        """
        長いテキストをTTS API制限に合わせて分割

        OpenAI TTSは1リクエストあたり4096文字まで対応。
        文の途中で切れないように、句点や改行で分割する。

        Args:
            text: 分割するテキスト
            max_chars: 1セグメントの最大文字数

        Returns:
            分割されたテキストのリスト
        """
        if len(text) <= max_chars:
            return [text]

        segments = []
        current_segment = ""

        # 改行で分割
        lines = text.split("\n")

        for line in lines:
            # 行が長すぎる場合は句点で分割
            if len(line) > max_chars:
                sentences = line.split("。")
                for sentence in sentences:
                    if not sentence.strip():
                        continue

                    sentence = sentence + "。"

                    # 現在のセグメントに追加できるか確認
                    if len(current_segment) + len(sentence) <= max_chars:
                        current_segment += sentence
                    else:
                        # 現在のセグメントを保存して新しいセグメントを開始
                        if current_segment:
                            segments.append(current_segment.strip())
                        current_segment = sentence
            else:
                # 行全体を追加できるか確認
                if len(current_segment) + len(line) + 1 <= max_chars:
                    current_segment += line + "\n"
                else:
                    # 現在のセグメントを保存して新しいセグメントを開始
                    if current_segment:
                        segments.append(current_segment.strip())
                    current_segment = line + "\n"

        # 残りのセグメントを追加
        if current_segment:
            segments.append(current_segment.strip())

        logger.info(f"Split text into {len(segments)} segments")
        return segments

    def generate_voice_with_auto_split(
        self, text: str, output_dir: str, filename: str = "voice"
    ) -> list[str]:
        """
        長いテキストを自動分割して音声を生成

        Args:
            text: 読み上げるテキスト
            output_dir: 出力ディレクトリ
            filename: ファイル名（拡張子なし）

        Returns:
            生成された音声ファイルパスのリスト
        """
        # テキストを分割
        segments = self.split_text_for_tts(text)

        if len(segments) == 1:
            # 分割不要の場合は単一ファイルとして生成
            output_path = Path(output_dir) / f"{filename}.{self.format}"
            return [self.generate_voice(text, str(output_path))]
        else:
            # 複数セグメントの場合は番号付きファイルを生成
            return self.generate_voice_segments(segments, output_dir, filename)

    def transcribe_audio_with_timestamps(
        self, audio_file: str, script_text: Optional[str] = None
    ) -> list[dict]:
        """
        音声ファイルをWhisperで文字起こしし、タイムスタンプ付きのセグメントを返す
        台本が提供されている場合は、それを使って認識精度を向上させる

        Args:
            audio_file: 音声ファイルパス
            script_text: 台本テキスト（オプション）

        Returns:
            タイムスタンプ付きセグメントのリスト
            [{"start": 0.0, "end": 2.5, "text": "こんにちは"}, ...]
        """
        logger.info(f"Transcribing audio file: {audio_file}")

        try:
            with open(audio_file, "rb") as f:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )

            # セグメントを抽出
            segments = []
            if hasattr(transcript, 'segments') and transcript.segments:
                for seg in transcript.segments:
                    segments.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip()
                    })

            logger.info(f"Transcribed {len(segments)} segments")

            # 台本が提供されている場合は補正を適用
            if script_text and segments:
                segments = self._correct_subtitles_with_script(segments, script_text)
                logger.info(f"Applied script-based correction to subtitles")

            return segments

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise

    def _correct_subtitles_with_script(
        self, segments: List[dict], script_text: str
    ) -> List[dict]:
        """
        台本を参照してWhisperの字幕を補正

        Args:
            segments: Whisperが生成した字幕セグメント
            script_text: 元の台本テキスト

        Returns:
            補正された字幕セグメント
        """
        # Whisperの全テキストを結合
        whisper_text = " ".join([seg["text"] for seg in segments])

        # 台本を正規化（空白や改行を統一）
        script_normalized = " ".join(script_text.split())

        logger.info(f"Matching Whisper output ({len(whisper_text)} chars) with script ({len(script_normalized)} chars)")

        # 文字レベルでマッチング
        matcher = SequenceMatcher(None, whisper_text, script_normalized)

        corrected_segments = []
        whisper_pos = 0

        for seg in segments:
            seg_text = seg["text"]
            seg_len = len(seg_text)

            # この区間に対応する台本部分を探す
            # Whisperのテキスト位置から、台本上の対応位置を推定
            match_start = whisper_pos
            match_end = whisper_pos + seg_len

            # マッチング範囲を台本上の位置に変換
            script_start = self._map_position(whisper_pos, matcher)
            script_end = self._map_position(match_end, matcher)

            # 台本から対応部分を抽出
            if script_start is not None and script_end is not None:
                script_segment = script_normalized[script_start:script_end].strip()

                # 類似度をチェック
                similarity = self._calculate_similarity(seg_text, script_segment)

                if similarity > 0.6:  # 60%以上の類似度なら台本を採用
                    corrected_text = script_segment
                    logger.debug(
                        f"Corrected: '{seg_text}' → '{corrected_text}' (similarity: {similarity:.2f})"
                    )
                else:
                    corrected_text = seg_text
                    logger.debug(
                        f"Kept original: '{seg_text}' (similarity too low: {similarity:.2f})"
                    )
            else:
                # マッチング失敗時はWhisperのテキストをそのまま使用
                corrected_text = seg_text
                logger.debug(f"No match found, kept: '{seg_text}'")

            corrected_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": corrected_text
            })

            whisper_pos = match_end

        return corrected_segments

    def _map_position(self, whisper_pos: int, matcher: SequenceMatcher) -> Optional[int]:
        """
        Whisperテキスト上の位置を台本テキスト上の位置にマッピング

        Args:
            whisper_pos: Whisperテキスト上の文字位置
            matcher: SequenceMatcher オブジェクト

        Returns:
            台本テキスト上の対応位置
        """
        # マッチングブロックから対応位置を探す
        for block in matcher.get_matching_blocks():
            whisper_start, script_start, length = block

            if whisper_start <= whisper_pos < whisper_start + length:
                # マッチング範囲内
                offset = whisper_pos - whisper_start
                return script_start + offset
            elif whisper_pos < whisper_start:
                # マッチング範囲の前
                return script_start

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        2つのテキストの類似度を計算
        アルファベット略語の発音表記や同音異義語も考慮する

        Args:
            text1: テキスト1（Whisperの認識結果）
            text2: テキスト2（台本のテキスト）

        Returns:
            類似度（0.0～1.0）
        """
        # 空白を除去して比較
        text1_clean = "".join(text1.split())
        text2_clean = "".join(text2.split())

        if not text1_clean or not text2_clean:
            return 0.0

        # 基本的な文字レベル類似度
        base_similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio()

        # 読み仮名レベルの類似度（同音異義語対策）
        phonetic_similarity = self._calculate_phonetic_similarity(text1_clean, text2_clean)

        # 台本にアルファベット略語が含まれている場合は特別処理
        acronym_similarity = 0.0
        if self._contains_acronym(text2_clean):
            # 略語の発音表記マッチングを試みる
            acronym_similarity = self._check_acronym_pronunciation(text1_clean, text2_clean)

        # 最も高い類似度を採用
        max_similarity = max(base_similarity, phonetic_similarity, acronym_similarity)

        logger.debug(
            f"Similarity scores - base: {base_similarity:.2f}, "
            f"phonetic: {phonetic_similarity:.2f}, acronym: {acronym_similarity:.2f}, "
            f"max: {max_similarity:.2f}"
        )

        return max_similarity

    def _calculate_phonetic_similarity(self, text1: str, text2: str) -> float:
        """
        読み仮名（発音）レベルの類似度を計算
        同音異義語の誤認識を検出するため

        Args:
            text1: Whisperの認識テキスト
            text2: 台本テキスト

        Returns:
            発音ベースの類似度（0.0～1.0）
        """
        try:
            import pykakasi

            # pykakasi を初期化（初回のみ）
            if not hasattr(self, '_kakasi'):
                self._kakasi = pykakasi.kakasi()

            # 両方のテキストをひらがなに変換
            kana1 = self._to_hiragana(text1)
            kana2 = self._to_hiragana(text2)

            # ひらがなレベルで比較
            if kana1 and kana2:
                similarity = SequenceMatcher(None, kana1, kana2).ratio()
                logger.debug(f"Phonetic comparison: '{text1}' ({kana1}) vs '{text2}' ({kana2}) = {similarity:.2f}")
                return similarity

        except ImportError:
            # pykakasi がインストールされていない場合はスキップ
            logger.debug("pykakasi not available, skipping phonetic similarity check")
        except Exception as e:
            logger.debug(f"Error in phonetic similarity calculation: {e}")

        return 0.0

    def _to_hiragana(self, text: str) -> str:
        """
        テキストをひらがなに変換

        Args:
            text: 変換するテキスト

        Returns:
            ひらがな文字列
        """
        try:
            result = self._kakasi.convert(text)
            # 全ての要素の'hira'キーを結合
            hiragana = ''.join([item['hira'] for item in result if 'hira' in item])
            return hiragana
        except Exception as e:
            logger.debug(f"Error converting to hiragana: {e}")
            return ""

    def _contains_acronym(self, text: str) -> bool:
        """
        テキストにアルファベット略語が含まれているかチェック

        Args:
            text: チェックするテキスト

        Returns:
            略語が含まれている場合True
        """
        import re
        # 連続する大文字アルファベット（2文字以上）を略語とみなす
        return bool(re.search(r'[A-Z]{2,}', text))

    def _check_acronym_pronunciation(self, whisper_text: str, script_text: str) -> float:
        """
        アルファベット略語の発音表記をチェック

        Args:
            whisper_text: Whisperの認識テキスト
            script_text: 台本テキスト

        Returns:
            発音ベースの類似度（0.0～1.0）
        """
        import re

        # アルファベット略語を抽出
        acronyms = re.findall(r'[A-Z]{2,}', script_text)

        if not acronyms:
            return 0.0

        # 各略語について発音表記がWhisperテキストに含まれているかチェック
        match_count = 0
        total_count = len(acronyms)

        for acronym in acronyms:
            # 略語の発音パターンを生成
            pronunciation_patterns = self._generate_pronunciation_patterns(acronym)

            # いずれかのパターンがWhisperテキストに含まれているかチェック
            for pattern in pronunciation_patterns:
                if pattern in whisper_text:
                    match_count += 1
                    logger.debug(f"Acronym match: '{acronym}' → '{pattern}' found in Whisper text")
                    break

        # マッチ率を返す
        if total_count == 0:
            return 0.0

        match_rate = match_count / total_count
        logger.debug(f"Acronym pronunciation match rate: {match_rate:.2f} ({match_count}/{total_count})")

        # マッチ率が高ければ類似度を上げる（略語以外の部分も考慮）
        # 完全マッチなら0.7、部分マッチなら調整
        return 0.7 * match_rate

    def _generate_pronunciation_patterns(self, acronym: str) -> List[str]:
        """
        アルファベット略語の発音パターンを生成

        Args:
            acronym: アルファベット略語（例: "CZ", "AI", "CEO"）

        Returns:
            発音パターンのリスト
        """
        # アルファベットの日本語発音マッピング
        pronunciation_map = {
            'A': ['エー', 'エイ', 'A'],
            'B': ['ビー', 'B'],
            'C': ['シー', 'C'],
            'D': ['ディー', 'D'],
            'E': ['イー', 'E'],
            'F': ['エフ', 'F'],
            'G': ['ジー', 'G'],
            'H': ['エイチ', 'H'],
            'I': ['アイ', 'I'],
            'J': ['ジェー', 'J'],
            'K': ['ケー', 'K'],
            'L': ['エル', 'L'],
            'M': ['エム', 'M'],
            'N': ['エヌ', 'N'],
            'O': ['オー', 'O'],
            'P': ['ピー', 'P'],
            'Q': ['キュー', 'Q'],
            'R': ['アール', 'R'],
            'S': ['エス', 'S'],
            'T': ['ティー', 'T'],
            'U': ['ユー', 'U'],
            'V': ['ブイ', 'V'],
            'W': ['ダブリュー', 'W'],
            'X': ['エックス', 'X'],
            'Y': ['ワイ', 'Y'],
            'Z': ['ゼット', 'ゼツ', 'ジー', 'Z'],  # Zは複数の発音がある
        }

        patterns = []

        # パターン1: 各文字を発音に変換して連結（例: "CZ" → "シーゼット"）
        for i, char in enumerate(acronym):
            pronunciations = pronunciation_map.get(char.upper(), [char])
            if i == 0:
                patterns = [[p] for p in pronunciations]
            else:
                new_patterns = []
                for existing_pattern in patterns:
                    for pronunciation in pronunciations:
                        new_patterns.append(existing_pattern + [pronunciation])
                patterns = new_patterns

        # リストを文字列に変換
        result_patterns = [''.join(p) for p in patterns]

        # パターン2: カタカナ小文字表記（例: "CZ" → "Cゼツ"）
        if len(acronym) == 2:
            first_char = acronym[0]
            second_pronunciations = pronunciation_map.get(acronym[1].upper(), [])
            for second_pron in second_pronunciations:
                result_patterns.append(f"{first_char}{second_pron}")

        # パターン3: 元のアルファベットそのまま
        result_patterns.append(acronym)

        logger.debug(f"Generated pronunciation patterns for '{acronym}': {result_patterns}")
        return result_patterns

    def get_available_voices(self) -> list[str]:
        """
        利用可能な声のリストを取得

        Returns:
            利用可能な声のリスト
        """
        # OpenAI TTSの利用可能な声
        # https://platform.openai.com/docs/guides/text-to-speech/voice-options
        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        return voices

    def get_voice_characteristics(self) -> Dict[str, str]:
        """
        各声の特徴を取得

        Returns:
            声の特徴の辞書
        """
        characteristics = {
            "alloy": "中性的でバランスの取れた声",
            "echo": "男性的で落ち着いた声",
            "fable": "英国風のアクセントがある声",
            "onyx": "深みのある男性的な声",
            "nova": "明るく親しみやすい女性的な声",
            "shimmer": "柔らかく温かい女性的な声",
        }
        return characteristics


def main():
    """テスト実行用"""
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    # 設定読み込み
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)

    # サンプルテキスト
    sample_text = """
    こんにちは、経済ニュース解説チャンネルへようこそ。
    今日は、日本銀行が発表した最新の金融政策についてお伝えします。
    日銀は政策金利を0.25%から0.5%に引き上げることを決定しました。
    この決定の背景には、インフレ率の上昇があります。
    """

    # 音声生成
    generator = VoiceGenerator(config["voice"])

    print("\n=== 利用可能な声 ===")
    for voice in generator.get_available_voices():
        characteristics = generator.get_voice_characteristics()
        print(f"- {voice}: {characteristics.get(voice, '')}")

    print("\n=== 音声生成テスト ===")
    output_dir = "./output/audio"
    output_files = generator.generate_voice_with_auto_split(
        sample_text, output_dir, "test_voice"
    )

    print(f"\n生成されたファイル:")
    for file_path in output_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()
