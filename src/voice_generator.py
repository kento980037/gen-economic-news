"""
音声生成モジュール
OpenAI TTS APIを使用してテキストから音声を生成
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional
from openai import OpenAI

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
