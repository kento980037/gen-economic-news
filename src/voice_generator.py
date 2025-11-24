"""
音声生成モジュール
OpenAI TTS APIまたはGemini TTS APIを使用してテキストから音声を生成
"""

import os
import logging
import wave
import json
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
        self.provider = config.get("provider", "openai")

        # レート制限設定を読み込み
        rate_limit_config = config.get("rate_limit", {})
        self.requests_interval = rate_limit_config.get("requests_interval", 3.0)
        logger.info(f"Rate limit: {self.requests_interval} seconds between requests")

        # プロバイダー別の初期化
        if self.provider == "gemini":
            self._init_gemini()
        else:
            self._init_openai()

    def _init_openai(self):
        """OpenAI TTSの初期化"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        openai_config = self.config.get("openai", {})
        self.model = openai_config.get("model", "tts-1-hd")
        self.voice = openai_config.get("voice", "alloy")
        self.speed = openai_config.get("speed", 1.0)
        self.format = openai_config.get("format", "mp3")

        # 会話形式用の2人の声を設定
        dialogue_voices_config = openai_config.get("dialogue_voices", {})
        self.dialogue_voice_a = dialogue_voices_config.get("speaker_a", "shimmer")
        self.dialogue_voice_b = dialogue_voices_config.get("speaker_b", "nova")

        logger.info(f"Initialized OpenAI TTS: model={self.model}, voice={self.voice}")
        logger.info(f"Dialogue voices: A={self.dialogue_voice_a}, B={self.dialogue_voice_b}")

    def _init_gemini(self):
        """Gemini TTSの初期化"""
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.genai_types = types
        except ImportError:
            raise ImportError(
                "google-genai package is required for Gemini TTS. "
                "Install it with: pip install google-genai"
            )

        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set in environment variables")

        self.client = genai.Client(api_key=self.api_key)
        gemini_config = self.config.get("gemini", {})
        self.model = gemini_config.get("model", "gemini-2.5-flash-preview-tts")
        self.voice = gemini_config.get("voice", "Kore")
        self.format = gemini_config.get("format", "wav")
        self.style_prompt = gemini_config.get("style_prompt", "")

        # 会話形式用の2人の声を設定
        dialogue_voices_config = gemini_config.get("dialogue_voices", {})
        self.dialogue_voice_a = dialogue_voices_config.get("speaker_a", "Fenrir")
        self.dialogue_voice_b = dialogue_voices_config.get("speaker_b", "Aoede")

        logger.info(f"Initialized Gemini TTS: model={self.model}, voice={self.voice}")
        logger.info(f"Dialogue voices: A={self.dialogue_voice_a}, B={self.dialogue_voice_b}")

    def generate_voice(
        self, text: str, output_path: str, voice: Optional[str] = None, retry_count: int = 0
    ) -> str:
        """
        テキストから音声ファイルを生成

        Args:
            text: 読み上げるテキスト
            output_path: 出力ファイルパス
            voice: 使用する声（None の場合は設定ファイルの声を使用）
            retry_count: リトライ回数（内部使用）

        Returns:
            生成された音声ファイルのパス
        """
        if not text:
            raise ValueError("Text cannot be empty")

        # 出力ディレクトリが存在しない場合は作成
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating voice ({self.provider}): {len(text)} characters")
        logger.info(f"Output path: {output_path}")

        max_retries = 3

        try:
            if self.provider == "gemini":
                return self._generate_voice_gemini(text, output_path, voice)
            else:
                return self._generate_voice_openai(text, output_path, voice)

        except Exception as e:
            logger.error(f"Error generating voice with {self.provider}: {e}")

            # リトライロジック
            if retry_count < max_retries:
                import time
                wait_time = 2 ** retry_count  # 指数バックオフ: 1秒, 2秒, 4秒
                logger.warning(f"Retrying in {wait_time} seconds... (attempt {retry_count + 1}/{max_retries})")
                time.sleep(wait_time)
                return self.generate_voice(text, output_path, voice, retry_count + 1)

            # 最大リトライ後は失敗として処理を停止
            logger.error(f"Failed to generate voice after {max_retries} retries. Stopping pipeline.")
            raise

    def _generate_voice_openai(
        self, text: str, output_path: str, voice: Optional[str] = None
    ) -> str:
        """OpenAI TTSで音声を生成"""
        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice or self.voice,
            input=text,
            speed=self.speed,
            response_format=self.format,
        )

        # 音声データをファイルに保存
        response.stream_to_file(output_path)

        logger.info(f"Voice file generated successfully (OpenAI): {output_path}")
        return output_path

    def _generate_voice_gemini(
        self, text: str, output_path: str, voice: Optional[str] = None
    ) -> str:
        """Gemini TTSで音声を生成"""
        # スタイル指示を追加（設定されている場合）
        prompt_text = text
        if self.style_prompt:
            prompt_text = f"{self.style_prompt}：{text}"

        # Gemini TTS APIを呼び出し
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt_text,
            config=self.genai_types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=self.genai_types.SpeechConfig(
                    voice_config=self.genai_types.VoiceConfig(
                        prebuilt_voice_config=self.genai_types.PrebuiltVoiceConfig(
                            voice_name=voice or self.voice,
                        )
                    )
                ),
            )
        )

        # レスポンス構造の検証とエラーハンドリング
        if not response or not hasattr(response, 'candidates'):
            raise ValueError(f"Invalid response from Gemini API: no candidates found (text: '{text}')")

        if not response.candidates:
            raise ValueError(f"Empty candidates list from Gemini API (text: '{text}')")

        candidate = response.candidates[0]
        if not candidate or not hasattr(candidate, 'content') or candidate.content is None:
            raise ValueError(f"Invalid candidate content from Gemini API (text: '{text}')")

        if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            raise ValueError(f"No parts in candidate content from Gemini API (text: '{text}')")

        part = candidate.content.parts[0]
        if not hasattr(part, 'inline_data') or not part.inline_data:
            raise ValueError(f"No inline_data in part from Gemini API (text: '{text}')")

        if not hasattr(part.inline_data, 'data') or not part.inline_data.data:
            raise ValueError(f"No audio data in inline_data from Gemini API (text: '{text}')")

        # 音声データを取得
        audio_data = part.inline_data.data

        # WAVファイルとして保存（PCM audio: 24kHz, mono, 16-bit）
        self._save_wav_file(audio_data, output_path, sample_rate=24000, channels=1, sample_width=2)

        logger.info(f"Voice file generated successfully (Gemini): {output_path}")
        return output_path

    def _save_wav_file(
        self, audio_data: bytes, output_path: str, sample_rate: int = 24000,
        channels: int = 1, sample_width: int = 2
    ):
        """PCM音声データをWAVファイルとして保存"""
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

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

            # レート制限対策：2個目以降は設定された時間待機
            if i > 0:
                import time
                logger.info(f"Rate limiting: waiting {self.requests_interval} seconds before next request...")
                time.sleep(self.requests_interval)

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

    def generate_voice_dialogue(
        self, script_text: str, output_dir: str, filename: str = "voice"
    ) -> list[str]:
        """
        会話形式の台本から2人の声で音声を生成

        Args:
            script_text: 会話形式の台本テキスト（A: とB: で発言者を明記）
            output_dir: 出力ディレクトリ
            filename: ファイル名（拡張子なし）

        Returns:
            生成された音声ファイルパスのリスト
        """
        logger.info("Generating dialogue voice with two speakers")

        # 台本を解析してA:とB:の発言を抽出
        dialogue_segments = self._parse_dialogue_script(script_text)

        if not dialogue_segments:
            logger.warning("No dialogue segments found, falling back to single voice generation")
            return self.generate_voice_with_auto_split(script_text, output_dir, filename)

        # 出力ディレクトリを作成
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # 各発言を生成
        audio_files = []
        for i, segment in enumerate(dialogue_segments):
            speaker = segment["speaker"]
            text = segment["text"]

            # 発言者に応じて声を選択
            if speaker == "A":
                voice = self.dialogue_voice_a
            elif speaker == "B":
                voice = self.dialogue_voice_b
            else:
                voice = None  # デフォルトの声を使用

            # 音声ファイルのパス
            segment_path = output_dir_path / f"{filename}_seg{i:03d}.{self.format}"

            try:
                # レート制限対策：2個目以降は設定された時間待機
                if i > 0:
                    import time
                    logger.info(f"Rate limiting: waiting {self.requests_interval} seconds before next request...")
                    time.sleep(self.requests_interval)

                # 音声生成
                logger.info(f"Generating segment {i}/{len(dialogue_segments)}: Speaker {speaker} ({voice}) - {len(text)} chars")
                self.generate_voice(text, str(segment_path), voice=voice)
                audio_files.append(str(segment_path))
                logger.info(f"✓ Segment {i} generated successfully")
            except Exception as e:
                logger.error(f"✗ Error generating dialogue segment {i}: {e}")
                # 最初の数個だけ失敗した場合は続行、それ以外は中断
                if i < 3:
                    # 最初の3個以内のエラーは続行
                    continue
                else:
                    # 3個以降でエラーが発生したら、それまでのを使う
                    logger.warning(f"Too many errors, using {len(audio_files)} segments generated so far")
                    break

        if not audio_files:
            raise RuntimeError("Failed to generate any dialogue audio segments")

        if len(audio_files) < len(dialogue_segments) // 2:
            logger.warning(
                f"Only {len(audio_files)}/{len(dialogue_segments)} segments generated successfully. "
                f"This may result in a short video."
            )

        # 音声ファイルを結合し、タイミング情報を記録
        merged_path = output_dir_path / f"{filename}.{self.format}"
        logger.info(f"Merging {len(audio_files)} dialogue segments into {merged_path}")
        segment_timings = self._merge_audio_files_with_timing(audio_files, str(merged_path), dialogue_segments)

        # タイミング情報をJSONファイルに保存（字幕生成で使用）
        timing_json_path = output_dir_path / f"{filename}_timings.json"
        with open(timing_json_path, 'w', encoding='utf-8') as f:
            json.dump(segment_timings, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved segment timing information to {timing_json_path}")

        # 一時ファイルを削除
        for audio_file in audio_files:
            try:
                Path(audio_file).unlink()
            except Exception as e:
                logger.warning(f"Could not delete temporary file {audio_file}: {e}")

        logger.info(f"Dialogue voice generation complete: {merged_path}")
        return [str(merged_path)]

    def _parse_dialogue_script(self, script_text: str) -> List[Dict]:
        """
        会話形式の台本を解析してA:とB:の発言を抽出

        Args:
            script_text: 会話形式の台本テキスト

        Returns:
            発言セグメントのリスト [{"speaker": "A", "text": "..."}, ...]
        """
        import re

        segments = []

        # 空行で区切ってブロックに分割
        blocks = script_text.split("\n\n")

        last_speaker = None  # 最後に使用した発言者を記録

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # 区切り線や記号のみの行をスキップ
            if re.match(r'^[-=*_]+$', block):
                logger.info(f"Skipping separator line: '{block}'")
                continue

            # ブロック内の行を処理
            lines = block.split("\n")
            block_text = []
            speaker = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 区切り線や記号のみの行をスキップ
                if re.match(r'^[-=*_]+$', line):
                    logger.info(f"Skipping separator line: '{line}'")
                    continue

                # 発言者の識別（A: またはB: で始まる行）
                speaker_match = re.match(r'^([AB])[:：]\s*(.*)$', line)

                if speaker_match:
                    speaker = speaker_match.group(1)
                    text = speaker_match.group(2)
                    if text:
                        block_text.append(text)
                else:
                    block_text.append(line)

            # ブロックに発言者指定がない場合、AとBを交互に割り当て
            if not speaker:
                if last_speaker == "A":
                    speaker = "B"
                else:
                    speaker = "A"

            # セグメントを追加
            if block_text:
                segments.append({
                    "speaker": speaker,
                    "text": " ".join(block_text).strip()
                })
                last_speaker = speaker

        logger.info(f"Parsed dialogue script: {len(segments)} segments (A: {sum(1 for s in segments if s['speaker'] == 'A')}, B: {sum(1 for s in segments if s['speaker'] == 'B')})")

        # 短すぎるセグメントを処理（Gemini TTSは短いテキストを処理できないため）
        MIN_TEXT_LENGTH = 5
        merged_segments = []
        skip_next = False

        for i, segment in enumerate(segments):
            if skip_next:
                skip_next = False
                continue

            text = segment["text"]
            speaker = segment["speaker"]

            # 短すぎるセグメントをチェック
            if len(text) < MIN_TEXT_LENGTH:
                # 前のセグメントと結合を試みる（同じ発言者の場合）
                if merged_segments and merged_segments[-1]["speaker"] == speaker:
                    logger.info(f"Merging short segment ({len(text)} chars) with previous: '{text}'")
                    merged_segments[-1]["text"] += " " + text
                    continue
                # 次のセグメントと結合を試みる（同じ発言者の場合）
                elif i + 1 < len(segments) and segments[i + 1]["speaker"] == speaker:
                    logger.info(f"Merging short segment ({len(text)} chars) with next: '{text}'")
                    merged_text = text + " " + segments[i + 1]["text"]
                    merged_segments.append({
                        "speaker": speaker,
                        "text": merged_text
                    })
                    skip_next = True
                    continue
                else:
                    # 結合できない場合は、警告を出してそのまま追加（スキップしてもよいが、一応残す）
                    logger.warning(f"Cannot merge very short segment ({len(text)} chars), keeping as-is: '{text}'")

            merged_segments.append(segment)

        if len(merged_segments) < len(segments):
            logger.info(f"Merged short segments: {len(segments)} → {len(merged_segments)} segments")

        return merged_segments

    def _merge_audio_files(self, audio_files: List[str], output_path: str):
        """
        複数の音声ファイルを1つに結合

        Args:
            audio_files: 結合する音声ファイルのリスト
            output_path: 出力ファイルパス
        """
        try:
            # pydubを使用して音声を結合
            from pydub import AudioSegment

            # 最初の音声を読み込み
            combined = AudioSegment.from_file(audio_files[0])

            # 残りの音声を順次追加
            for audio_file in audio_files[1:]:
                audio = AudioSegment.from_file(audio_file)
                combined += audio

            # 結合した音声を保存
            combined.export(output_path, format=self.format)

            logger.info(f"Merged {len(audio_files)} audio files into {output_path}")

        except ImportError:
            logger.error("pydub is required for merging audio files. Install it with: pip install pydub")
            raise
        except Exception as e:
            logger.error(f"Error merging audio files: {e}")
            raise

    def _merge_audio_files_with_timing(
        self, audio_files: List[str], output_path: str, dialogue_segments: List[Dict]
    ) -> List[Dict]:
        """
        複数の音声ファイルを1つに結合し、各セグメントのタイミング情報を記録

        Args:
            audio_files: 結合する音声ファイルのリスト
            output_path: 出力ファイルパス
            dialogue_segments: 台本から解析した発言セグメント情報

        Returns:
            各セグメントのタイミング情報のリスト
            [{"speaker": "A", "text": "...", "start": 0.0, "end": 2.5}, ...]
        """
        try:
            from pydub import AudioSegment

            segment_timings = []
            current_time = 0.0

            # 最初の音声を読み込み
            combined = AudioSegment.from_file(audio_files[0])
            duration = len(combined) / 1000.0  # ミリ秒から秒に変換

            # 最初のセグメントのタイミング情報を記録
            if len(dialogue_segments) > 0:
                segment_timings.append({
                    "speaker": dialogue_segments[0]["speaker"],
                    "text": dialogue_segments[0]["text"],
                    "start": current_time,
                    "end": current_time + duration
                })
                current_time += duration

            # 残りの音声を順次追加
            for i, audio_file in enumerate(audio_files[1:], start=1):
                audio = AudioSegment.from_file(audio_file)
                combined += audio

                duration = len(audio) / 1000.0  # ミリ秒から秒に変換

                # タイミング情報を記録
                if i < len(dialogue_segments):
                    segment_timings.append({
                        "speaker": dialogue_segments[i]["speaker"],
                        "text": dialogue_segments[i]["text"],
                        "start": current_time,
                        "end": current_time + duration
                    })
                    current_time += duration

            # 結合した音声を保存
            combined.export(output_path, format=self.format)

            logger.info(f"Merged {len(audio_files)} audio files with timing info: {len(segment_timings)} segments")
            return segment_timings

        except ImportError:
            logger.error("pydub is required for merging audio files. Install it with: pip install pydub")
            raise
        except Exception as e:
            logger.error(f"Error merging audio files with timing: {e}")
            raise

    def _split_long_subtitles(self, segments: List[Dict], max_chars: int = 40) -> List[Dict]:
        """
        長すぎる字幕を適切な位置で分割

        Args:
            segments: 字幕セグメントのリスト
            max_chars: 1行の最大文字数

        Returns:
            分割された字幕セグメントのリスト
        """
        split_segments = []

        for segment in segments:
            text = segment["text"]

            # 鉤括弧を削除（音声では読まれないため）
            text = text.replace("「", "").replace("」", "")

            start_time = segment["start"]
            end_time = segment["end"]
            duration = end_time - start_time

            # 最大文字数以下ならそのまま（鉤括弧削除済みのテキストを使用）
            if len(text) <= max_chars:
                split_segments.append({
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "speaker": segment.get("speaker")
                })
                continue

            # 句読点で分割
            import re
            # 句読点（。、）で分割し、句読点も含める
            parts = re.split(r'([。、])', text)

            # 句読点を前の部分に結合
            merged_parts = []
            i = 0
            while i < len(parts):
                if parts[i]:  # 空文字列をスキップ
                    part = parts[i]
                    # 次が句読点なら結合
                    if i + 1 < len(parts) and parts[i + 1] in ['。', '、']:
                        part += parts[i + 1]
                        i += 2
                    else:
                        i += 1
                    if part.strip():
                        merged_parts.append(part.strip())
                else:
                    i += 1

            # 分割できない場合（句読点がない）は、スペースで分割を試みる
            if not merged_parts:
                # max_charsごとに強制分割
                merged_parts = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

            # さらに長い部分をmax_charsで再分割
            final_parts = []
            for part in merged_parts:
                if len(part) <= max_chars:
                    final_parts.append(part)
                else:
                    # スペースで分割を試みる
                    words = part.split()
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) + 1 <= max_chars:
                            current_line += (word + " ")
                        else:
                            if current_line:
                                final_parts.append(current_line.strip())
                            current_line = word + " "
                    if current_line:
                        final_parts.append(current_line.strip())

            # 各パートに時間を割り当て
            if not final_parts:
                final_parts = [text]

            total_chars = sum(len(p) for p in final_parts)
            current_time = start_time

            for part in final_parts:
                # 文字数比率で時間を計算
                char_ratio = len(part) / total_chars if total_chars > 0 else 1.0 / len(final_parts)
                part_duration = duration * char_ratio

                split_segments.append({
                    "start": current_time,
                    "end": current_time + part_duration,
                    "text": part,
                    "speaker": segment.get("speaker")  # 発言者情報を保持
                })

                current_time += part_duration

        logger.info(f"Split {len(segments)} segments into {len(split_segments)} subtitle lines")
        return split_segments

    def transcribe_audio_with_timestamps(
        self, audio_file: str, script_text: Optional[str] = None, subtitle_config: Optional[Dict] = None
    ) -> list[dict]:
        """
        音声ファイルをWhisperで文字起こしし、タイムスタンプ付きのセグメントを返す
        台本が提供されている場合は、それを使って認識精度を向上させる

        Args:
            audio_file: 音声ファイルパス
            script_text: 台本テキスト（オプション）
            subtitle_config: 字幕設定（オフセット等、オプション）

        Returns:
            タイムスタンプ付きセグメントのリスト
            [{"start": 0.0, "end": 2.5, "text": "こんにちは"}, ...]
        """
        logger.info(f"Transcribing audio file: {audio_file}")

        # 字幕設定のデフォルト値
        if subtitle_config is None:
            subtitle_config = {}

        # 会話形式の音声の場合、タイミング情報ファイルが存在するかチェック
        audio_path = Path(audio_file)
        timing_json_path = audio_path.parent / f"{audio_path.stem}_timings.json"

        if timing_json_path.exists():
            logger.info(f"Found dialogue timing info: {timing_json_path}")
            logger.info("Using speaker-based segmentation for dialogue subtitles")

            try:
                with open(timing_json_path, 'r', encoding='utf-8') as f:
                    segment_timings = json.load(f)

                logger.info(f"Loaded {len(segment_timings)} dialogue segments from timing file")

                # 長すぎる字幕を分割
                split_segments = self._split_long_subtitles(segment_timings, max_chars=40)
                logger.info(f"Split into {len(split_segments)} subtitle segments (max 40 chars per line)")

                return split_segments
            except Exception as e:
                logger.warning(f"Failed to load timing info, falling back to Whisper: {e}")

        try:
            # OpenAI Whisperで文字起こし（プロバイダーに関係なくWhisperを使用）
            # Gemini TTSで生成した音声でもOpenAI Whisperで文字起こし可能
            if self.provider == "gemini":
                # Gemini用にOpenAIクライアントを一時的に初期化
                openai_key = os.getenv("OPENAI_API_KEY")
                if not openai_key:
                    raise ValueError("OPENAI_API_KEY is required for transcription")
                openai_client = OpenAI(api_key=openai_key)
            else:
                openai_client = self.client

            with open(audio_file, "rb") as f:
                transcript = openai_client.audio.transcriptions.create(
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

            logger.info(f"Transcribed {len(segments)} Whisper segments")

            # 台本が提供されている場合は、OpenAI APIで書き起こしを台本で補正
            if script_text:
                logger.info("Correcting Whisper transcription with script using OpenAI API")
                segments = self._correct_whisper_with_script_ai(segments, script_text)
            else:
                # 台本がない場合は、Whisperの結果を句読点で再分割
                logger.info("No script provided, using Whisper text with punctuation-based segmentation")
                segments = self._resegment_by_punctuation(segments)

            logger.info(f"Final subtitle count: {len(segments)}")
            return segments

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise

    def _correct_whisper_with_script_ai(self, whisper_segments: List[dict], script_text: str) -> List[dict]:
        """
        OpenAI APIを使ってWhisperの書き起こしを台本で補正
        各セグメントのタイミングは保持し、テキストのみを台本に合わせる

        Args:
            whisper_segments: Whisperの書き起こしセグメント
            script_text: 台本テキスト

        Returns:
            補正された字幕セグメント
        """
        logger.info(f"[AI CORRECTION] Correcting {len(whisper_segments)} segments with script")

        # OpenAIクライアントを取得
        if self.provider == "gemini":
            # Gemini使用時もOpenAIクライアントを初期化
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                logger.warning("OPENAI_API_KEY not found, skipping AI correction")
                return self._resegment_by_punctuation(whisper_segments)
            openai_client = OpenAI(api_key=openai_key)
        else:
            openai_client = self.client

        # Whisperの書き起こし全文を作成
        whisper_text = " ".join([seg["text"] for seg in whisper_segments])

        # OpenAI APIで台本とWhisperテキストをマッピング
        try:
            prompt = f"""あなたは字幕編集のプロフェッショナルです。

以下の「台本」と「Whisperによる書き起こし」を比較して、各Whisperセグメントに対応する台本テキストを特定してください。

【台本（正確なテキスト）】
{script_text}

【Whisperによる書き起こし（タイミングは正確だがテキストに誤認識がある可能性）】
{whisper_text}

【タスク】
Whisperの各セグメントに対応する台本の部分を特定し、JSON形式で出力してください。
タイミングはWhisperのものをそのまま使用し、テキストのみを台本通りに補正します。

⚠️ 【最重要】セグメント数の厳守：
- Whisperセグメント数: {len(whisper_segments)}個
- 出力するセグメント数: {len(whisper_segments)}個（必ず同じ数にする）
- セグメントを統合したり分割したりしない
- 各Whisperセグメントに対して、1対1で台本テキストを割り当てる

⚠️ 【超重要】テキストの正確性について：
- 台本のテキストを一字一句正確にコピーしてください
- 漢字、ひらがな、カタカナ、句読点、すべて台本通りに
- Whisperの誤認識を台本で完全に修正してください
- 台本にない言葉を追加したり、台本の言葉を変更したりしないでください

【Whisperセグメント数】
合計: {len(whisper_segments)}個

【Whisperセグメント詳細】
"""
            # Whisperセグメント情報をJSON形式で追加
            whisper_segments_json = json.dumps(
                [{"index": i, "text": seg["text"], "start": seg["start"], "end": seg["end"]}
                 for i, seg in enumerate(whisper_segments)],
                ensure_ascii=False,
                indent=2
            )

            prompt += whisper_segments_json

            prompt += f"""

【出力形式】
以下のJSON形式で出力してください（他のテキストは一切含めないでください）:
{{
  "corrected_segments": [
    {{"index": 0, "text": "台本から抽出した対応テキスト", "start": 0.0, "end": 2.5}},
    {{"index": 1, "text": "台本から抽出した対応テキスト", "start": 2.5, "end": 5.0}},
    ...
    {{"index": {len(whisper_segments) - 1}, "text": "最後のセグメント", "start": X, "end": Y}}
  ]
}}

⚠️ 配列の要素数は必ず{len(whisper_segments)}個にしてください。

【重要なルール】
1. 【絶対厳守】出力セグメント数 = {len(whisper_segments)}個（1個も多くせず、1個も少なくしない）
2. 各セグメントのstart/end時間はWhisperのものをそのままコピーする
3. textのみを台本から抽出して置き換える
4. 台本の順序を保ち、Whisperのタイミングに合わせて配分する
5. 句読点も台本通りに含める
6. 各Whisperセグメントに対応する台本部分を特定し、そのテキストを割り当てる
7. もしWhisperセグメント境界が台本の文の途中にある場合：
   - そのWhisperセグメントに対応する台本部分（文の一部でも可）をそのまま割り当てる
   - セグメントを統合せず、分割もせず、{len(whisper_segments)}個を維持する
   - 例：Whisper seg1「...インタビューで」 → 台本「...インタビューで、」
        Whisper seg2「インフレと...」 → 台本「「インフレと生産性の動向に注目している」と述べ、...」

【漢字・表記の訂正について】
Whisperは音声認識のため、同音異義語や漢字の誤認識が発生します。必ず台本の正確な表記を使用してください。
よくある誤認識の例：
- 「市場」(しじょう) と「市場」(いちば) の混同
- 「上場」→「城」「場合」などの誤認識
- 固有名詞（企業名、人名）の誤認識
- 数字の誤認識（「15」→「十五」など）
- 専門用語の誤認識（「量的緩和」→「両手機関は」など）

→ 必ず台本のテキストをそのまま使用してください

【出力例】
Whisperセグメント0: 0-10秒「きょうわけいざいにゅーすです」
→ 出力: {{"index": 0, "text": "今日は経済ニュースです。", "start": 0.0, "end": 10.0}}

Whisperセグメント1: 10-20秒「にちぎんがはっぴょうしました」
→ 出力: {{"index": 1, "text": "日銀が発表しました。", "start": 10.0, "end": 20.0}}

⚠️ 再確認：出力配列の長さは必ず{len(whisper_segments)}個です。多くても少なくてもいけません。
"""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "あなたは字幕編集のプロフェッショナルです。JSON形式で正確に出力してください。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            corrected_segments = result.get("corrected_segments", [])

            # セグメント数のチェック（タイミング精度のため厳格にチェック）
            if len(corrected_segments) != len(whisper_segments):
                logger.warning(
                    f"Segment count mismatch: expected {len(whisper_segments)}, "
                    f"got {len(corrected_segments)}. Falling back to punctuation-based segmentation."
                )
                # セグメント数が異なる場合は、タイミングズレを防ぐためフォールバック
                return self._resegment_by_punctuation(whisper_segments)

            # 補正されたセグメントを返す（タイミングはWhisperのものを使用）
            final_segments = []
            for i, corrected in enumerate(corrected_segments):
                # Whisperのタイミング情報を使用（音声と完全に同期）
                whisper_seg = whisper_segments[i]
                original_text = whisper_seg["text"]
                original_start = whisper_seg["start"]
                original_end = whisper_seg["end"]
                original_duration = original_end - original_start

                # AI補正されたテキストを取得
                corrected_text = corrected["text"]

                # 次のセグメントの開始時間を取得（タイミング延長の判定用）
                next_start = whisper_segments[i + 1]["start"] if i + 1 < len(whisper_segments) else None

                # 補正内容をログ出力（変更があった場合のみ）
                if original_text != corrected_text:
                    logger.info(f"[AI CORRECTION] Segment {i}: '{original_text}' -> '{corrected_text}'")

                # タイミング情報を詳細にログ出力（特に前半10個）
                if i < 10:
                    timing_info = f"[TIMING] Segment {i}:"
                    timing_info += f"\n  Whisper timing: {original_start:.2f}s - {original_end:.2f}s (duration: {original_duration:.2f}s)"
                    if next_start:
                        timing_info += f"\n  Next segment starts at: {next_start:.2f}s"
                    timing_info += f"\n  Corrected text ({len(corrected_text)} chars): '{corrected_text[:50]}...'"
                    logger.info(timing_info)

                # Whisperのタイミングをそのまま使用（タイミングズレを防ぐ）
                final_segments.append({
                    "start": original_start,
                    "end": original_end,
                    "text": corrected_text
                })

            logger.info(f"[AI CORRECTION] Successfully corrected {len(final_segments)} segments")

            # AI補正済みのセグメントをそのまま返す（再分割しない）
            # 句読点での再分割は文字数按分のため、実際の音声とズレが生じる
            return final_segments

        except Exception as e:
            logger.error(f"[AI CORRECTION] Failed: {e}")
            # エラー時はフォールバック
            return self._resegment_by_punctuation(whisper_segments)

    def _create_subtitles_from_script(self, script_text: str, whisper_segments: List[dict]) -> List[dict]:
        """
        台本から字幕を生成（Whisperセグメント単位で時間マッピング）
        Whisperの各セグメントに対応する台本部分を特定し、そのセグメントの時間範囲内で配置

        Args:
            script_text: 台本テキスト
            whisper_segments: Whisperのセグメント（時間情報用）

        Returns:
            台本ベースの字幕セグメント
        """
        logger.info(f"[SUBTITLE] Starting _create_subtitles_from_script with {len(whisper_segments)} whisper segments")

        if not whisper_segments:
            logger.info("[SUBTITLE] No whisper segments, returning empty")
            return []

        import re

        # 台本を句読点で分割
        logger.info(f"[SUBTITLE] Splitting script text ({len(script_text)} chars)...")
        sentences = re.split(r'([。、])', script_text)

        # 句読点を前の文に含める
        merged_sentences = []
        i = 0
        while i < len(sentences):
            if sentences[i]:
                text = sentences[i]
                if i + 1 < len(sentences) and sentences[i + 1] in ['。', '、']:
                    text += sentences[i + 1]
                    i += 2
                else:
                    i += 1
                text = text.strip()
                if text:
                    merged_sentences.append(text)
            else:
                i += 1
        logger.info(f"[SUBTITLE] Merged into {len(merged_sentences)} sentences")

        if not merged_sentences:
            logger.warning("[SUBTITLE] No sentences from script, returning original")
            return whisper_segments

        # 各Whisperセグメントに台本の文を割り当て
        # Whisperセグメント数と台本の文数の比率を計算
        num_whisper_segs = len(whisper_segments)
        num_script_sentences = len(merged_sentences)

        logger.info(f"[SUBTITLE] Mapping {num_script_sentences} script sentences to {num_whisper_segs} Whisper segments")

        # 台本の文をWhisperセグメントに均等に割り当て
        sentences_per_segment = num_script_sentences / num_whisper_segs

        new_segments = []
        script_idx = 0

        for seg_idx, whisper_seg in enumerate(whisper_segments):
            seg_start = whisper_seg["start"]
            seg_end = whisper_seg["end"]
            seg_duration = seg_end - seg_start

            # このWhisperセグメントに割り当てる台本の文数を計算
            start_sentence_idx = int(seg_idx * sentences_per_segment)
            end_sentence_idx = int((seg_idx + 1) * sentences_per_segment)

            # 最後のセグメントは残り全てを含める
            if seg_idx == num_whisper_segs - 1:
                end_sentence_idx = num_script_sentences

            # このセグメントに対応する台本の文を取得
            segment_sentences = merged_sentences[start_sentence_idx:end_sentence_idx]

            if not segment_sentences:
                continue

            # このWhisperセグメントの時間範囲内で、台本の文を文字数比率で配置
            total_chars = sum(len(s) for s in segment_sentences)
            current_time = seg_start

            for sentence in segment_sentences:
                # 文字数比率で時間を計算
                char_ratio = len(sentence) / total_chars if total_chars > 0 else 1.0 / len(segment_sentences)
                duration = seg_duration * char_ratio

                new_segments.append({
                    "start": current_time,
                    "end": current_time + duration,
                    "text": sentence
                })

                current_time += duration

        logger.info(f"[SUBTITLE] Created {len(new_segments)} subtitle segments from script")
        return new_segments

    def _resegment_by_punctuation(self, segments: List[dict]) -> List[dict]:
        """
        句読点（、。）で字幕を再分割
        各Whisperセグメント内で句読点分割を行い、セグメント境界は維持する

        Args:
            segments: 字幕セグメントのリスト

        Returns:
            句読点で再分割された字幕セグメント
        """
        if not segments:
            return segments

        import re
        new_segments = []

        # 各Whisperセグメントを個別に処理
        for seg in segments:
            seg_text = seg["text"]
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_duration = seg_end - seg_start

            if seg_duration <= 0:
                logger.warning(f"Invalid segment duration, skipping: {seg_text[:20]}...")
                continue

            # このセグメント内で句読点分割
            sentences = re.split(r'([。、])', seg_text)

            # 分割結果を結合（句読点を前の文に含める）
            merged_sentences = []
            i = 0
            while i < len(sentences):
                if sentences[i]:  # 空文字列をスキップ
                    text = sentences[i]
                    # 次が句読点なら結合
                    if i + 1 < len(sentences) and sentences[i + 1] in ['。', '、']:
                        text += sentences[i + 1]
                        i += 2
                    else:
                        i += 1

                    text = text.strip()
                    if text:
                        merged_sentences.append(text)
                else:
                    i += 1

            # 句読点がない、または分割できない場合は元のセグメントを使用
            if not merged_sentences:
                new_segments.append(seg)
                continue

            # 分割が1つだけの場合も、テキストは補正済みのものを使用
            if len(merged_sentences) == 1:
                new_segments.append({
                    "start": seg_start,
                    "end": seg_end,
                    "text": merged_sentences[0]
                })
                continue

            # セグメント内の各文に時間を按分
            total_chars = sum(len(s) for s in merged_sentences)
            current_time = seg_start

            for sentence in merged_sentences:
                # 文字数比率で時間を計算
                char_ratio = len(sentence) / total_chars if total_chars > 0 else 0
                duration = seg_duration * char_ratio

                new_segments.append({
                    "start": current_time,
                    "end": current_time + duration,
                    "text": sentence
                })

                current_time += duration

        logger.info(f"Resegmented: {len(segments)} Whisper segments → {len(new_segments)} punctuation-based segments")
        return new_segments

    def _merge_short_segments(self, segments: List[dict]) -> List[dict]:
        """
        短いセグメントや不自然な区切りを結合

        Args:
            segments: 字幕セグメントのリスト

        Returns:
            結合された字幕セグメント
        """
        if not segments:
            return segments

        merged = []
        i = 0

        while i < len(segments):
            current = segments[i]
            current_text = current["text"].strip()

            # 次のセグメントと結合すべきか判定
            should_merge = False

            if i < len(segments) - 1:
                next_seg = segments[i + 1]
                next_text = next_seg["text"].strip()

                # 結合条件1: 現在のセグメントが短すぎる（10文字未満）
                if len(current_text) < 10:
                    should_merge = True
                    logger.debug(f"Merging short segment: '{current_text}' (length: {len(current_text)})")

                # 結合条件1.5: 次のセグメントが非常に短い（3文字以下）
                # 「さん」「です」「ます」など断片的なセグメントを拾う
                if len(next_text) <= 3:
                    should_merge = True
                    logger.debug(f"Merging very short next segment: '{next_text}' (length: {len(next_text)})")

                # 結合条件2: 文が不自然に途切れている（助詞で終わる、補助動詞など）
                # 日本語の接続パターン
                incomplete_endings = [
                    # 助詞
                    'が', 'は', 'を', 'に', 'で', 'と', 'から', 'まで', 'より', 'へ', 'も', 'や',
                    # 接続助詞・活用
                    'て', 'で', 'し', 'た', 'だ', 'な', 'ば', 'ても', 'でも',
                    # 補助動詞の途中（完全な形と途中の形の両方）
                    'ている', 'てい', 'ており', 'ておr', 'てお', 'ています', 'ていま', 'ていm',
                    'している', 'してい', 'してお', 'しています', 'していま', 'してi',
                    'されて', 'されてい', 'されており', 'されています',
                    # 丁寧語の途中
                    'ました', 'まし', 'です', 'でし', 'でした', 'であ', 'であり',
                    # 形式名詞・接続詞
                    'の', 'こと', 'もの', 'ため', 'よう', 'ところ',
                    # 否定・推量
                    'ない', 'なく', 'ず', 'ぬ', 'ん',
                    'だろ', 'でしょ', 'かもし', 'らし',
                    # 動詞の連体形（未然形・連用形含む）
                    'る', 'れ', 'ら', 'り', 'ろ',  # 五段活用の語尾
                    'う', 'く', 'ぐ', 'す', 'つ', 'ぬ', 'ぶ', 'む', 'ゆ', 'る',  # 動詞語幹
                    'ま', 'み', 'め', 'も',  # マ行の活用
                    'な', 'に', 'ね', 'の',  # ナ行の活用
                    'さ', 'せ', 'そ',  # サ行の活用
                    'か', 'き', 'け', 'こ',  # カ行の活用
                ]

                for ending in incomplete_endings:
                    if current_text.endswith(ending):
                        should_merge = True
                        logger.debug(f"Merging incomplete phrase: '{current_text}' + '{next_text}'")
                        break

                # 結合条件2.5: 次のセグメントが補助動詞・語尾で始まる
                incomplete_beginnings = [
                    'ます', 'ました', 'ません', 'ませんでした',
                    'です', 'でした', 'ではありません',
                    'いる', 'いた', 'いない', 'います', 'いました',
                    'おり', 'おる', 'おります',
                    'あり', 'ある', 'あります',
                    'など', 'なども',
                    # 形式名詞（連体形の後に続く）
                    'こと', 'ことが', 'ことで', 'ことに', 'ことを', 'ことは',
                    'もの', 'ものが', 'ものの', 'ものを', 'ものは',
                    'ため', 'ために', 'ための',
                    'よう', 'ように', 'ような',
                    'わけ', 'わけが', 'わけで', 'わけは',
                    'はず', 'はずが', 'はずで', 'はずは',
                    'つもり', 'つもりが', 'つもりで',
                ]

                for beginning in incomplete_beginnings:
                    if next_text.startswith(beginning):
                        should_merge = True
                        logger.debug(f"Merging with auxiliary verb/formal noun beginning: '{current_text}' + '{next_text}'")
                        break

            if should_merge and i < len(segments) - 1:
                # 次のセグメントと結合
                next_seg = segments[i + 1]
                merged_text = current_text + next_seg["text"].strip()

                merged.append({
                    "start": current["start"],
                    "end": next_seg["end"],
                    "text": merged_text
                })

                i += 2  # 2つ消費したので2進める
            else:
                # 結合しない
                merged.append(current)
                i += 1

        logger.info(f"Segment merging: {len(segments)} → {len(merged)} segments")
        return merged

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
        # 短いテキスト（10文字未満）は発音チェックをスキップ（高速化）
        if len(text1) < 10 or len(text2) < 10:
            return 0.0

        try:
            import pykakasi

            # pykakasi を初期化（初回のみ）
            if not hasattr(self, '_kakasi'):
                self._kakasi = pykakasi.kakasi()

            # ひらがなキャッシュ（セッション内で同じテキストの変換を避ける）
            if not hasattr(self, '_hiragana_cache'):
                self._hiragana_cache = {}

            # 両方のテキストをひらがなに変換（キャッシュ使用）
            kana1 = self._to_hiragana_cached(text1)
            kana2 = self._to_hiragana_cached(text2)

            # ひらがなレベルで比較
            if kana1 and kana2:
                similarity = SequenceMatcher(None, kana1, kana2).ratio()
                logger.debug(f"Phonetic comparison: '{text1[:20]}...' vs '{text2[:20]}...' = {similarity:.2f}")
                return similarity

        except ImportError:
            # pykakasi がインストールされていない場合はスキップ
            logger.debug("pykakasi not available, skipping phonetic similarity check")
        except Exception as e:
            logger.debug(f"Error in phonetic similarity calculation: {e}")

        return 0.0

    def _to_hiragana_cached(self, text: str) -> str:
        """
        テキストをひらがなに変換（キャッシュ付き）

        Args:
            text: 変換するテキスト

        Returns:
            ひらがな文字列
        """
        if text in self._hiragana_cache:
            return self._hiragana_cache[text]

        hiragana = self._to_hiragana(text)
        self._hiragana_cache[text] = hiragana
        return hiragana

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
            acronym: アルファベット略語（例: "CZ", "AI", "CEO", "EV"）

        Returns:
            発音パターンのリスト
        """
        # アルファベットの日本語発音マッピング
        pronunciation_map = {
            'A': ['エー', 'エイ', 'A', 'a', 'え', 'えい'],
            'B': ['ビー', 'B', 'b', 'び'],
            'C': ['シー', 'C', 'c', 'し'],
            'D': ['ディー', 'D', 'd', 'でぃ'],
            'E': ['イー', 'E', 'e', 'い', 'え'],
            'F': ['エフ', 'F', 'f', 'えふ'],
            'G': ['ジー', 'G', 'g', 'じ'],
            'H': ['エイチ', 'H', 'h', 'えいち'],
            'I': ['アイ', 'I', 'i', 'あい'],
            'J': ['ジェー', 'J', 'j', 'じぇ'],
            'K': ['ケー', 'K', 'k', 'け'],
            'L': ['エル', 'L', 'l', 'える'],
            'M': ['エム', 'M', 'm', 'えむ'],
            'N': ['エヌ', 'N', 'n', 'えぬ'],
            'O': ['オー', 'O', 'o', 'お'],
            'P': ['ピー', 'P', 'p', 'ぴ'],
            'Q': ['キュー', 'Q', 'q', 'きゅ'],
            'R': ['アール', 'R', 'r', 'ある'],
            'S': ['エス', 'S', 's', 'えす'],
            'T': ['ティー', 'T', 't', 'てぃ'],
            'U': ['ユー', 'U', 'u', 'ゆ'],
            'V': ['ブイ', 'V', 'v', 'ぶい'],
            'W': ['ダブリュー', 'W', 'w', 'だぶりゅ'],
            'X': ['エックス', 'X', 'x', 'えっくす'],
            'Y': ['ワイ', 'Y', 'y', 'わい'],
            'Z': ['ゼット', 'ゼツ', 'ジー', 'Z', 'z', 'ぜっと', 'ぜつ'],
        }

        patterns = []

        # パターン1: 各文字を発音に変換して連結（例: "EV" → "イーブイ"）
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

        # パターン2: カタカナ小文字表記（例: "CZ" → "Cゼツ", "EV" → "Eブイ"）
        if len(acronym) == 2:
            first_char = acronym[0]
            second_pronunciations = pronunciation_map.get(acronym[1].upper(), [])
            for second_pron in second_pronunciations:
                result_patterns.append(f"{first_char}{second_pron}")
                result_patterns.append(f"{first_char.lower()}{second_pron}")

        # パターン3: 完全小文字版（例: "EV" → "ev", "eb"など）
        result_patterns.append(acronym.lower())

        # パターン4: 各文字の小文字組み合わせ（例: "ev", "eb", "iv"など）
        # よくある誤認識パターン: E→e/i, V→v/b
        common_misrecognitions = {
            'E': ['e', 'i'],
            'V': ['v', 'b'],
            'I': ['i', 'l', '1'],
            'O': ['o', '0'],
            'S': ['s', '5'],
            'Z': ['z', '2'],
            'B': ['b', '8'],
            'G': ['g', '9'],
        }

        if len(acronym) == 2:
            first_variations = common_misrecognitions.get(acronym[0].upper(), [acronym[0].lower()])
            second_variations = common_misrecognitions.get(acronym[1].upper(), [acronym[1].lower()])
            for first in first_variations:
                for second in second_variations:
                    result_patterns.append(f"{first}{second}")

        # パターン5: 元のアルファベットそのまま（大文字・小文字）
        result_patterns.append(acronym)
        result_patterns.append(acronym.upper())

        # 重複を除去
        result_patterns = list(set(result_patterns))

        logger.debug(f"Generated pronunciation patterns for '{acronym}': {result_patterns[:10]}... (total: {len(result_patterns)})")
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
