"""
動画生成モジュール
音声ファイルと設定から動画ファイルを生成
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_audioclips,
)
from moviepy.video.fx.all import fadein, fadeout
from PIL import Image, ImageDraw, ImageFont
import numpy as np

logger = logging.getLogger(__name__)


class VideoGenerator:
    """動画生成クラス"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書（config.yamlから読み込んだvideo設定）
        """
        self.config = config
        self.resolution = (
            config.get("resolution", {}).get("width", 1920),
            config.get("resolution", {}).get("height", 1080),
        )
        self.fps = config.get("fps", 30)
        self.background_config = config.get("background", {})
        self.text_overlay_config = config.get("text_overlay", {})
        self.thumbnail_config = config.get("thumbnail", {})

    def create_video(
        self,
        audio_files: List[str],
        output_path: str,
        title: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        subtitles: Optional[List[Dict]] = None,
    ) -> str:
        """
        音声ファイルから動画を生成

        Args:
            audio_files: 音声ファイルパスのリスト
            output_path: 出力動画ファイルパス
            title: 動画タイトル（オーバーレイ表示用）
            keywords: キーワードリスト（オーバーレイ表示用）

        Returns:
            生成された動画ファイルのパス
        """
        logger.info(f"Creating video with {len(audio_files)} audio file(s)")

        try:
            # 出力ディレクトリを作成
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # 音声クリップを読み込み
            if len(audio_files) == 1:
                audio_clip = AudioFileClip(audio_files[0])
            else:
                # 複数の音声ファイルを結合
                audio_clips = [AudioFileClip(f) for f in audio_files]
                audio_clip = concatenate_audioclips(audio_clips)

            duration = audio_clip.duration
            logger.info(f"Total audio duration: {duration:.2f} seconds")

            # 背景を作成
            background = self._create_background(duration)

            # テキストオーバーレイを作成
            clips = [background]

            # 字幕を追加（最優先）
            if subtitles:
                subtitle_clips = self._create_subtitle_overlays(subtitles, duration)
                clips.extend(subtitle_clips)

            if self.text_overlay_config.get("enabled", True):
                if title and self.text_overlay_config.get("show_title", True):
                    title_clip = self._create_title_overlay(title)
                    clips.append(title_clip)

                # キーワードオーバーレイは無効化
                # if keywords and self.text_overlay_config.get("show_keywords", True):
                #     keyword_clips = self._create_keyword_overlays(keywords, duration)
                #     clips.extend(keyword_clips)

            # クリップを合成
            video = CompositeVideoClip(clips, size=self.resolution)

            # 音声を設定
            video = video.set_audio(audio_clip)
            video = video.set_duration(duration)

            # 動画を書き出し
            logger.info(f"Writing video to: {output_path}")
            video.write_videofile(
                output_path,
                fps=self.fps,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                logger=None,  # MoviePyのログを抑制
            )

            # クリーンアップ
            video.close()
            audio_clip.close()

            logger.info(f"Video created successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating video: {e}")
            raise

    def _create_background(self, duration: float):
        """背景クリップを作成"""
        bg_type = self.background_config.get("type", "gradient")
        width, height = self.resolution

        if bg_type == "solid":
            # 単色背景
            color = self._hex_to_rgb(self.background_config.get("color1", "#1a1a2e"))
            background = ColorClip(size=self.resolution, color=color, duration=duration)

        elif bg_type == "gradient":
            # グラデーション背景
            color1 = self._hex_to_rgb(self.background_config.get("color1", "#1a1a2e"))
            color2 = self._hex_to_rgb(self.background_config.get("color2", "#16213e"))
            direction = self.background_config.get("direction", "vertical")

            gradient_img = self._create_gradient_image(
                self.resolution, color1, color2, direction
            )
            background = ImageClip(gradient_img, duration=duration)

        elif bg_type == "image":
            # 画像背景
            image_path = self.background_config.get("image_path")
            if image_path and os.path.exists(image_path):
                background = ImageClip(image_path, duration=duration)
                background = background.resize(self.resolution)
            else:
                logger.warning(f"Background image not found: {image_path}")
                # フォールバックとして単色背景
                color = self._hex_to_rgb(self.background_config.get("color1", "#1a1a2e"))
                background = ColorClip(
                    size=self.resolution, color=color, duration=duration
                )

        else:
            # デフォルトは黒背景
            background = ColorClip(size=self.resolution, color=(0, 0, 0), duration=duration)

        return background.set_fps(self.fps)

    def _create_gradient_image(
        self, size: Tuple[int, int], color1: Tuple, color2: Tuple, direction: str
    ) -> np.ndarray:
        """グラデーション画像を生成"""
        width, height = size
        image = np.zeros((height, width, 3), dtype=np.uint8)

        if direction == "horizontal":
            for x in range(width):
                ratio = x / width
                color = tuple(
                    int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2)
                )
                image[:, x] = color

        elif direction == "vertical":
            for y in range(height):
                ratio = y / height
                color = tuple(
                    int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2)
                )
                image[y, :] = color

        elif direction == "diagonal":
            for y in range(height):
                for x in range(width):
                    ratio = (x + y) / (width + height)
                    color = tuple(
                        int(c1 * (1 - ratio) + c2 * ratio)
                        for c1, c2 in zip(color1, color2)
                    )
                    image[y, x] = color

        return image

    def _create_title_overlay(self, title: str) -> ImageClip:
        """タイトルオーバーレイを作成（Pillowで画像生成）"""
        font_size = self.text_overlay_config.get("font_size", 48)
        color = self.text_overlay_config.get("color", "white")
        title_duration = self.text_overlay_config.get("title_duration", 5)

        # 色を変換
        if isinstance(color, str):
            if color.lower() == "white":
                text_color = (255, 255, 255, 255)
            elif color.lower() == "black":
                text_color = (0, 0, 0, 255)
            else:
                # 16進数カラーコード
                rgb = self._hex_to_rgb(color)
                text_color = (*rgb, 255)
        else:
            text_color = (*color, 255)

        # 画像を作成
        img = self._create_text_image(
            title,
            font_size,
            text_color,
            max_width=self.resolution[0] - 200,
        )

        # ImageClipを作成
        txt_clip = ImageClip(img, duration=title_duration)

        # 位置を設定
        position = self.text_overlay_config.get("position", "center")
        if position == "top":
            txt_clip = txt_clip.set_position(("center", 100))
        elif position == "bottom":
            txt_clip = txt_clip.set_position(("center", self.resolution[1] - 200))
        else:  # center
            txt_clip = txt_clip.set_position("center")

        # エフェクトを設定
        txt_clip = fadein(txt_clip, 0.5)
        txt_clip = fadeout(txt_clip, 0.5)

        return txt_clip

    def _create_keyword_overlays(
        self, keywords: List[str], duration: float
    ) -> List[ImageClip]:
        """キーワードオーバーレイを作成（Pillowで画像生成）"""
        clips = []
        font_size = self.text_overlay_config.get("font_size", 48) - 12  # 少し小さく
        color = self.text_overlay_config.get("color", "white")

        # 色を変換
        if isinstance(color, str):
            if color.lower() == "white":
                text_color = (255, 255, 255, 255)
            elif color.lower() == "black":
                text_color = (0, 0, 0, 255)
            else:
                rgb = self._hex_to_rgb(color)
                text_color = (*rgb, 255)
        else:
            text_color = (*color, 255)

        # キーワードを順次表示（最大5個）
        display_keywords = keywords[:5]
        interval = min(duration / len(display_keywords), 10)  # 最大10秒間隔

        for i, keyword in enumerate(display_keywords):
            start_time = 5 + i * interval  # タイトル表示後から開始
            if start_time >= duration:
                break

            # キーワード画像を生成
            keyword_img = self._create_text_image(
                f"#{keyword}",
                font_size,
                text_color,
                add_background=True,
            )

            # ImageClipを作成
            txt_clip = ImageClip(keyword_img)

            # 右下に配置
            txt_clip = txt_clip.set_position(
                (self.resolution[0] - keyword_img.shape[1] - 50, self.resolution[1] - 100)
            )

            # 表示時間を設定
            txt_clip = txt_clip.set_start(start_time)
            txt_clip = txt_clip.set_duration(min(5, duration - start_time))
            txt_clip = fadein(txt_clip, 0.3)
            txt_clip = fadeout(txt_clip, 0.3)

            clips.append(txt_clip)

        return clips

    def _create_subtitle_overlays(
        self, subtitles: List[Dict], duration: float
    ) -> List[ImageClip]:
        """
        字幕オーバーレイを作成

        Args:
            subtitles: タイムスタンプ付き字幕リスト
                      [{"start": 0.0, "end": 2.5, "text": "こんにちは"}, ...]
            duration: 動画の長さ

        Returns:
            字幕クリップのリスト
        """
        clips = []
        font_size = 40  # 字幕用のフォントサイズ
        text_color = (255, 255, 255, 255)  # 白

        logger.info(f"Creating {len(subtitles)} subtitle clips")

        for subtitle in subtitles:
            start_time = subtitle.get("start", 0)
            end_time = subtitle.get("end", 0)
            text = subtitle.get("text", "")

            if not text or end_time <= start_time:
                continue

            # 字幕が動画の長さを超える場合は調整
            if start_time >= duration:
                continue
            if end_time > duration:
                end_time = duration

            # 字幕画像を生成
            subtitle_img = self._create_text_image(
                text,
                font_size,
                text_color,
                max_width=self.resolution[0] - 100,
                add_background=True,
            )

            # ImageClipを作成
            subtitle_clip = ImageClip(subtitle_img)

            # 画面下部中央に配置
            subtitle_clip = subtitle_clip.set_position(
                ("center", self.resolution[1] - subtitle_img.shape[0] - 50)
            )

            # 表示時間を設定
            subtitle_clip = subtitle_clip.set_start(start_time)
            subtitle_clip = subtitle_clip.set_duration(end_time - start_time)

            clips.append(subtitle_clip)

        logger.info(f"Created {len(clips)} subtitle clips")
        return clips

    def create_thumbnail(self, title: str, output_path: str) -> str:
        """
        サムネイル画像を生成

        Args:
            title: 動画タイトル
            output_path: 出力画像ファイルパス

        Returns:
            生成されたサムネイル画像のパス
        """
        logger.info(f"Creating thumbnail: {title}")

        try:
            # 出力ディレクトリを作成
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # サムネイルサイズ
            width = self.thumbnail_config.get("width", 1280)
            height = self.thumbnail_config.get("height", 720)

            # 画像を作成
            img = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(img)

            # 背景色
            bg_color = self._hex_to_rgb(
                self.thumbnail_config.get("background_color", "#0f3460")
            )
            draw.rectangle([(0, 0), (width, height)], fill=bg_color)

            # タイトルを描画
            font_size = self.thumbnail_config.get("font_size", 72)
            text_color = self._hex_to_rgb(
                self.thumbnail_config.get("text_color", "#ffffff")
            )

            # タイトルを折り返し
            max_title_length = self.thumbnail_config.get("max_title_length", 40)
            if len(title) > max_title_length:
                title = title[:max_title_length] + "..."

            # フォントを読み込み
            font = self._load_japanese_font(font_size)

            # テキストの位置を計算
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = (width - text_width) // 2
            text_y = (height - text_height) // 2

            # 影を追加
            shadow_offset = 4
            draw.text(
                (text_x + shadow_offset, text_y + shadow_offset),
                title,
                font=font,
                fill=(0, 0, 0, 128),
            )

            # テキストを描画
            draw.text((text_x, text_y), title, font=font, fill=text_color)

            # "経済ニュース" バッジを追加
            badge_text = "経済ニュース"
            badge_font_size = font_size // 3
            badge_font = self._load_japanese_font(badge_font_size)

            bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
            badge_width = bbox[2] - bbox[0]
            badge_height = bbox[3] - bbox[1]
            badge_x = 50
            badge_y = 50

            # バッジ背景
            draw.rectangle(
                [
                    (badge_x - 10, badge_y - 10),
                    (badge_x + badge_width + 10, badge_y + badge_height + 10),
                ],
                fill=(255, 69, 0),
            )
            draw.text((badge_x, badge_y), badge_text, font=badge_font, fill=text_color)

            # 画像を保存
            img.save(output_path, quality=95)

            logger.info(f"Thumbnail created: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            raise

    def _create_text_image(
        self,
        text: str,
        font_size: int,
        text_color: Tuple[int, int, int, int],
        max_width: Optional[int] = None,
        add_background: bool = False,
    ) -> np.ndarray:
        """
        Pillowを使ってテキスト画像を生成

        Args:
            text: 表示するテキスト
            font_size: フォントサイズ
            text_color: テキスト色 (R, G, B, A)
            max_width: 最大幅（折り返し用）
            add_background: 背景を追加するか

        Returns:
            numpy配列の画像
        """
        font = self._load_japanese_font(font_size)

        # テキストサイズを計算
        temp_img = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(temp_img)

        # 複数行対応（日本語対応版）
        lines = []
        if max_width:
            # 日本語の場合は文字単位で折り返す
            current_line = ""
            for char in text:
                test_line = current_line + char
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = char
            if current_line:
                lines.append(current_line)
        else:
            lines = [text]

        # 全体のサイズを計算
        max_line_width = 0
        total_height = 0
        line_heights = []
        line_spacing = font_size // 4  # 行間

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
            max_line_width = max(max_line_width, line_width)
            line_heights.append(line_height)
            total_height += line_height

        # 行間を追加
        if len(lines) > 1:
            total_height += line_spacing * (len(lines) - 1)

        # 余白を追加
        padding = 20
        img_width = max_line_width + padding * 2
        img_height = total_height + padding * 2

        # 背景付きの場合
        if add_background:
            img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 180))
        else:
            img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))

        draw = ImageDraw.Draw(img)

        # テキストを描画
        y_offset = padding
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            x = (img_width - line_width) // 2

            # 影を追加
            draw.text((x + 2, y_offset + 2), line, font=font, fill=(0, 0, 0, 200))
            # テキスト本体
            draw.text((x, y_offset), line, font=font, fill=text_color)

            y_offset += line_heights[i] + (line_spacing if i < len(lines) - 1 else 0)

        # numpy配列に変換
        return np.array(img)

    def _load_japanese_font(self, font_size: int):
        """
        日本語フォントを読み込み

        Args:
            font_size: フォントサイズ

        Returns:
            フォントオブジェクト
        """
        # 複数のフォントパスを試す（環境に応じて）
        font_paths = [
            "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",  # macOS
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",  # Linux (Noto)
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",  # Linux (Noto alt)
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux (Noto Regular)
        ]

        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    logger.debug(f"Loading font: {font_path}")
                    return ImageFont.truetype(font_path, font_size)
            except Exception as e:
                logger.debug(f"Could not load font {font_path}: {e}")
                continue

        # フォールバック: デフォルトフォント
        logger.warning("Could not load Japanese font, using default font")
        return ImageFont.load_default()

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """16進数カラーコードをRGBタプルに変換"""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def main():
    """テスト実行用"""
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    # 設定読み込み
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO)

    generator = VideoGenerator(config["video"])

    # サムネイル生成テスト
    print("\n=== サムネイル生成テスト ===")
    thumbnail_path = "./output/thumbnails/test_thumbnail.jpg"
    generator.create_thumbnail("日銀が政策金利を引き上げ、今後の経済への影響は?", thumbnail_path)
    print(f"サムネイル生成完了: {thumbnail_path}")

    # 動画生成テスト（音声ファイルが必要）
    # audio_files = ["./output/audio/test_voice.mp3"]
    # if os.path.exists(audio_files[0]):
    #     print("\n=== 動画生成テスト ===")
    #     video_path = "./output/videos/test_video.mp4"
    #     generator.create_video(
    #         audio_files,
    #         video_path,
    #         title="日銀が政策金利を引き上げ",
    #         keywords=["金融政策", "日銀", "金利", "インフレ"]
    #     )
    #     print(f"動画生成完了: {video_path}")


if __name__ == "__main__":
    main()
