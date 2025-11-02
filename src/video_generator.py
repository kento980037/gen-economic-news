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

            if self.text_overlay_config.get("enabled", True):
                if title and self.text_overlay_config.get("show_title", True):
                    title_clip = self._create_title_overlay(title)
                    clips.append(title_clip)

                if keywords and self.text_overlay_config.get("show_keywords", True):
                    keyword_clips = self._create_keyword_overlays(keywords, duration)
                    clips.extend(keyword_clips)

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

    def _create_title_overlay(self, title: str) -> TextClip:
        """タイトルオーバーレイを作成"""
        font_size = self.text_overlay_config.get("font_size", 48)
        color = self.text_overlay_config.get("color", "white")
        title_duration = self.text_overlay_config.get("title_duration", 5)

        # テキストクリップを作成
        txt_clip = TextClip(
            title,
            fontsize=font_size,
            color=color,
            font=self.text_overlay_config.get("font", "Arial-Bold"),
            method="caption",
            size=(self.resolution[0] - 200, None),  # 左右に余白
            align="center",
        )

        # 位置を設定
        position = self.text_overlay_config.get("position", "center")
        if position == "top":
            txt_clip = txt_clip.set_position(("center", 100))
        elif position == "bottom":
            txt_clip = txt_clip.set_position(("center", self.resolution[1] - 200))
        else:  # center
            txt_clip = txt_clip.set_position("center")

        # 表示時間とエフェクトを設定
        txt_clip = txt_clip.set_duration(title_duration)
        txt_clip = fadein(txt_clip, 0.5)
        txt_clip = fadeout(txt_clip, 0.5)

        return txt_clip

    def _create_keyword_overlays(
        self, keywords: List[str], duration: float
    ) -> List[TextClip]:
        """キーワードオーバーレイを作成"""
        clips = []
        font_size = self.text_overlay_config.get("font_size", 48) - 12  # 少し小さく
        color = self.text_overlay_config.get("color", "white")

        # キーワードを順次表示（最大5個）
        display_keywords = keywords[:5]
        interval = min(duration / len(display_keywords), 10)  # 最大10秒間隔

        for i, keyword in enumerate(display_keywords):
            start_time = 5 + i * interval  # タイトル表示後から開始
            if start_time >= duration:
                break

            txt_clip = TextClip(
                f"#{keyword}",
                fontsize=font_size,
                color=color,
                font=self.text_overlay_config.get("font", "Arial-Bold"),
            )

            # 右下に配置
            txt_clip = txt_clip.set_position(
                (self.resolution[0] - txt_clip.w - 50, self.resolution[1] - 100)
            )

            # 表示時間を設定
            txt_clip = txt_clip.set_start(start_time)
            txt_clip = txt_clip.set_duration(min(5, duration - start_time))
            txt_clip = fadein(txt_clip, 0.3)
            txt_clip = fadeout(txt_clip, 0.3)

            clips.append(txt_clip)

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

            # フォントを読み込み（デフォルトフォント使用）
            try:
                # システムフォントを使用（環境に応じて調整が必要）
                font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", font_size)
            except:
                logger.warning("Could not load font, using default")
                font = ImageFont.load_default()

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
            try:
                badge_font = ImageFont.truetype(
                    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", badge_font_size
                )
            except:
                badge_font = ImageFont.load_default()

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
