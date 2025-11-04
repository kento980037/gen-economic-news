#!/usr/bin/env python3
"""
経済ニュース動画自動生成システム - メインスクリプト

使用方法:
    python main.py              # 通常実行（最新ニュースから動画を1本生成）
    python main.py --dry-run    # ドライラン（実際の生成は行わない）
    python main.py --test       # テストモード（サンプルデータで動作確認）
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent))

# 各モジュールをインポート
from news_fetcher import NewsFetcher
from script_generator import ScriptGenerator
from voice_generator import VoiceGenerator
from video_generator import VideoGenerator
from metadata_generator import MetadataGenerator
from utils import (
    load_config,
    setup_logging,
    setup_output_directories,
    save_json,
    cleanup_temp_files,
    send_slack_notification,
    validate_env_variables,
    VideoGenerationContext,
)

logger = logging.getLogger(__name__)


class VideoGenerationPipeline:
    """動画生成パイプライン"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        # プロジェクトルートを基準にパスを解決
        if not os.path.isabs(config_path):
            # 相対パスの場合、srcディレクトリの親（プロジェクトルート）を基準にする
            project_root = Path(__file__).parent.parent
            config_path = str(project_root / config_path)

        # 設定読み込み
        self.config = load_config(config_path)

        # ロギング設定
        setup_logging(self.config.get("logging", {}))

        # 環境変数チェック
        required_vars = ["OPENAI_API_KEY"]
        if not validate_env_variables(required_vars):
            raise ValueError("Required environment variables are not set")

        # 各モジュールを初期化
        self.news_fetcher = NewsFetcher(self.config["news"])
        self.script_generator = ScriptGenerator(self.config["script"])
        self.voice_generator = VoiceGenerator(self.config["voice"])
        self.video_generator = VideoGenerator(self.config["video"])
        self.metadata_generator = MetadataGenerator(self.config["metadata"])

        # コンテキストを初期化
        self.context = VideoGenerationContext(self.config)

        logger.info("VideoGenerationPipeline initialized")

    def run(self, dry_run: bool = False) -> Optional[str]:
        """
        動画生成パイプラインを実行

        Args:
            dry_run: True の場合、実際の生成は行わずログのみ出力

        Returns:
            生成された動画ファイルのパス、または None
        """
        try:
            logger.info("=" * 60)
            logger.info("Starting video generation pipeline")
            logger.info("=" * 60)

            # 1. ニュース取得
            logger.info("\n[Step 1/6] Fetching news...")
            news_article = self._fetch_news()
            if not news_article:
                logger.error("No news articles found")
                return None

            self.context.news_article = news_article.to_dict()
            logger.info(f"Selected news: {news_article.title}")

            if dry_run:
                logger.info("[DRY RUN] Skipping actual generation")
                return None

            # 2. 台本生成
            logger.info("\n[Step 2/6] Generating script...")
            script_data = self._generate_script(news_article)
            self.context.script_data = script_data
            logger.info(
                f"Script generated: {len(script_data['script'])} characters, "
                f"~{script_data['estimated_duration']}s"
            )

            # 3. 音声生成
            logger.info("\n[Step 3/6] Generating voice...")
            audio_files = self._generate_voice(script_data)
            self.context.audio_files = audio_files
            logger.info(f"Voice generated: {len(audio_files)} file(s)")

            # 4. 動画生成
            logger.info("\n[Step 4/6] Generating video...")
            video_file = self._generate_video(script_data, audio_files)
            self.context.video_file = video_file
            logger.info(f"Video generated: {video_file}")

            # 5. メタデータ生成
            logger.info("\n[Step 5/6] Generating metadata...")
            metadata = self._generate_metadata(script_data, news_article)
            self.context.metadata = metadata
            logger.info(f"Metadata generated: {metadata['title']}")

            # 6. サムネイル生成
            logger.info("\n[Step 6/6] Generating thumbnail...")
            thumbnail_file = self._generate_thumbnail(metadata)
            self.context.thumbnail_file = thumbnail_file
            logger.info(f"Thumbnail generated: {thumbnail_file}")

            # データを保存
            self._save_data()

            # クリーンアップ
            self._cleanup()

            # 完了通知
            self._send_completion_notification()

            logger.info("\n" + "=" * 60)
            logger.info("Video generation completed successfully!")
            logger.info(f"Elapsed time: {self.context.get_elapsed_time()}")
            logger.info("=" * 60)

            return video_file

        except Exception as e:
            logger.error(f"Error in video generation pipeline: {e}", exc_info=True)
            self._send_error_notification(str(e))
            raise

    def _fetch_news(self):
        """ニュースを取得"""
        # 最新のメイン記事を取得
        main_article = self.news_fetcher.get_top_news()

        if not main_article:
            return None

        return main_article

    def _generate_script(self, news_article):
        """台本を生成（同じトピックの複数ソース参照）"""
        target_duration = self.config.get("app", {}).get("target_duration", 180)

        # 同じトピックに関する関連記事を取得
        related_articles = self.news_fetcher.get_related_articles(news_article, max_related=3)

        # 関連記事を参考情報として追加
        additional_context = None
        if related_articles:
            additional_sources = []
            additional_sources.append(
                f"【メイン記事】{news_article.title} (出典: {news_article.source})\n"
            )
            for i, article in enumerate(related_articles, 1):
                additional_sources.append(
                    f"\n【参考記事{i}】\n"
                    f"タイトル: {article.title}\n"
                    f"出典: {article.source}\n"
                    f"要約: {article.summary[:300]}...\n"
                    f"URL: {article.url}"
                )

            additional_context = (
                "以下は同じトピックについて複数のメディアが報道した記事です。\n"
                "これらの情報を総合的に分析して、多角的な視点から解説してください。\n\n"
                + "\n".join(additional_sources)
            )

            logger.info(
                f"Using {len(related_articles) + 1} articles (1 main + {len(related_articles)} related) "
                f"for script generation on topic: {news_article.title[:50]}..."
            )
        else:
            logger.info(
                f"No related articles found for: {news_article.title[:50]}... "
                f"Using single source only"
            )

        return self.script_generator.generate_script(
            news_article.to_dict(),
            target_duration=target_duration,
            additional_context=additional_context
        )

    def _generate_voice(self, script_data):
        """音声を生成"""
        paths = self.context.get_output_paths()
        audio_dir = Path(paths["audio"]).parent

        return self.voice_generator.generate_voice_with_auto_split(
            script_data["script"], str(audio_dir), "voice"
        )

    def _generate_video(self, script_data, audio_files):
        """動画を生成"""
        paths = self.context.get_output_paths()

        # Whisperで音声から字幕を生成（台本を使って補正）
        logger.info("Generating subtitles from audio with script-based correction...")
        subtitles = []
        try:
            # 字幕設定を取得
            subtitle_config = self.config.get("video", {}).get("subtitle", {})

            # 最初の音声ファイルから字幕を取得
            # 台本テキストを渡して認識精度を向上
            if audio_files:
                subtitles = self.voice_generator.transcribe_audio_with_timestamps(
                    audio_files[0],
                    script_text=script_data.get("script"),  # 台本を渡す
                    subtitle_config=subtitle_config  # 字幕設定を渡す
                )
                logger.info(f"Generated {len(subtitles)} subtitle segments with script correction")
        except Exception as e:
            logger.warning(f"Could not generate subtitles: {e}")
            # 字幕生成に失敗しても動画生成は続行

        return self.video_generator.create_video(
            audio_files,
            paths["video"],
            title=script_data["title"],
            keywords=script_data["keywords"],
            subtitles=subtitles,
        )

    def _generate_metadata(self, script_data, news_article):
        """メタデータを生成"""
        return self.metadata_generator.generate_metadata(
            script_data, news_article.to_dict()
        )

    def _generate_thumbnail(self, metadata):
        """サムネイルを生成"""
        paths = self.context.get_output_paths()
        return self.video_generator.create_thumbnail(metadata["title"], paths["thumbnail"])

    def _save_data(self):
        """生成データを保存"""
        paths = self.context.get_output_paths()

        # 台本を保存
        if self.context.script_data:
            save_json(self.context.script_data, paths["script"])

        # メタデータを保存
        if self.context.metadata:
            save_json(self.context.metadata, paths["metadata"])

        # サマリーを保存
        summary = self.context.summary()
        summary_path = paths["metadata"].replace("_metadata.json", "_summary.json")
        save_json(summary, summary_path)

        logger.info("Data saved successfully")

    def _cleanup(self):
        """クリーンアップ処理"""
        if self.config.get("output", {}).get("cleanup_temp_files", True):
            cleanup_temp_files(self.context.directories)

        # 古いファイルを削除
        retention_days = self.config.get("output", {}).get("retention_days", 30)
        if retention_days > 0:
            from utils import cleanup_old_files

            for dir_path in self.context.directories.values():
                cleanup_old_files(dir_path, retention_days)

    def _send_completion_notification(self):
        """完了通知を送信"""
        notification_config = self.config.get("notification", {}).get("slack", {})

        if not notification_config.get("enabled", False):
            return

        if not notification_config.get("notify_on_success", True):
            return

        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not set, skipping notification")
            return

        summary = self.context.summary()

        message = f"""
動画生成が完了しました

タイトル: {summary.get('title', 'N/A')}
動画時間: {summary.get('duration', 'N/A')}秒
処理時間: {summary['elapsed_time']}
動画ファイル: {summary.get('video_file', 'N/A')}
"""

        send_slack_notification(webhook_url, message, "動画生成完了")

    def _send_error_notification(self, error_message: str):
        """エラー通知を送信"""
        notification_config = self.config.get("notification", {}).get("slack", {})

        if not notification_config.get("enabled", False):
            return

        if not notification_config.get("notify_on_failure", True):
            return

        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        if not webhook_url:
            return

        message = f"""
動画生成でエラーが発生しました

エラー: {error_message}
"""

        send_slack_notification(webhook_url, message, "動画生成エラー")


def main():
    """メイン関数"""
    # 環境変数を読み込み
    load_dotenv()

    # コマンドライン引数を解析
    parser = argparse.ArgumentParser(
        description="経済ニュース動画自動生成システム"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ドライラン（実際の生成は行わない）",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="テストモード（サンプルデータで動作確認）",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="設定ファイルのパス",
    )

    args = parser.parse_args()

    try:
        # パイプラインを作成
        pipeline = VideoGenerationPipeline(args.config)

        # 実行
        if args.test:
            logger.info("Running in TEST mode")
            # テストモード: サンプルデータで実行
            # TODO: テスト用のサンプルデータを使った実行を実装
            logger.warning("Test mode is not fully implemented yet")
            return

        video_file = pipeline.run(dry_run=args.dry_run)

        if video_file:
            print(f"\n生成された動画: {video_file}")
            sys.exit(0)
        else:
            print("\n動画の生成に失敗しました")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
