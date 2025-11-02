"""
ユーティリティモジュール
共通処理や補助機能を提供
"""

import os
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    設定ファイルを読み込み

    Args:
        config_path: 設定ファイルのパス

    Returns:
        設定辞書
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def setup_logging(config: Optional[Dict] = None) -> None:
    """
    ロギングを設定

    Args:
        config: ロギング設定辞書（config.yamlのlogging部分）
    """
    if config is None:
        config = {}

    log_level = config.get("level", "INFO")
    log_format = config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file = config.get("file")

    # ログレベルを設定
    level = getattr(logging, log_level.upper(), logging.INFO)

    # ハンドラーを設定
    handlers = []

    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    # ファイルハンドラー
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler

        max_bytes = config.get("max_bytes", 10485760)  # 10MB
        backup_count = config.get("backup_count", 5)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    # ロギングを設定
    logging.basicConfig(level=level, format=log_format, handlers=handlers)

    logger.info("Logging configured successfully")


def setup_output_directories(config: Dict) -> Dict[str, Path]:
    """
    出力ディレクトリを作成

    Args:
        config: 出力設定辞書（config.yamlのoutput部分）

    Returns:
        ディレクトリパスの辞書
    """
    base_dir = Path(config.get("base_dir", "./output"))
    subdirs = config.get("subdirs", {})

    directories = {"base": base_dir}

    # ベースディレクトリを作成
    base_dir.mkdir(parents=True, exist_ok=True)

    # サブディレクトリを作成
    for name, subdir in subdirs.items():
        dir_path = base_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        directories[name] = dir_path

    logger.info(f"Output directories created: {list(directories.keys())}")
    return directories


def get_timestamp(timezone: str = "Asia/Tokyo") -> str:
    """
    現在時刻のタイムスタンプを取得

    Args:
        timezone: タイムゾーン

    Returns:
        ISO形式のタイムスタンプ
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    return now.isoformat()


def get_date_string(timezone: str = "Asia/Tokyo", format: str = "%Y%m%d") -> str:
    """
    現在日付の文字列を取得

    Args:
        timezone: タイムゾーン
        format: 日付フォーマット

    Returns:
        フォーマットされた日付文字列
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    return now.strftime(format)


def get_time_string(timezone: str = "Asia/Tokyo", format: str = "%H%M%S") -> str:
    """
    現在時刻の文字列を取得

    Args:
        timezone: タイムゾーン
        format: 時刻フォーマット

    Returns:
        フォーマットされた時刻文字列
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    return now.strftime(format)


def generate_filename(
    config: Dict, prefix: str = "video", extension: str = "mp4"
) -> str:
    """
    ファイル名を生成

    Args:
        config: 出力設定辞書
        prefix: ファイル名のプレフィックス
        extension: 拡張子

    Returns:
        生成されたファイル名
    """
    filename_format = config.get("filename_format", "{prefix}_{date}_{time}")
    timezone = config.get("timezone", "Asia/Tokyo")

    date_str = get_date_string(timezone)
    time_str = get_time_string(timezone)

    filename = filename_format.format(
        prefix=prefix, date=date_str, time=time_str
    )

    return f"{filename}.{extension}"


def save_json(data: Dict, filepath: str) -> None:
    """
    辞書をJSONファイルとして保存

    Args:
        data: 保存するデータ
        filepath: 保存先ファイルパス
    """
    try:
        # ディレクトリを作成
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"JSON saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        raise


def load_json(filepath: str) -> Dict:
    """
    JSONファイルを読み込み

    Args:
        filepath: 読み込むファイルパス

    Returns:
        読み込んだデータ
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"JSON loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        raise


def cleanup_old_files(directory: Path, days: int = 30) -> int:
    """
    古いファイルを削除

    Args:
        directory: 対象ディレクトリ
        days: 保存期間（日数）

    Returns:
        削除したファイル数
    """
    if not directory.exists():
        return 0

    now = datetime.now()
    deleted_count = 0

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        # ファイルの更新日時を取得
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        age_days = (now - mtime).days

        if age_days > days:
            try:
                file_path.unlink()
                deleted_count += 1
                logger.debug(f"Deleted old file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {e}")

    logger.info(f"Cleaned up {deleted_count} old files from {directory}")
    return deleted_count


def cleanup_temp_files(directories: Dict[str, Path], extensions: list = None) -> int:
    """
    一時ファイルを削除

    Args:
        directories: ディレクトリパスの辞書
        extensions: 削除対象の拡張子リスト（Noneの場合は全て）

    Returns:
        削除したファイル数
    """
    if extensions is None:
        extensions = [".tmp", ".temp", ".cache"]

    deleted_count = 0

    for dir_path in directories.values():
        if not dir_path.exists():
            continue

        for ext in extensions:
            for file_path in dir_path.glob(f"*{ext}"):
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.debug(f"Deleted temp file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting temp file {file_path}: {e}")

    logger.info(f"Cleaned up {deleted_count} temporary files")
    return deleted_count


def send_slack_notification(
    webhook_url: str, message: str, title: Optional[str] = None
) -> bool:
    """
    Slackに通知を送信

    Args:
        webhook_url: Slack Webhook URL
        message: 通知メッセージ
        title: 通知タイトル（オプション）

    Returns:
        成功した場合True
    """
    try:
        import requests

        payload = {"text": message}

        if title:
            payload = {
                "attachments": [
                    {"title": title, "text": message, "color": "good"}
                ]
            }

        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()

        logger.info("Slack notification sent successfully")
        return True

    except Exception as e:
        logger.error(f"Error sending Slack notification: {e}")
        return False


def format_duration(seconds: float) -> str:
    """
    秒数を読みやすい形式に変換

    Args:
        seconds: 秒数

    Returns:
        フォーマットされた時間文字列（例: "3:45"）
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def validate_env_variables(required_vars: list) -> bool:
    """
    必要な環境変数が設定されているか確認

    Args:
        required_vars: 必要な環境変数のリスト

    Returns:
        全て設定されている場合True
    """
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        return False

    logger.info("All required environment variables are set")
    return True


class VideoGenerationContext:
    """動画生成処理のコンテキスト管理クラス"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.directories = setup_output_directories(config.get("output", {}))
        self.timezone = config.get("app", {}).get("timezone", "Asia/Tokyo")
        self.start_time = datetime.now(pytz.timezone(self.timezone))

        # データ保存用
        self.news_article = None
        self.script_data = None
        self.metadata = None
        self.audio_files = []
        self.video_file = None
        self.thumbnail_file = None

    def get_output_paths(self, prefix: str = "video") -> Dict[str, str]:
        """
        出力ファイルパスを生成

        Args:
            prefix: ファイル名のプレフィックス

        Returns:
            パスの辞書
        """
        date_str = get_date_string(self.timezone)
        time_str = get_time_string(self.timezone)
        base_filename = f"{prefix}_{date_str}_{time_str}"

        return {
            "video": str(self.directories["videos"] / f"{base_filename}.mp4"),
            "audio": str(self.directories["audio"] / f"{base_filename}.mp3"),
            "thumbnail": str(
                self.directories["thumbnails"] / f"{base_filename}.jpg"
            ),
            "script": str(
                self.directories["scripts"] / f"{base_filename}_script.json"
            ),
            "metadata": str(
                self.directories["scripts"] / f"{base_filename}_metadata.json"
            ),
        }

    def get_elapsed_time(self) -> str:
        """
        経過時間を取得

        Returns:
            経過時間の文字列
        """
        now = datetime.now(pytz.timezone(self.timezone))
        elapsed = (now - self.start_time).total_seconds()
        return format_duration(elapsed)

    def summary(self) -> Dict[str, Any]:
        """
        処理結果のサマリーを取得

        Returns:
            サマリー辞書
        """
        return {
            "elapsed_time": self.get_elapsed_time(),
            "video_file": self.video_file,
            "thumbnail_file": self.thumbnail_file,
            "title": self.metadata.get("title") if self.metadata else None,
            "duration": (
                self.script_data.get("estimated_duration")
                if self.script_data
                else None
            ),
            "completed_at": get_timestamp(self.timezone),
        }


def main():
    """テスト実行用"""
    from dotenv import load_dotenv

    load_dotenv()

    # 設定読み込み
    config = load_config("config/config.yaml")

    # ロギング設定
    setup_logging(config.get("logging"))

    # ディレクトリ作成
    directories = setup_output_directories(config.get("output"))

    print("\n=== 作成されたディレクトリ ===")
    for name, path in directories.items():
        print(f"{name}: {path}")

    # タイムスタンプ
    print(f"\n現在時刻: {get_timestamp()}")
    print(f"日付: {get_date_string()}")
    print(f"時刻: {get_time_string()}")

    # ファイル名生成
    filename = generate_filename(config.get("output"), "test_video", "mp4")
    print(f"\n生成されたファイル名: {filename}")

    # コンテキスト
    print("\n=== VideoGenerationContext テスト ===")
    context = VideoGenerationContext(config)
    paths = context.get_output_paths("economic_news")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
