# 経済ニュース動画自動生成システム

AIを活用して経済ニュースの解説動画を自動生成するシステムです。Claude APIで台本を生成し、OpenAI TTSで音声を作成、MoviePyで動画を合成します。

## 特徴

- 最新の経済ニュースを自動取得（RSS/NewsAPI）
- Claude APIによる自然な日本語台本の自動生成
- OpenAI TTS APIによる高品質な音声合成
- 動画・サムネイル・メタデータの自動生成
- Renderでのcron実行またはGitHub Actionsでのスケジュール実行
- Slack通知機能

## システム構成

```
gen-economic-news/
├── src/
│   ├── main.py              # メインエントリポイント
│   ├── news_fetcher.py      # ニュース取得
│   ├── script_generator.py  # 台本生成（Claude API）
│   ├── voice_generator.py   # 音声生成（OpenAI TTS）
│   ├── video_generator.py   # 動画生成（MoviePy）
│   ├── metadata_generator.py # メタデータ生成
│   └── utils.py             # ユーティリティ
├── config/
│   └── config.yaml          # 設定ファイル
├── output/                  # 生成ファイルの出力先
│   ├── videos/             # 動画ファイル
│   ├── audio/              # 音声ファイル
│   ├── thumbnails/         # サムネイル画像
│   ├── scripts/            # 台本・メタデータ
│   └── logs/               # ログファイル
├── requirements.txt         # Python依存パッケージ
├── render.yaml              # Render設定
└── .github/workflows/       # GitHub Actions設定
```

## 必要要件

- Python 3.11以上
- FFmpeg
- 以下のAPIキー:
  - Anthropic Claude API
  - OpenAI API
  - NewsAPI（オプション）

## セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/yourusername/gen-economic-news.git
cd gen-economic-news
```

### 2. Python環境のセットアップ

```bash
# 仮想環境を作成（推奨）
python -m venv venv

# 仮想環境を有効化
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 依存パッケージをインストール
pip install -r requirements.txt
```

### 3. システム依存パッケージのインストール

```bash
# macOS (Homebrew使用)
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg

# Windows
# https://ffmpeg.org/download.html からダウンロード
```

### 4. 環境変数の設定

`.env.example`をコピーして`.env`を作成し、APIキーを設定:

```bash
cp .env.example .env
```

`.env`ファイルを編集:

```env
# 必須
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# オプション
NEWS_API_KEY=your_newsapi_key_here
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
```

### 5. 設定ファイルのカスタマイズ

`config/config.yaml`を必要に応じて編集してください。主な設定項目:

- ニュースソース（RSS URLs、NewsAPI設定）
- 台本のスタイル・長さ
- 音声の声質・速度
- 動画の解像度・背景色
- メタデータのスタイル

## 使用方法

### ローカル実行

```bash
# 基本実行
cd src
python main.py

# ドライラン（実際の生成は行わない）
python main.py --dry-run

# 設定ファイルを指定
python main.py --config ../config/config.yaml
```

### 個別モジュールのテスト

各モジュールは単独でテスト実行できます:

```bash
cd src

# ニュース取得テスト
python news_fetcher.py

# 台本生成テスト
python script_generator.py

# 音声生成テスト
python voice_generator.py

# 動画生成テスト
python video_generator.py

# メタデータ生成テスト
python metadata_generator.py

# ユーティリティテスト
python utils.py
```

## Renderでのデプロイ

### 方法1: Render Cron Job（推奨）

1. [Render](https://render.com)にサインアップ
2. 新しいCron Jobを作成
3. このリポジトリを接続
4. `render.yaml`が自動検出される
5. 環境変数を設定:
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
   - `NEWS_API_KEY`（オプション）
   - `SLACK_WEBHOOK_URL`（オプション）

6. デプロイを実行

### スケジュール設定

`render.yaml`の`schedule`を編集してcronスケジュールを変更:

```yaml
schedule: "0 9 * * *"  # 毎日午前9時
# または
schedule: "0 */6 * * *"  # 6時間ごと
```

### 注意事項

- Renderの無料プランでは永続ストレージが利用できません
- 生成した動画を保存する場合は、S3やGoogle Cloud Storageなどの外部ストレージを使用してください
- 有料プランでは永続ディスクが利用可能です

## GitHub Actionsでのデプロイ（代替案）

Renderの代わりにGitHub Actionsを使用することもできます。

### 1. GitHub Secretsを設定

リポジトリの Settings > Secrets and variables > Actions で以下を設定:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `NEWS_API_KEY`（オプション）
- `SLACK_WEBHOOK_URL`（オプション）

### 2. ワークフローの有効化

`.github/workflows/generate-video.yml`が自動的に実行されます。

### 3. スケジュール変更

`.github/workflows/generate-video.yml`の`schedule`を編集:

```yaml
schedule:
  - cron: '0 0 * * *'  # 毎日午前0時UTC（午前9時JST）
```

### 4. 手動実行

GitHub ActionsのUIから手動で実行することもできます:

1. リポジトリの「Actions」タブを開く
2. 「Generate Economic News Video」を選択
3. 「Run workflow」をクリック

## 出力ファイル

生成されるファイル:

```
output/
├── videos/
│   └── economic_news_20250102_090000.mp4    # 動画ファイル
├── audio/
│   └── voice_000.mp3                        # 音声ファイル
├── thumbnails/
│   └── economic_news_20250102_090000.jpg    # サムネイル
└── scripts/
    ├── economic_news_20250102_090000_script.json     # 台本データ
    ├── economic_news_20250102_090000_metadata.json   # メタデータ
    └── economic_news_20250102_090000_summary.json    # サマリー
```

## カスタマイズ

### 台本のスタイル変更

`config/config.yaml`の`script`セクション:

```yaml
script:
  tone: "professional"  # professional, casual, friendly
  style: "narration"    # narration, dialogue
  min_length: 800
  max_length: 2000
```

### 音声の変更

`config/config.yaml`の`voice`セクション:

```yaml
voice:
  model: "tts-1-hd"  # tts-1 または tts-1-hd
  voice: "alloy"     # alloy, echo, fable, onyx, nova, shimmer
  speed: 1.0         # 0.25 to 4.0
```

### 動画デザインの変更

`config/config.yaml`の`video`セクション:

```yaml
video:
  resolution:
    width: 1920
    height: 1080
  background:
    type: "gradient"  # solid, gradient, image
    color1: "#1a1a2e"
    color2: "#16213e"
```

## トラブルシューティング

### FFmpegのエラー

```bash
# FFmpegがインストールされているか確認
ffmpeg -version

# パスが通っているか確認
which ffmpeg
```

### APIエラー

- APIキーが正しく設定されているか確認
- APIの利用制限を超えていないか確認
- ログファイル（`output/logs/app.log`）を確認

### メモリ不足

- 動画の解像度を下げる
- 音声を分割して生成する
- サーバーのメモリを増やす

### 日本語フォントが見つからない

動画生成時にフォントエラーが出る場合:

```python
# video_generator.py の該当箇所を編集
font = ImageFont.truetype("/path/to/your/japanese/font.ttc", font_size)
```

## ライセンス

MIT License

## 参考リンク

- [Claude API Documentation](https://docs.anthropic.com/)
- [OpenAI TTS Documentation](https://platform.openai.com/docs/guides/text-to-speech)
- [MoviePy Documentation](https://zulko.github.io/moviepy/)
- [Render Documentation](https://render.com/docs)

## 謝辞

このプロジェクトは以下の記事を参考にしています:
- [ZennのXTM記事](https://zenn.dev/xtm_blog/articles/da1eba90525f91)

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## サポート

問題が発生した場合は、GitHubのIssuesで報告してください。
