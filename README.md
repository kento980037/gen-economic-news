# Economic News Video Generator

An automated system that generates narrated video content from economic news using AI. The system fetches latest economic news, generates natural-sounding scripts with Claude AI, synthesizes voice with OpenAI TTS or Gemini TTS, and produces polished videos with MoviePy.

## Features

- **Automated News Collection**: Fetches latest economic news from RSS feeds and NewsAPI
- **AI-Powered Script Generation**: Creates natural Japanese narratives using Claude API
- **High-Quality Voice Synthesis**: Supports both OpenAI TTS and Gemini TTS for realistic voice-overs
- **Automated Video Production**: Generates videos with synchronized subtitles, thumbnails, and metadata
- **Flexible Deployment**: Run locally or via GitHub Actions
- **Notifications**: Optional Slack integration for workflow updates

## Technology Stack

- **AI & APIs**: Claude (Anthropic), OpenAI TTS, Gemini TTS, NewsAPI
- **Video Processing**: MoviePy, FFmpeg, Pillow
- **Audio Processing**: pydub, OpenAI Whisper (for transcription)
- **Deployment**: GitHub Actions
- **Notifications**: Slack SDK

## System Architecture

```
gen-economic-news/
├── src/
│   ├── main.py                  # Main entry point
│   ├── news_fetcher.py          # News aggregation orchestrator
│   ├── rss_news_fetcher.py      # RSS feed parser
│   ├── openai_news_fetcher.py   # NewsAPI integration
│   ├── news_scraper.py          # Web scraping utilities
│   ├── script_generator.py      # Script generation (Claude API)
│   ├── voice_generator.py       # Voice synthesis (OpenAI/Gemini TTS)
│   ├── video_generator.py       # Video composition (MoviePy)
│   ├── metadata_generator.py    # Metadata creation
│   └── utils.py                 # Utility functions
├── config/
│   └── config.yaml              # Configuration file
├── output/                      # Generated files
│   ├── videos/                  # Video files
│   ├── audio/                   # Audio files
│   ├── thumbnails/              # Thumbnail images
│   ├── scripts/                 # Scripts & metadata
│   └── logs/                    # Application logs
├── requirements.txt             # Python dependencies
└── .github/workflows/           # GitHub Actions workflows
```

## Prerequisites

- Python 3.11 or higher
- FFmpeg
- API keys for:
  - Anthropic Claude API (script generation)
  - OpenAI API (voice synthesis & transcription) **or** Gemini API (voice synthesis)
  - NewsAPI (optional)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kento980037/gen-economic-news.git
cd gen-economic-news
```

### 2. Set Up Python Environment

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install System Dependencies

```bash
# macOS (using Homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

Edit the `.env` file:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Voice synthesis (choose one or both)
OPENAI_API_KEY=your_openai_api_key_here      # For OpenAI TTS
GEMINI_API_KEY=your_gemini_api_key_here      # For Gemini TTS

# Optional
NEWS_API_KEY=your_newsapi_key_here
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
```

### 5. Customize Configuration

Edit `config/config.yaml` to customize:
- News sources (RSS URLs, NewsAPI settings)
- Script style and length
- Voice characteristics and speed
- Video resolution and design
- Metadata formatting

## Usage

### Local Execution

```bash
# Basic execution
cd src
python main.py

# Dry run (no actual generation)
python main.py --dry-run

# Test mode (with sample data)
python main.py --test

# Specify custom config file
python main.py --config ../config/config.yaml
```

### Testing Individual Modules

Each module can be tested independently:

```bash
cd src

# Test news fetching
python news_fetcher.py

# Test script generation
python script_generator.py

# Test voice synthesis
python voice_generator.py

# Test video generation
python video_generator.py

# Test metadata generation
python metadata_generator.py

# Test utilities
python utils.py
```

## Deployment with GitHub Actions

Use GitHub Actions for automated execution.

**1. Configure GitHub Secrets:**

In your repository: Settings > Secrets and variables > Actions

Add:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY` or `GEMINI_API_KEY`
- `NEWS_API_KEY` (optional)
- `SLACK_WEBHOOK_URL` (optional)

**2. Enable Workflow:**

The workflow `.github/workflows/generate-video.yml` can be triggered manually or on push to main branch.

**3. Manual Execution:**

You can also trigger manually:
1. Go to the "Actions" tab in your repository
2. Select "Generate Economic News Video"
3. Click "Run workflow"

## Output Files

Generated files structure:

```
output/
├── videos/
│   └── economic_news_20250102_090000.mp4    # Video file
├── audio/
│   └── voice_000.mp3                        # Audio file
├── thumbnails/
│   └── economic_news_20250102_090000.jpg    # Thumbnail
└── scripts/
    ├── economic_news_20250102_090000_script.json     # Script data
    ├── economic_news_20250102_090000_metadata.json   # Metadata
    └── economic_news_20250102_090000_summary.json    # Summary
```

## Customization

### Script Style

Configure in `config/config.yaml` under the `script` section:

```yaml
script:
  tone: "professional"  # professional, casual, friendly
  style: "narration"    # narration, dialogue
  min_length: 7500      # Minimum script length in characters
  max_length: 15000     # Maximum script length in characters
```

### Voice Settings

Configure in `config/config.yaml` under the `voice` section:

```yaml
voice:
  # Choose provider: "openai" or "gemini"
  provider: "gemini"  # Gemini produces more natural Japanese voices

  # OpenAI TTS settings
  openai:
    model: "tts-1-hd"  # tts-1 or tts-1-hd
    voice: "alloy"     # alloy, echo, fable, onyx, nova, shimmer
    speed: 1.0         # 0.25 to 4.0
    format: "mp3"

  # Gemini TTS settings (more natural Japanese)
  gemini:
    model: "gemini-2.5-flash-preview-tts"
    voice: "Kore"  # Choose from 30+ available voices
    format: "wav"
    style_prompt: "Calm, professional narration tone"
```

**Voice Preview:** Try all 30+ Gemini voices at [Google AI Studio](https://aistudio.google.com/generate-speech).

### Video Design

Configure in `config/config.yaml` under the `video` section:

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

## Troubleshooting

### FFmpeg Errors

```bash
# Verify FFmpeg installation
ffmpeg -version

# Check if FFmpeg is in PATH
which ffmpeg
```

### API Errors

- Verify API keys are correctly set in `.env`
- Check if you've exceeded API rate limits
- Review logs at `output/logs/app.log`

### Memory Issues

- Reduce video resolution
- Split audio generation into smaller segments
- Increase server memory allocation

### Japanese Font Not Found

If you encounter font errors during video generation:

```python
# Edit the font path in video_generator.py
font = ImageFont.truetype("/path/to/your/japanese/font.ttc", font_size)
```

## References

- [Claude API Documentation](https://docs.anthropic.com/)
- [OpenAI TTS Documentation](https://platform.openai.com/docs/guides/text-to-speech)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [MoviePy Documentation](https://zulko.github.io/moviepy/)

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Support

If you encounter any issues, please report them on [GitHub Issues](https://github.com/kento980037/gen-economic-news/issues).
