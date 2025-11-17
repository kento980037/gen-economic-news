#!/usr/bin/env python3
"""
マルチターン会話生成のテスト
"""

import sys
import yaml
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_config_loading():
    """設定ファイルの読み込みテスト"""

    # 設定ファイルを読み込み
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    script_config = config.get("script", {})

    print("\n=== Script Configuration ===")
    print(f"Tone: {script_config.get('tone')}")
    print(f"Style: {script_config.get('style')}")

    # マルチターン設定の確認
    multiturn_config = script_config.get("multiturn_dialogue", {})
    print(f"\n=== Multi-turn Configuration ===")
    print(f"Enabled: {multiturn_config.get('enabled')}")
    print(f"Turns per section: {multiturn_config.get('turns_per_section')}")

    # キャラクタープロンプトの確認
    speaker_a_prompt = script_config.get("system_prompt_speaker_a", "")
    speaker_b_prompt = script_config.get("system_prompt_speaker_b", "")

    print(f"\n=== Speaker Prompts ===")
    print(f"Speaker A prompt length: {len(speaker_a_prompt)} chars")
    print(f"Speaker B prompt length: {len(speaker_b_prompt)} chars")

    if speaker_a_prompt:
        print(f"\nSpeaker A preview: {speaker_a_prompt[:100]}...")

    if speaker_b_prompt:
        print(f"\nSpeaker B preview: {speaker_b_prompt[:100]}...")

    # 検証
    assert multiturn_config.get('enabled') == True, "Multi-turn should be enabled"
    assert speaker_a_prompt, "Speaker A prompt should exist"
    assert speaker_b_prompt, "Speaker B prompt should exist"

    print("\n✓ テスト成功！マルチターン設定が正しく読み込まれています")


def test_script_generator_init():
    """ScriptGeneratorの初期化テスト"""

    # 設定ファイルを読み込み
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    script_config = config.get("script", {})

    # OpenAI API keyの確認
    import os
    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY が設定されていません")
        print("実際の台本生成には API key が必要です")
        return

    # ScriptGeneratorの初期化
    from script_generator import ScriptGenerator
    generator = ScriptGenerator(script_config)

    print("\n=== ScriptGenerator Initialization ===")
    print(f"Multi-turn enabled: {generator.multiturn_enabled}")
    print(f"Turns per section: {generator.turns_per_section}")
    print(f"Tone: {generator.tone}")
    print(f"Style: {generator.style}")
    print(f"Speaker A prompt loaded: {len(generator.system_prompt_speaker_a) > 0}")
    print(f"Speaker B prompt loaded: {len(generator.system_prompt_speaker_b) > 0}")

    # 検証
    assert generator.multiturn_enabled == True, "Multi-turn should be enabled"
    assert generator.tone == "chill", "Tone should be chill"
    assert generator.style == "dialogue", "Style should be dialogue"

    print("\n✓ テスト成功！ScriptGeneratorが正しく初期化されています")


if __name__ == "__main__":
    print("=" * 60)
    print("マルチターン会話生成のテスト")
    print("=" * 60)

    try:
        test_config_loading()
        print("\n" + "=" * 60)
        test_script_generator_init()
        print("\n" + "=" * 60)
        print("\n✓ 全てのテストが成功しました！")
        print("\n次のステップ:")
        print("- python src/main.py を実行して実際の台本を生成")
        print("- マルチターン会話で2人のキャラクターが交互に話します")

    except Exception as e:
        print(f"\n✗ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
