#!/usr/bin/env python3
"""
会話形式の音声生成テスト
"""

import re
from typing import List, Dict


def parse_dialogue_script(script_text: str) -> List[Dict]:
    """
    会話形式の台本を解析してA:とB:の発言を抽出
    （VoiceGenerator._parse_dialogue_scriptのテスト用コピー）
    """
    segments = []
    lines = script_text.split("\n")

    current_speaker = None
    current_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 発言者の識別（A: またはB: で始まる行）
        speaker_match = re.match(r'^([AB])[:：]\s*(.*)$', line)

        if speaker_match:
            # 前の発言を保存
            if current_speaker and current_text:
                segments.append({
                    "speaker": current_speaker,
                    "text": " ".join(current_text).strip()
                })

            # 新しい発言を開始
            current_speaker = speaker_match.group(1)
            current_text = [speaker_match.group(2)]
        else:
            # 継続行（発言者の指定がない行は前の発言者の続き）
            if current_speaker:
                current_text.append(line)

    # 最後の発言を保存
    if current_speaker and current_text:
        segments.append({
            "speaker": current_speaker,
            "text": " ".join(current_text).strip()
        })

    return segments


def test_parse_dialogue_script():
    """台本解析のテスト"""

    # サンプル台本
    sample_script = """
A: はい、今夜もやってきましたー
B: どうもー。今日も一杯やりながらですね
A: 今日のニュース、なんかすごいことになってますよ
B: まあ、簡単に言うと〜ってことですね
A: へー、そうなんですね
そういえば、前にも似たようなことありましたよね
B: ええ、確かにありました。
あの時は〜という状況でした
"""

    # 台本を解析
    segments = parse_dialogue_script(sample_script)

    print("\n=== 解析結果 ===")
    print(f"総セグメント数: {len(segments)}")
    print(f"A の発言数: {sum(1 for s in segments if s['speaker'] == 'A')}")
    print(f"B の発言数: {sum(1 for s in segments if s['speaker'] == 'B')}")

    print("\n=== 各セグメント ===")
    for i, segment in enumerate(segments):
        speaker = segment["speaker"]
        text = segment["text"]
        print(f"{i+1}. {speaker}: {text[:50]}{'...' if len(text) > 50 else ''}")

    # 検証
    assert len(segments) > 0, "セグメントが抽出されていません"
    assert any(s["speaker"] == "A" for s in segments), "Aの発言が見つかりません"
    assert any(s["speaker"] == "B" for s in segments), "Bの発言が見つかりません"

    print("\n✓ テスト成功！台本の解析が正しく動作しています")


if __name__ == "__main__":
    test_parse_dialogue_script()
