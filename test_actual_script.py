#!/usr/bin/env python3
"""
実際の台本で解析をテスト
"""

import re
from typing import List, Dict


def parse_dialogue_script(script_text: str) -> List[Dict]:
    """
    会話形式の台本を解析してA:とB:の発言を抽出
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


# 実際の台本
actual_script = """A: 「はい、今夜もやってきましたー！深夜の経済ニュースポッドキャスト、始まりますよー！」

B: 「どうもー。今日も一杯やりながらニュースを語っていきましょうか。」

A: 「今日のニュース、ちょっとドキッとする内容なんですよ！アリババが中国軍をサポートしてるって話が出てるみたいで…」

B: 「具体的にはどんな内容なんですか？」

A: 「ホワイトハウスのメモによると、アリババが米国をターゲットにした中国軍の作戦を手伝ってるらしいんですよ！」

B: 「正確には、アリババが中国軍の"作戦"に技術支援をしているとされているということですね。ただし、信憑性は確認されていないようです。」

A: 「それって、アリババが中国軍のためにテクノロジーを提供してるってことですか？」

B: 「そうですね。アリババはこの報道を完全に否定していて、『これは虚偽だ』とコメントしています。」

A: 「でも、アリババの株価も影響受けるんじゃないですか？」

B: 「実際、報道を受けてアリババの株価はアメリカ市場で3.78%下落しました。市場は非常に敏感ですから。」

A: 「それにしても、中国のAI業界全体が影響を受けるってことですよね？」

B: 「はい、特にアリババはオープンソースのQwen AIモデルを開発中で、これがシリコンバレーでも注目されています。」

A: 「アリババのニュースが影響するかもしれないですね。」

B: 「そうですね。投資家たちが警戒感を強めているのは事実です。特に米中間のテクノロジー戦争が絡んでくると、リスクが高まりますから。」

A: 「なるほど。じゃあ、アリババの株価の動きに注目する必要がありそうですね。」

B: 「その通りです。株価の動向だけでなく、今後の規制強化や国際的な反応も見逃せません。」

A: 「あー、そういえば、過去にもHuaweiが同じような問題で騒がれてましたよね。」

B: 「そうですね。Huaweiは米国の国家安全保障に対する脅威と見なされ、厳しい制裁を受けました。アリババも同じ道を辿る可能性があります。」

A: 「それにしても、アリババの発表は気になりますね。11月25日には四半期の結果を発表するみたいですし。」

B: 「そうですね。このニュースの影響で、投資家がどのように反応するのか注目です。」

A: 「今後の動きにも注目ですね。ちなみに、他に関連するニュースはありますか？」

B: 「実は、最近の中国の輸出が減少したというニュースがあります。特に米国への輸出が前年同月比で25%減少しています。」

A: 「それって、アリババの影響もあるんじゃないですか？」

B: 「そうですね。アリババのニュースが影響を与える中、中国全体の経済にも影響を及ぼす可能性があります。」

A: 「なんか、経済の動きがますます複雑になってきますね。」

B: 「その通りです。投資家はリスクをしっかり管理する必要があります。」

A: 「それでは、今日のポイントを軽く振り返りましょうか。」

B: 「今日のポイントは以下の通りです。
1. アリババが中国軍に関与しているとの報道があり、株価に影響を与えた。
2. アリババは報道を否定し、企業の信頼性が問われている。
3. 中国の輸出が減少していることが、アリババの影響を受ける可能性がある。」

A: 「なるほどー、勉強になりました！」

B: 「それでは、また次回もお楽しみに。チャンネル登録も忘れずにお願いします。」

A: 「それでは、良い投資を！」

B: 「良い投資を〜！」"""

print("=== 台本解析テスト ===\n")
segments = parse_dialogue_script(actual_script)

print(f"総セグメント数: {len(segments)}")
print(f"A の発言数: {sum(1 for s in segments if s['speaker'] == 'A')}")
print(f"B の発言数: {sum(1 for s in segments if s['speaker'] == 'B')}")

print("\n=== 各セグメント ===")
for i, segment in enumerate(segments):
    speaker = segment["speaker"]
    text = segment["text"]
    print(f"{i+1}. {speaker}: {text[:80]}{'...' if len(text) > 80 else ''}")

print(f"\n=== 総文字数 ===")
total_chars = sum(len(s["text"]) for s in segments)
print(f"総文字数: {total_chars}")
print(f"推定時間: {total_chars // 5}秒")
