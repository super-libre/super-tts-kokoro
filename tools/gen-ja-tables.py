#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Generate the Japanese pronunciation tables in `data/`.

As with Mandarin, the target is not "correct Japanese IPA" in the abstract but
*what misaki emits*, because that is what Kokoro was trained on.

Which misaki, though, is the part worth writing down. `misaki.ja.JAG2P` has two
paths and defaults to **cutlet** (fugashi + UniDic), not to `pyopenjtalk`. They
disagree, and only cutlet is usable here:

    こんにちは、世界。
      cutlet      koɲɲiʨiβa, sekai.
      pyopenjtalk koɴniʨiwa, sekai.__-------jj^^___j

The pyopenjtalk branch appends its pitch string to the phonemes and emits `ᶄ`,
which has no token in Kokoro's vocabulary. So this file follows cutlet.

  ja-kana.tsv.gz      ~190  kana or digraph -> IPA, plus punctuation
  ja-kata-ext.tsv.gz    16  small katakana -> its full-size form
  ja-words.tsv.gz  147,570  words cutlet re-joins tokens into
  ja-golden.tsv               misaki's own answers, replayed as a test

The two katakana tables are not interchangeable and are kept apart on purpose:
`ja-kana` maps a kana to *IPA*, while `ja-kata-ext` rewrites the Katakana
Phonetic Extensions block (ㇰ -> ク) during normalisation, before anything is
pronounced. Merging them would put a kana on the right-hand side of a table
whose values are supposed to be phonemes.

`ja-words.txt` is not decoration: cutlet re-groups the tokenizer's output by
looking for the longest run whose concatenated surface is a known word, which
is what keeps 日本 + 語 from being pronounced as two words.

Regenerating (only when upstream pronunciation data changes):

    uv venv --python 3.12 && uv pip install \
        fugashi==1.5.1 unidic-lite==1.0.8 jaconv==0.4.0 mojimoji==0.0.13 \
        pypinyin==0.55.0 jieba==0.42.1 cn2an==0.5.24 ordered-set==4.1.0 \
        addict==2.4.0
    git clone https://github.com/hexgrad/misaki /tmp/misaki   # fba1236
    PYTHONPATH=/tmp/misaki .venv/bin/python tools/gen-ja-tables.py

Pin those versions. A different UniDic is a different pronunciation dictionary,
and the point of these tables is that they match one known g2p.
"""

import gzip
import pathlib
import sys

try:
    from misaki.cutlet import HEPBURN, JA_WORDS, Katakana_Phonetic_Extensions, Cutlet
except ImportError:
    sys.exit("misaki not importable; see the regeneration steps in this file's docstring")

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def write(name, rows):
    body = "\n".join(f"{k}\t{v}" for k, v in sorted(rows.items())).encode("utf-8")
    path = DATA / name
    path.write_bytes(gzip.compress(body, 9))
    print(f"  {name:18} {len(rows):>7} entries  {len(body) / 1024:>7.0f} KB raw"
          f"  {path.stat().st_size / 1024:>6.0f} KB gz")


# Sentences chosen for what is easy to get wrong: the sokuon (っ -> ʔ) and its
# spacing, the moraic nasal ん whose realisation depends on what follows
# (m/ŋ/ɲ/n/ɴ), long vowels, digraphs, katakana loanwords, and every punctuation
# mark that has to survive into the token stream.
SENTENCES = [
    "こんにちは、世界。", "今日はいい天気ですね。", "音楽を聴くのが好きです。",
    "日本語の音声合成のテストです。", "客観的な評価が必要だ。", "銀行で働いています。",
    "東京大学の学生です。", "ありがとうございます！", "百科事典を読みました。",
    "写真を撮りましょう。", "彼は新しい車を買った。", "明日の会議は十時からです。",
    "この本はとても面白かった。", "駅までどのくらいかかりますか。", "少々お待ちください。",
    "カタカナのテスト", "コンピューターを使う", "アメリカとイギリス",
    "電車が遅れています", "すみません、道に迷いました", "日本の四季は美しい",
    "水を一杯ください", "頑張ってください！", "お誕生日おめでとう",
    "東京スカイツリーに行った", "図書館で勉強します", "ラーメンとチャーハン",
    "しっかり掴んでください", "はっきり言ってほしい", "切符を三枚買った",
    "新聞を読んでいます", "参観日に行きます", "散歩に行こう",
    "先生がいらっしゃった", "会社に電話をかける", "「こんにちは」と言った",
    "彼女はピアノを弾く", "ゆっくり歩きましょう", "きょうはあついですね",
    "そうですか？", "本当に？！", "一、二、三。",
    # Numbers, which misaki spells out before tokenizing. Left as digits they
    # reach the phoneme table, find no token, and vanish.
    "価格は99円です。", "2024年3月15日。", "3個あります。",
    "1000円をください。", "10000人が参加した。", "電話は0120です。",
    "第1回の会議。", "彼は21歳です。", "気温は35度でした。",
    "300人と800人。", "1024と2048。",
]


def main():
    DATA.mkdir(exist_ok=True)
    print("generating Japanese tables:")

    # Drop anything that cannot round-trip through the tab-separated format.
    kana = {k: v for k, v in HEPBURN.items() if "\t" not in k and "\n" not in k}
    write("ja-kana.tsv.gz", kana)
    write("ja-kata-ext.tsv.gz", dict(Katakana_Phonetic_Extensions))

    write("ja-words.tsv.gz", {w: "" for w in JA_WORDS if w and "\t" not in w})

    # `JAG2P()` delegates to `Cutlet` for every input; using it directly keeps
    # `pyopenjtalk`, which `misaki.ja` imports but this path never calls, out of
    # the requirements.
    g = Cutlet()
    # Sentences plus a deterministic sample of the word list, so the fixture
    # covers ordinary prose and a long tail of individual readings.
    corpus = list(SENTENCES)
    words = sorted(w for w in JA_WORDS if w)
    corpus += words[::210]

    out = []
    seen = set()
    for text in corpus:
        if text in seen or "\t" in text:
            continue
        seen.add(text)
        ipa = g(text)[0]
        out.append(f"{text}\t{ipa}")
    path = DATA / "ja-golden.tsv"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  {'ja-golden.tsv':18} {len(out):>7} cases   {path.stat().st_size / 1024:>7.0f} KB")


if __name__ == "__main__":
    main()
