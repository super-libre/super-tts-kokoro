#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Generate the Mandarin pronunciation tables in `data/`.

Kokoro was trained on phonemes from `misaki`, its reference g2p, so the target
here is not "correct Mandarin IPA" in the abstract — it is *what misaki emits*.
Anything else is a confident mispronunciation the model was never taught.

Rather than port misaki's Python at runtime, its behaviour is compiled down to
three lookup tables. That is possible because the pieces are finite:

  zh-chars.tsv.gz    41,923  hanzi           -> tone-3 pinyin syllable
  zh-phrases.tsv.gz  47,111  multi-char word -> syllables (space separated)
  zh-ipa.tsv.gz        1,492  syllable        -> misaki IPA

It also writes `zh-golden.tsv`, misaki's own answers for a corpus covering
natural sentences and the most frequent words in jieba's dictionary. That file
is the regression test: `matches_the_reference_g2p` replays it and asserts this
backend agrees with misaki line for line.

The phrase table is what makes pronunciation contextual rather than
per-character. Measured over jieba's dictionary weighted by word frequency,
character-by-character lookup misreads 5.7% of running text, and the words it
misreads are the commonest in the language — 一个 is yi2ge4, not yi1ge4.

Every phrase is carried, including those whose reading happens to equal the
concatenation of their characters'. Dropping those looks like a free 80%
saving and is not: pypinyin resolves a word by longest match against the whole
phrase dictionary, so a longer entry has to be able to outrank a shorter one.
With the "redundant" entries removed, 好好学习 stops matching as a unit and the
shorter 好好 wins instead, turning hao3hao3 into hao3hao4.

The syllable table is what removes the need to reimplement pypinyin's strict
initial/final segmentation, which is the fiddly part of pinyin-to-IPA. Mandarin
has ~1,500 syllables once tones are counted, so the whole mapping fits in 7 KB.

Regenerating (only needed when upstream pronunciation data changes):

    python3 -m venv .venv && .venv/bin/pip install \
        pypinyin==0.55.0 jieba==0.42.1 cn2an==0.5.24 ordered-set==4.1.0
    git clone https://github.com/hexgrad/misaki /tmp/misaki   # fba1236
    PYTHONPATH=/tmp/misaki .venv/bin/python tools/gen-zh-tables.py

Pin those versions. A different pypinyin is a different pronunciation
dictionary, and the point of these tables is that they match one known g2p.
"""

import gzip
import pathlib
import sys

from pypinyin import Style, lazy_pinyin
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT

try:
    from misaki.zh import ZHG2P
except ImportError:
    sys.exit("misaki not importable; see the regeneration steps in this file's docstring")

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

# The exact call misaki makes. `neutral_tone_with_five` is what makes the
# neutral tone an explicit `5` rather than a bare syllable, so every syllable
# in the table carries a tone digit and the Rust side never has to guess.
def say(text):
    return lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)


def write(name, rows):
    body = "\n".join(f"{k}\t{v}" for k, v in sorted(rows.items())).encode("utf-8")
    path = DATA / name
    path.write_bytes(gzip.compress(body, 9))
    print(f"  {name:20} {len(rows):>6} entries  {len(body) / 1024:>6.0f} KB raw"
          f"  {path.stat().st_size / 1024:>5.0f} KB gz")


def main():
    DATA.mkdir(exist_ok=True)
    print("generating Mandarin tables:")

    chars = {}
    for codepoint in PINYIN_DICT:
        ch = chr(codepoint)
        reading = say(ch)
        if reading:
            chars[ch] = reading[0]
    write("zh-chars.tsv.gz", chars)

    phrases = {}
    for phrase in PHRASES_DICT:
        if "\t" in phrase or "\n" in phrase:
            continue
        phrases[phrase] = " ".join(say(phrase))
    write("zh-phrases.tsv.gz", phrases)

    syllables = set(chars.values())
    for value in phrases.values():
        syllables.update(value.split())
    ipa = {s: ZHG2P.py2ipa(s) for s in sorted(syllables)}
    write("zh-ipa.tsv.gz", ipa)

    write_golden(phrases)


# Sentences chosen for the things that are easy to get wrong: tone sandhi on
# 一/不, heteronyms that only a phrase dictionary settles (银行, 行, 重), and
# every punctuation mark that has to survive into the token stream.
SENTENCES = [
    "你好，世界。", "今天天气很好。", "我喜欢听音乐。", "中文语音合成测试。",
    "他在银行工作。", "这个人很行。", "北京大学的学生。", "请问洗手间在哪里？",
    "谢谢你的帮助！", "一个人不是两个人。", "我认为这个要求不会太难。",
    "长江是中国最长的河流。", "他重新称了一下重量。", "音乐会明天晚上开始。",
    "这本书的部分内容很有意思。", "他的头发长得很长。", "我们应该好好学习。",
    "红色的花儿开了。", "他还了我还欠的钱。", "空调的空气很干燥。",
    "他说：「明天见。」", "真的？！", "《红楼梦》是一部小说。",
    "这是（重要的）事情。", "一二三四五六七八九十。",
    # Numbers, which cn2an rewrites before anything is segmented. Left as
    # digits they reach the phoneme table, find no token, and vanish.
    "价格是99元。", "2024年3月15日。", "我有3个苹果。", "第1名。",
    "折扣是50%。", "气温是35℃。", "1/4的人同意。", "5-10个人。",
    "重量是2.5公斤。", "电话是13800138000。", "会议在10点开始。",
]


def write_golden(phrases):
    """misaki's answers for a corpus this backend must reproduce exactly."""
    import pkgutil

    corpus = list(SENTENCES)

    # The most frequent multi-character words in jieba's dictionary, which is
    # where the phrase table earns its place.
    rows = []
    for line in pkgutil.get_data("jieba", "dict.txt").decode("utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) > 1 and all("\u4e00" <= c <= "\u9fff" for c in parts[0]):
            rows.append((parts[0], int(parts[1])))
    rows.sort(key=lambda r: -r[1])
    corpus += [w for w, _ in rows[:400]]

    # Phrase-table entries, sampled deterministically for breadth.
    corpus += sorted(phrases)[::2000]

    g = ZHG2P()
    seen, out = set(), []
    for text in corpus:
        if text in seen:
            continue
        seen.add(text)
        ipa = g(text)[0]
        # Tabs and newlines would break the fixture format; none occur here.
        out.append(f"{text}\t{ipa}")

    path = DATA / "zh-golden.tsv"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"  {'zh-golden.tsv':20} {len(out):>6} cases   {path.stat().st_size / 1024:>6.0f} KB")


if __name__ == "__main__":
    main()
