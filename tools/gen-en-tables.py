#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Generate the English pronunciation data in `data/`.

Kokoro's English voices were trained on phonemes from `misaki.en`, a
POS-aware, lexicon-first G2P — espeak is only its fallback for words the
lexicon does not know. So English parity means reproducing that pipeline, not
raw espeak: heteronyms picked by part of speech (read/record/live), numbers,
years, ordinals and currency verbalized to words, punctuation preserved for
prosody, and misaki's single-codepoint vowel symbols (A I O Q W Y, the flap T)
rather than espeak's multi-character diphthongs.

Four files are copied *verbatim* from the pinned misaki wheel and gzipped —
no transformation, so there is nothing to get wrong:

  en-gold-us.json.gz     90,201 words -> phonemes (US; 790 keyed by POS)
  en-silver-us.json.gz   93,361 lower-confidence US entries
  en-gold-gb.json.gz     the British twin of gold
  en-silver-gb.json.gz   the British twin of silver

It also writes `en-golden.tsv`: misaki's own answers, per dialect, for a
corpus that leans into everything espeak gets wrong — heteronym pairs in
disambiguating contexts, a systematic number/year/ordinal/currency sweep,
ordinary sentences, punctuation, a deterministic sample of the lexicon
itself, and out-of-vocabulary words that exercise the espeak fallback. The
Rust test `matches_the_reference_g2p` replays it line for line.

The oracle is pinned to what Kokoro itself pins: the `kokoro` package
requires `misaki[en]>=0.9.4`, and constructs `en.G2P(trf=False,
british=..., fallback=espeak.EspeakFallback(british=...), unk='')` — this
script constructs it identically.

Regenerating (only needed if the pinned reference ever changes):

    # SHORT venv path, deliberately: espeak-ng holds its data path in a
    # 160-byte buffer and silently truncates longer ones (see build.rs).
    uv venv --python 3.12 /tmp/mv
    VIRTUAL_ENV=/tmp/mv uv pip install "misaki[en]==0.9.4" \
        "en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    /tmp/mv/bin/python tools/gen-en-tables.py

Pin those versions. The spaCy model *is* the POS tagger; a different model
tags differently, and a different tag is a different pronunciation.
"""

import gzip
import json
import pathlib
import sys

try:
    from misaki import en, espeak
except ImportError:
    sys.exit("misaki not importable; see the regeneration steps in this file's docstring")

import misaki

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# The corpus. Grouped so a failure names the behaviour that regressed.

HETERONYMS = [
    "I read the book yesterday, and now she reads it too.",
    "He has read every record the record label released.",
    "They will record a new record next week.",
    "Live wires are dangerous where those musicians live.",
    "The wind was too strong to wind the sail.",
    "She shed a tear over the tear in her dress.",
    "The lead singer returned the lead pipe.",
    "He took a bow after tying the bow.",
    "Please close the door; the store is close by.",
    "The dove dove into the bushes.",
    "I object to that object.",
    "The suspect refused to suspect anyone.",
    "They produce fresh produce on the farm.",
    "The contract says we contract the work out.",
    "In the desert, soldiers rarely desert their posts.",
    "The content creator was content with the result.",
    "Use the tool the way it was meant to be used.",
    "A minute is a minute amount of time.",
    "The bass player caught a bass.",
    "Did you project the project timeline?",
    "The invalid entry came from an invalid sensor.",
    "They were too polite to subject the subject to more questions.",
    "Present the present to the winner.",
    "Progress reports show we progress slowly.",
    "The rebels rebel against the convict they convict.",
    "Perfect practice makes it perfect.",
    "The graduate will graduate in June.",
    "It is separate; please separate the pages.",
    "An expert can estimate the estimate quickly.",
    "The delegate will delegate the task.",
    "I refuse to take out the refuse.",
    "The intern will intern at the console and console the patient.",
    "Exports increase while imports decrease.",
    "The increase in exports did not offset the decrease in imports.",
    "Do not insult me with that insult.",
    "The addict became addicted to the digest he would digest.",
    "Attribute the quote; the attribute matters.",
    "The compound contains a compound problem.",
    "The conflict began when their stories conflict.",
    "The contest winner would contest the ruling.",
]

ORDINARY = [
    "Please call me back tomorrow morning.",
    "The train leaves at seven fifteen.",
    "I bought three apples and a loaf of bread.",
    "She works as a doctor in the city hospital.",
    "Turn left at the next traffic light.",
    "It has been raining since Tuesday.",
    "Could you send the report by Friday?",
    "The children are playing in the garden.",
    "This is the best book I have read this year.",
    "The kettle is boiling; tea will be ready soon.",
    "A gentle breeze moved through the tall grass.",
    "He answered every question without hesitation.",
    "Their flight lands early in the afternoon.",
    "We watched the storm roll in from the porch.",
    "The library closes at nine on weekdays.",
    "Someone left an umbrella by the front door.",
    "The recipe calls for two cups of flour.",
    "Every window in the old house was open.",
    "The orchestra tuned their instruments quietly.",
    "A stray cat followed us halfway home.",
]

OOV = [
    "Download the eBook onto your Kindle.",
    "Kubernetes clusters restarted after the rollout.",
    "She posted it on TikTok and Instagram.",
    "NASA and the FBI use USB drives.",
    "The flumbostic quandrizzle vorbled quietly.",
    "OpenAI and DeepMind published new papers.",
    "My MacBook crashed during the demo.",
    "The zorbling frimbulator needs recalibrating.",
]

PUNCT = [
    "Don't worry — it's only a test.",
    '"Well," she said, "that\'s that."',
    "It's ten o'clock; we're already late.",
    "The well-known author co-wrote a best-seller.",
    "Wait... what happened?",
    "Yes! No? Maybe, then.",
    "He said (quite loudly) that we'd won.",
]


def number_snippets():
    """A systematic sweep of everything `get_number` verbalizes."""
    out = []
    out += [str(n) for n in range(0, 21)]
    out += [str(n) for n in range(21, 130, 7)]
    out += ["1000", "1234", "10000", "100000", "1000000", "2500000"]
    out += ["1066", "1492", "1776", "1900", "1901", "1984", "1999",
            "2000", "2001", "2010", "2024", "2038"]
    out += ["1st", "2nd", "3rd", "4th", "5th", "11th", "12th", "13th",
            "21st", "22nd", "23rd", "30th", "31st", "100th"]
    out += ["$1", "$5.50", "$100", "$1,250.99", "£3.50", "£20", "€99"]
    out += ["3.14", "0.5", "12.75"]
    out += ["5%", "15%", "100%"]
    out += ["3:30", "12:45", "9:05"]
    sentences = [
        "The meeting is at 3:30 pm on May 5th.",
        "It costs $1,250.99 in total.",
        "That's a 15% increase over 2024.",
        "She was born in 1984 and moved here in 2001.",
        "Pi is roughly 3.14, not 3.",
        "He finished 1st, she finished 22nd, and I came 3rd.",
        "The invoice was £3.50 plus €20 shipping.",
        "Chapter 7 begins on page 101.",
        "In 1900 the town had 1,000,000 residents.",
        "The temperature dropped to -5 degrees.",
        "A dozen is 12 and a gross is 144.",
        "Flight 370 departs at 12:45 from gate 22.",
    ]
    return out + sentences


def lexicon_sample():
    """Every Nth word of the pinned lexicons, deterministically."""
    import json
    pkg = pathlib.Path(misaki.__file__).parent / "data"
    gold = json.loads((pkg / "us_gold.json").read_text())
    silver = json.loads((pkg / "us_silver.json").read_text())
    words = sorted(gold)[::200] + sorted(silver)[::400]
    return [w for w in words if w.isalpha()]


def main():
    import shutil

    pkg = pathlib.Path(misaki.__file__).parent / "data"
    for src, dest in [
        ("us_gold.json", "en-gold-us.json.gz"),
        ("us_silver.json", "en-silver-us.json.gz"),
        ("gb_gold.json", "en-gold-gb.json.gz"),
        ("gb_silver.json", "en-silver-gb.json.gz"),
    ]:
        raw = (pkg / src).read_bytes()
        with gzip.open(DATA / dest, "wb", compresslevel=9) as f:
            f.write(raw)
        print(f"{dest}: {len(raw):,} bytes raw")

    corpus = (
        [("heteronym", s) for s in HETERONYMS]
        + [("numbers", s) for s in number_snippets()]
        + [("ordinary", s) for s in ORDINARY]
        + [("oov", s) for s in OOV]
        + [("punct", s) for s in PUNCT]
        + [("lexicon", w) for w in lexicon_sample()]
    )

    rows = []
    token_rows = {}
    for lang, british in (("en-us", False), ("en-gb", True)):
        fallback = espeak.EspeakFallback(british=british)
        g2p = en.G2P(trf=False, british=british, fallback=fallback, unk="")
        for group, text in corpus:
            ps, _ = g2p(text)
            assert "\t" not in text and "\t" not in ps
            rows.append(f"{lang}\t{group}\t{text}\t{ps}\n")
            if text not in token_rows:
                doc = g2p.nlp(text)
                token_rows[text] = json.dumps(
                    [[t.text, t.tag_, t.whitespace_] for t in doc],
                    ensure_ascii=False,
                )
    (DATA / "en-golden.tsv").write_text("".join(rows), encoding="utf-8")
    print(f"en-golden.tsv: {len(rows)} rows")

    # spaCy's tokenization + tags for every corpus line, so the Rust port of
    # the post-tagger pipeline (lexicon, numbers, stress, merging) can be
    # tested in isolation from the tokenizer/tagger port. Dialect-independent:
    # both dialects load the same en_core_web_sm.
    with open(DATA / "en-tokens.tsv", "w", encoding="utf-8") as f:
        for group, text in corpus:
            f.write(f"{text}\t{token_rows[text]}\n")
    print(f"en-tokens.tsv: {len(corpus)} rows")

    # num2words ground truth: `get_number` builds on num2words' exact wording
    # ("one thousand, two hundred and fifty", year "nineteen eighty-four"),
    # so the Rust port of it is verified against this sweep verbatim.
    from num2words import num2words
    ns = (
        list(range(0, 1001))
        + [1066, 1492, 1776, 1900, 1901, 1984, 1999, 2000, 2001, 2010, 2024, 2038]
        + [1234, 2500, 9999, 10000, 12345, 100000, 123456, 1000000,
           2500000, 999999999, 1000000000, 1234567890]
    )
    with open(DATA / "en-numwords.tsv", "w", encoding="utf-8") as f:
        for n in ns:
            f.write(f"{n}\t{num2words(n)}\t{num2words(n, to='ordinal')}\t{num2words(n, to='year')}\n")
    print(f"en-numwords.tsv: {len(ns)} rows")

    # The complete definition of spaCy's English tokenizer, dumped from the
    # pinned model: the 1,347 special-case rules (ORTH pieces only — tags
    # come from the tagger) and the effective prefix/suffix/infix/url
    # regexes. The Rust port loads this and reimplements the algorithm of
    # spacy/tokenizer.pyx; the en-tokens.tsv fixture is its judge.
    import spacy.attrs

    g2p = en.G2P(trf=False, british=False, fallback=None, unk="")
    tok = g2p.nlp.tokenizer
    orth = spacy.attrs.ORTH
    dump = {
        "rules": {
            text: [piece[orth] for piece in pieces]
            for text, pieces in tok.rules.items()
        },
        "prefix": tok.prefix_search.__self__.pattern,
        "suffix": tok.suffix_search.__self__.pattern,
        "infix": tok.infix_finditer.__self__.pattern,
        "url": tok.url_match.__self__.pattern,
    }
    with gzip.open(DATA / "en-tokenizer.json.gz", "wb", compresslevel=9) as f:
        f.write(json.dumps(dump, ensure_ascii=False).encode("utf-8"))
    print(f"en-tokenizer.json.gz: {len(dump['rules'])} rules")


if __name__ == "__main__":
    main()
