#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Train the English POS tagger in `data/en-tagger.json.gz`.

misaki picks heteronym pronunciations by part-of-speech tag, and the tags it
was built against are spaCy's (`en_core_web_sm`, a small neural pipeline).
Shipping that network is not an option for this backend, so the tagger here
is an averaged perceptron — Honnibal's classic 200-line design, the same one
NLTK ships — trained not on a treebank but *on spaCy's own output*, so that
it imitates the exact tagger misaki used, quirks and all. Where the
perceptron disagrees with spaCy it usually disagrees on tags that do not
change pronunciation; the end-to-end golden test in `src/en.rs` is the
number that matters.

Training text: public-domain Gutenberg prose (varied register, cached in
/tmp), plus a few thousand generated sentences dense in what this backend
actually speaks — times, currency, ordinals, years, and heteronym contexts.
Everything is tagged by the pinned spaCy and the perceptron learns to
reproduce it. Deterministic: fixed seed, fixed book list, fixed epochs.

Run inside the same venv as gen-en-tables.py (see its docstring), then:

    /tmp/mv/bin/python tools/gen-en-tagger.py
"""

import gzip
import json
import pathlib
import random
import re
import sys
import urllib.request
from collections import defaultdict

try:
    import spacy
except ImportError:
    sys.exit("spacy not importable; see the regeneration steps in this file's docstring")

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
CACHE = pathlib.Path("/tmp/en-tagger-corpus")
SEED = 20260901
EPOCHS = 8
# Weights below this are noise that costs megabytes; NLTK prunes similarly.
PRUNE = 0.5

BOOKS = {
    "1342": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",   # Pride and Prejudice
    "84": "https://www.gutenberg.org/cache/epub/84/pg84.txt",         # Frankenstein
    "2701": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",   # Moby Dick
    "1661": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",   # Sherlock Holmes
    "76": "https://www.gutenberg.org/cache/epub/76/pg76.txt",         # Huckleberry Finn
    "98": "https://www.gutenberg.org/cache/epub/98/pg98.txt",         # A Tale of Two Cities
    "158": "https://www.gutenberg.org/cache/epub/158/pg158.txt",      # Emma
    "35": "https://www.gutenberg.org/cache/epub/35/pg35.txt",         # The Time Machine
}


def fetch_books():
    CACHE.mkdir(exist_ok=True)
    texts = []
    for book_id, url in BOOKS.items():
        path = CACHE / f"{book_id}.txt"
        if not path.exists():
            print(f"fetching {url}")
            with urllib.request.urlopen(url) as r:
                path.write_bytes(r.read())
        raw = path.read_text(encoding="utf-8", errors="replace")
        # Strip the Gutenberg boilerplate.
        m = re.search(r"\*\*\* START OF.*?\*\*\*", raw, re.S)
        if m:
            raw = raw[m.end():]
        m = re.search(r"\*\*\* END OF", raw)
        if m:
            raw = raw[: m.start()]
        texts.append(raw)
    return texts


def domain_sentences(rng):
    """Sentences dense in the constructs this backend speaks."""
    out = []
    verbs = ["record", "present", "object", "produce", "contract", "permit",
             "project", "subject", "suspect", "refuse", "conduct", "insult",
             "progress", "protest", "convict", "increase", "decrease"]
    for v in verbs:
        out += [
            f"They will {v} the results tomorrow.",
            f"The {v} was reviewed by the board.",
            f"We {v} new items every week.",
            f"Each {v} takes time to review.",
            f"Did anyone {v} the change?",
            f"An old {v} sat in the archive.",
        ]
    out += [
        "I read the letter yesterday.", "She reads the paper daily.",
        "He has read all of them.", "Please read this aloud.",
        "They live near the lake.", "It was a live broadcast.",
        "The wind howled all night.", "Wind the clock before bed.",
        "She shed a tear.", "There is a tear in the fabric.",
    ]
    # Noun/verb frames beyond the "will X" / "the X" pair, so the decision
    # boundary is not carried by two function words.
    for v in verbs:
        out += [
            f"Sales {v} every quarter.",
            f"The {v} shocked everyone.",
            f"Critics {v} the decision loudly.",
            f"Their {v} was rejected.",
            f"We must {v} carefully.",
            f"A sudden {v} in demand followed.",
        ]
    out += [
        "She dove into the pool.", "A dove cooed on the ledge.",
        "What's the use of it?", "They use it daily.",
        "The waves roll in slowly.", "The tides roll in at dusk.",
        "Storms roll in from the coast.",
    ]
    # Straight-quote dialogue: spaCy's open/close-quote tags drive misaki's
    # curly-quote phonemes, so the tagger has to know them cold.
    speakers = ["she said", "he replied", "the guide answered", "Anna whispered",
                "the captain shouted", "her brother muttered"]
    lines = ["Well", "No", "Not yet", "Come along", "That's enough", "Why not",
             "Wait here", "It's late", "Look out", "Never mind"]
    for i, line in enumerate(lines):
        for j, speaker in enumerate(speakers):
            end = "?" if line.startswith("Why") else ","
            out.append(f'"{line}{end}" {speaker}.')
            if (i + j) % 2 == 0:
                out.append(f'{speaker.capitalize()}, "{line}."')
    for _ in range(600):
        h, m = rng.randrange(1, 13), rng.randrange(0, 60)
        d, c = rng.randrange(1, 500), rng.randrange(0, 100)
        y = rng.randrange(1500, 2100)
        n = rng.randrange(1, 32)
        cur = rng.choice(["$", "£", "€"])
        out += [
            f"The meeting is at {h}:{m:02d} on the {n}th.",
            f"It costs {cur}{d}.{c:02d} plus tax.",
            f"The fee is {cur}{n} per month.",
            f"She was born in {y}.",
            f"About {d}% of them agreed.",
            f"Chapter {n} starts on page {d}.",
        ]
    rng.shuffle(out)
    return out


def chunks_from(texts, rng):
    chunks = []
    for text in texts:
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if 30 < len(para) < 1200:
                chunks.append(para)
    rng.shuffle(chunks)
    return chunks


def normalize(word):
    if "-" in word and word and word[0] != "-":
        return "!HYPHEN"
    if word.isdigit() and len(word) == 4:
        return "!YEAR"
    if word and word[0].isdigit():
        return "!DIGITS"
    return word.lower()


def get_features(i, word, context, prev, prev2):
    """NLTK's PerceptronTagger feature templates, verbatim. The Rust
    inference in src/en_tagger.rs must build the identical strings."""
    i += 2  # context is padded with two START entries
    f = [
        "bias",
        "i suffix " + word[-3:],
        "i pref1 " + (word[0] if word else ""),
        "i-1 tag " + prev,
        "i-2 tag " + prev2,
        "i tag+i-2 tag " + prev + " " + prev2,
        "i word " + context[i],
        "i-1 tag+i word " + prev + " " + context[i],
        "i-1 word " + context[i - 1],
        "i-1 suffix " + context[i - 1][-3:],
        "i-2 word " + context[i - 2],
        "i+1 word " + context[i + 1],
        "i+1 suffix " + context[i + 1][-3:],
        "i+2 word " + context[i + 2],
    ]
    return f


class Perceptron:
    def __init__(self):
        self.weights = defaultdict(lambda: defaultdict(float))
        self.classes = set()
        self._totals = defaultdict(float)
        self._tstamps = defaultdict(int)
        self.i = 0

    def predict(self, features):
        scores = defaultdict(float)
        for f in features:
            if f in self.weights:
                for label, weight in self.weights[f].items():
                    scores[label] += weight
        return max(self.classes, key=lambda label: (scores[label], label))

    def update(self, truth, guess, features):
        self.i += 1
        if truth == guess:
            return
        for f in features:
            for label, delta in ((truth, 1.0), (guess, -1.0)):
                key = (f, label)
                self._totals[key] += (self.i - self._tstamps[key]) * self.weights[f][label]
                self._tstamps[key] = self.i
                self.weights[f][label] += delta

    def average(self):
        for f, ws in self.weights.items():
            for label in ws:
                key = (f, label)
                total = self._totals[key] + (self.i - self._tstamps[key]) * ws[label]
                ws[label] = total / self.i


def main():
    rng = random.Random(SEED)
    nlp = spacy.load("en_core_web_sm", enable=["tok2vec", "tagger"])

    chunks = chunks_from(fetch_books(), rng) + domain_sentences(rng)
    rng.shuffle(chunks)
    print(f"{len(chunks)} chunks; tagging with spaCy…")
    tagged = []
    for doc in nlp.pipe(chunks, batch_size=256):
        sent = [(t.text, t.tag_) for t in doc]
        if len(sent) > 2:
            tagged.append(sent)
    n_tokens = sum(len(s) for s in tagged)
    print(f"{len(tagged)} sequences, {n_tokens} tokens")

    holdout = tagged[: len(tagged) // 20]
    train = tagged[len(tagged) // 20:]

    model = Perceptron()
    model.classes = {tag for sent in tagged for _, tag in sent}
    for epoch in range(EPOCHS):
        correct = total = 0
        for sent in train:
            context = ["-START2-", "-START-"] + [normalize(w) for w, _ in sent] + ["-END-", "-END2-"]
            prev, prev2 = "-START-", "-START2-"
            for i, (word, truth) in enumerate(sent):
                feats = get_features(i, word, context, prev, prev2)
                guess = model.predict(feats)
                model.update(truth, guess, feats)
                prev2, prev = prev, guess
                correct += guess == truth
                total += 1
        rng.shuffle(train)
        print(f"epoch {epoch + 1}: {100 * correct / total:.2f}% train accuracy")
    model.average()

    def tag_accuracy(sents):
        correct = total = 0
        for sent in sents:
            context = ["-START2-", "-START-"] + [normalize(w) for w, _ in sent] + ["-END-", "-END2-"]
            prev, prev2 = "-START-", "-START2-"
            for i, (word, truth) in enumerate(sent):
                guess = model.predict(get_features(i, word, context, prev, prev2))
                prev2, prev = prev, guess
                correct += guess == truth
                total += 1
        return 100 * correct / total

    print(f"holdout vs spaCy: {tag_accuracy(holdout):.2f}%")

    classes = sorted(model.classes)
    index = {c: k for k, c in enumerate(classes)}
    packed = {}
    kept = dropped = 0
    for f, ws in model.weights.items():
        row = [[index[label], round(w, 3)] for label, w in sorted(ws.items()) if abs(w) >= PRUNE]
        if row:
            packed[f] = row
            kept += len(row)
        else:
            dropped += 1
    print(f"{len(packed)} features, {kept} weights kept, {dropped} features pruned")

    payload = json.dumps({"classes": classes, "weights": packed}, ensure_ascii=False)
    with gzip.open(DATA / "en-tagger.json.gz", "wb", compresslevel=9) as f:
        f.write(payload.encode("utf-8"))
    print(f"en-tagger.json.gz: {len(payload):,} bytes raw")


if __name__ == "__main__":
    main()
