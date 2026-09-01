// SPDX-License-Identifier: GPL-3.0-only
//! Part-of-speech tagging for the English G2P.
//!
//! misaki chooses heteronym pronunciations by POS tag ("will record" vs "a
//! record"), and its tags come from spaCy's `en_core_web_sm` — a neural
//! pipeline this backend is not going to ship. Instead, an averaged
//! perceptron (Honnibal's design, the one NLTK ships) is *trained on that
//! spaCy model's own output* by `tools/gen-en-tagger.py`, and this file runs
//! it. The feature strings here must match the trainer's `get_features`
//! byte for byte — they are the keys into the weight table.
//!
//! A perceptron is not a neural network and will not reproduce spaCy tag
//! for tag; it does not have to. A tag only matters where the lexicon keys
//! on it, and the end-to-end golden test in `src/en.rs` measures exactly
//! that.

use std::collections::HashMap;
use std::sync::OnceLock;

const TAGGER_GZ: &[u8] = include_bytes!("../data/en-tagger.json.gz");

pub struct Tagger {
    classes: Vec<String>,
    weights: HashMap<String, Vec<(u16, f64)>>,
}

/// The trainer's `normalize`, verbatim.
fn normalize(word: &str) -> String {
    let mut chars = word.chars();
    let first = chars.next();
    if word.contains('-') && first != Some('-') {
        "!HYPHEN".to_string()
    } else if word.chars().all(|c| c.is_ascii_digit()) && word.chars().count() == 4 {
        "!YEAR".to_string()
    } else if first.is_some_and(|c| c.is_ascii_digit()) {
        "!DIGITS".to_string()
    } else {
        word.to_lowercase()
    }
}

/// Python's `word[-3:]`: the last three characters.
fn suffix3(word: &str) -> &str {
    let n = word.chars().count();
    if n <= 3 {
        word
    } else {
        let cut = word
            .char_indices()
            .nth(n - 3)
            .expect("n > 3 so index exists")
            .0;
        &word[cut..]
    }
}

impl Tagger {
    fn load() -> Self {
        use std::io::Read;
        let mut json = String::new();
        flate2::read::GzDecoder::new(TAGGER_GZ)
            .read_to_string(&mut json)
            .expect("embedded tagger decompresses");
        let raw: serde_json::Value = serde_json::from_str(&json).expect("tagger data parses");
        let classes = raw["classes"]
            .as_array()
            .expect("classes is an array")
            .iter()
            .map(|c| c.as_str().expect("class is a string").to_string())
            .collect();
        let weights = raw["weights"]
            .as_object()
            .expect("weights is an object")
            .iter()
            .map(|(f, rows)| {
                let rows = rows
                    .as_array()
                    .expect("weight rows")
                    .iter()
                    .map(|row| {
                        let pair = row.as_array().expect("weight pair");
                        (
                            u16::try_from(pair[0].as_u64().expect("class index"))
                                .expect("few classes"),
                            pair[1].as_f64().expect("weight"),
                        )
                    })
                    .collect();
                (f.clone(), rows)
            })
            .collect();
        Self { classes, weights }
    }

    pub fn shared() -> &'static Self {
        static SHARED: OnceLock<Tagger> = OnceLock::new();
        SHARED.get_or_init(Tagger::load)
    }

    fn predict(&self, features: &[String]) -> &str {
        let mut scores = vec![0.0f64; self.classes.len()];
        for f in features {
            if let Some(rows) = self.weights.get(f.as_str()) {
                for (idx, w) in rows {
                    scores[usize::from(*idx)] += w;
                }
            }
        }
        // Python: max(classes, key=lambda label: (scores[label], label)) —
        // highest score, ties to the lexicographically greatest label.
        // `classes` is sorted, so a later index is a greater label and `>=`
        // implements the tie-break.
        let mut best = 0usize;
        for (i, score) in scores.iter().enumerate().skip(1) {
            if *score >= scores[best] {
                best = i;
            }
        }
        &self.classes[best]
    }

    /// Tag a token sequence. Greedy left to right, exactly like training.
    pub fn tag(&self, words: &[String]) -> Vec<String> {
        let mut context: Vec<String> = Vec::with_capacity(words.len() + 4);
        context.push("-START2-".to_string());
        context.push("-START-".to_string());
        context.extend(words.iter().map(|w| normalize(w)));
        context.push("-END-".to_string());
        context.push("-END2-".to_string());

        let mut tags = Vec::with_capacity(words.len());
        let (mut prev, mut prev2) = ("-START-".to_string(), "-START2-".to_string());
        for (i, word) in words.iter().enumerate() {
            let c = |k: usize| context[k].as_str();
            let i2 = i + 2;
            let first: String = word.chars().next().map(String::from).unwrap_or_default();
            let features = [
                "bias".to_string(),
                format!("i suffix {}", suffix3(word)),
                format!("i pref1 {first}"),
                format!("i-1 tag {prev}"),
                format!("i-2 tag {prev2}"),
                format!("i tag+i-2 tag {prev} {prev2}"),
                format!("i word {}", c(i2)),
                format!("i-1 tag+i word {prev} {}", c(i2)),
                format!("i-1 word {}", c(i2 - 1)),
                format!("i-1 suffix {}", suffix3(c(i2 - 1))),
                format!("i-2 word {}", c(i2 - 2)),
                format!("i+1 word {}", c(i2 + 1)),
                format!("i+1 suffix {}", suffix3(c(i2 + 1))),
                format!("i+2 word {}", c(i2 + 2)),
            ];
            let guess = self.predict(&features).to_string();
            prev2 = std::mem::replace(&mut prev, guess.clone());
            tags.push(guess);
        }
        tags
    }
}
