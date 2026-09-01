// SPDX-License-Identifier: GPL-3.0-only
//! Mandarin text to phonemes.
//!
//! `espeak-ng` cannot serve this language here: it writes Mandarin with numeric
//! tone marks (`ni2 hao3`) that have no token in Kokoro's vocabulary, so its
//! output would reach the model as a stream of dropped symbols. Kokoro's
//! reference g2p, `misaki`, writes the tones as the arrows `→ ↗ ↓ ↘` — tokens
//! 171, 172, 169 and 173 — which is what the model was trained on and what this
//! module reproduces.
//!
//! # Why tables and not a port
//!
//! `misaki` reaches its phonemes through `jieba` for segmentation, `pypinyin`
//! for pronunciation, and a pinyin-to-IPA stage that needs `pypinyin`'s strict
//! initial/final segmentation. Reimplementing that last part is the fiddly bit,
//! and it is also unnecessary: Mandarin has only ~1,500 syllables once tones are
//! counted, so the entire mapping is precomputed into a 7 KB table by
//! `tools/gen-zh-tables.py`. What is left here is lookup and assembly.
//!
//! # Why the phrase table exists
//!
//! Pronunciation is not per-character. 一个 is `yi2ge4`, not `yi1ge4`; 银行 is
//! `yin2hang2`, not `yin2xing2`. Measured over `jieba`'s dictionary weighted by
//! word frequency, character-by-character lookup misreads **5.7% of running
//! text**, and the words it misreads are the commonest in the language.

use std::collections::HashMap;
use std::io::Read;
use std::sync::OnceLock;

use anyhow::{Context, Result};

/// Hanzi to its tone-3 pinyin syllable (`你` -> `ni3`).
const CHARS_GZ: &[u8] = include_bytes!("../data/zh-chars.tsv.gz");
/// Multi-character words and their readings.
const PHRASES_GZ: &[u8] = include_bytes!("../data/zh-phrases.tsv.gz");
/// Tone-3 pinyin syllable to `misaki` IPA (`ni3` -> `ni↓`).
const IPA_GZ: &[u8] = include_bytes!("../data/zh-ipa.tsv.gz");

/// The Han ideograph range `misaki` segments on. Deliberately the same bounds:
/// text outside it is passed through untouched, exactly as the reference does.
const HAN: std::ops::RangeInclusive<char> = '\u{4E00}'..='\u{9FFF}';

/// Combining inverted breve below, which marks the non-syllabic half of a
/// diphthong in the syllable table (`ai̯`). Kokoro has no token for it and
/// `misaki` strips it from its final output, so this does too.
const NON_SYLLABIC: char = '\u{032F}';

struct Tables {
    chars: HashMap<char, String>,
    phrases: HashMap<String, Vec<String>>,
    ipa: HashMap<String, String>,
    /// Longest phrase in `phrases`, in characters, bounding the match scan.
    longest_phrase: usize,
    jieba: jieba_rs::Jieba,
}

fn gunzip(bytes: &[u8]) -> Result<String> {
    let mut out = String::new();
    flate2::read::GzDecoder::new(bytes)
        .read_to_string(&mut out)
        .context("decompressing an embedded Mandarin table")?;
    Ok(out)
}

fn tables() -> Result<&'static Tables> {
    static TABLES: OnceLock<Result<Tables, String>> = OnceLock::new();
    TABLES
        .get_or_init(|| load_tables().map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn load_tables() -> Result<Tables> {
    let mut chars = HashMap::new();
    for line in gunzip(CHARS_GZ)?.lines() {
        let Some((k, v)) = line.split_once('\t') else {
            continue;
        };
        let Some(c) = k.chars().next() else {
            continue;
        };
        chars.insert(c, v.to_string());
    }
    let mut phrases = HashMap::new();
    for line in gunzip(PHRASES_GZ)?.lines() {
        if let Some((k, v)) = line.split_once('\t') {
            phrases.insert(k.to_string(), v.split(' ').map(str::to_string).collect());
        }
    }
    let mut ipa = HashMap::new();
    for line in gunzip(IPA_GZ)?.lines() {
        if let Some((k, v)) = line.split_once('\t') {
            ipa.insert(k.to_string(), v.to_string());
        }
    }
    let longest_phrase = phrases.keys().map(|p| p.chars().count()).max().unwrap_or(1);
    Ok(Tables {
        chars,
        phrases,
        ipa,
        longest_phrase,
        jieba: jieba_rs::Jieba::new(),
    })
}

/// Rewrite CJK punctuation as its ASCII equivalent.
///
/// The token vocabulary has `,` and `.` but not `，` and `。`, and the
/// difference between them is a real difference in pause length. This is
/// `misaki`'s `map_punctuation`, kept in the same order.
fn map_punctuation(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '、' | '，' => out.push_str(", "),
            '。' | '．' => out.push_str(". "),
            '！' => out.push_str("! "),
            '：' => out.push_str(": "),
            '；' => out.push_str("; "),
            '？' => out.push_str("? "),
            '«' | '《' | '「' | '【' => out.push_str(" “"),
            '»' | '》' | '」' | '】' => out.push_str("” "),
            '（' => out.push_str(" ("),
            '）' => out.push_str(") "),
            _ => out.push(ch),
        }
    }
    out.trim().to_string()
}

/// Read one segmented word as pinyin syllables, longest phrase first.
///
/// A whole-word lookup is not enough. `pypinyin` sub-segments inside the word
/// it is given, so 音乐会 is read as 音乐 + 会 and keeps the yue4 reading of 乐;
/// looking up only 音乐会 would miss the phrase and fall back to per-character,
/// giving le4. Scanning greedily for the longest phrase at each position is
/// what reproduces that.
fn word_to_syllables(t: &Tables, word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let mut out = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let longest = t.longest_phrase.min(chars.len() - i);
        let mut matched = 0;
        for len in (2..=longest).rev() {
            let candidate: String = chars[i..i + len].iter().collect();
            if let Some(syllables) = t.phrases.get(&candidate) {
                out.extend(syllables.iter().cloned());
                matched = len;
                break;
            }
        }
        if matched == 0 {
            if let Some(s) = t.chars.get(&chars[i]) {
                out.push(s.clone());
            }
            matched = 1;
        }
        i += matched;
    }
    out
}

/// Pronounce one segmented word.
fn word_to_ipa(t: &Tables, word: &str) -> String {
    word_to_syllables(t, word)
        .iter()
        .filter_map(|s| t.ipa.get(s).map(String::as_str))
        .collect()
}

/// Convert Mandarin `text` to `misaki`-compatible IPA.
///
/// Runs of Han characters are segmented and pronounced; everything else is
/// passed through untouched, which is what the reference does. Latin words
/// embedded in Chinese therefore survive as Latin and are dropped later by the
/// token mapping — the same behaviour as `misaki`'s legacy path, which has no
/// English callback.
///
/// # Errors
/// Returns an error if the embedded tables cannot be decompressed.
pub fn phonemize(text: &str) -> Result<String> {
    let t = tables()?;
    // Numbers first, before punctuation and before segmentation, which is the
    // order the reference uses. Digits left alone would reach the phoneme table,
    // find no token there, and be dropped: 价格是99元 would lose its price.
    let text = crate::zh_numbers::transform(text);
    let text = map_punctuation(&text);

    let mut out = String::new();
    let mut run = String::new();
    let mut run_is_han = text.chars().next().is_some_and(|c| HAN.contains(&c));

    for ch in text.chars() {
        let is_han = HAN.contains(&ch);
        if is_han == run_is_han {
            run.push(ch);
        } else {
            flush(t, &mut out, &run, run_is_han);
            run.clear();
            run.push(ch);
            run_is_han = is_han;
        }
    }
    flush(t, &mut out, &run, run_is_han);

    Ok(out.replace(NON_SYLLABIC, ""))
}

fn flush(t: &Tables, out: &mut String, run: &str, is_han: bool) {
    if run.is_empty() {
        return;
    }
    if is_han {
        // HMM on, matching `jieba.lcut(..., cut_all=False)`, whose HMM
        // parameter defaults to True. With it off the segmenter produces
        // shorter words, and shorter words miss phrase readings: 音乐会 splits
        // and 乐 reverts from yue4 to le4.
        let words = t.jieba.cut(run, true);
        let spoken: Vec<String> = words.iter().map(|w| word_to_ipa(t, w.word)).collect();
        out.push_str(&spoken.join(" "));
    } else {
        out.push_str(run);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tables have to actually be there and parse. A truncated table would
    /// otherwise surface as silence for whichever words it lost.
    #[test]
    fn the_tables_load() {
        let t = tables().expect("tables load");
        assert!(t.chars.len() > 40_000, "chars: {}", t.chars.len());
        assert!(t.phrases.len() > 46_000, "phrases: {}", t.phrases.len());
        assert!(t.ipa.len() > 1_400, "ipa: {}", t.ipa.len());
    }

    /// The whole reason the phrase table is carried. Character-by-character
    /// lookup gets these wrong, and they are ordinary words.
    #[test]
    fn phrase_readings_beat_character_readings() {
        // 一个 is yi2ge4 (a↗ on 一), not yi1ge4 (a→).
        let one = phonemize("一个").unwrap();
        assert!(one.contains('↗'), "一个 lost its phrase reading: {one:?}");
        // 银行 is yin2hang2 (xa↗ŋ), not yin2xing2 (ɕi↗ŋ).
        let bank = phonemize("银行").unwrap();
        assert!(
            bank.contains("xa↗ŋ"),
            "银行 read as 'xing' rather than 'hang': {bank:?}"
        );
    }

    /// Digits used to reach the phoneme table, which has no token for them,
    /// and vanish — 价格是99元 was spoken as "the price is ___ yuan". They must
    /// be spelled out before anything else looks at the text.
    #[test]
    fn numbers_are_spoken_rather_than_dropped() {
        let out = phonemize("价格是99元。").unwrap();
        assert!(
            !out.contains('9'),
            "digits survived into the phonemes: {out:?}"
        );
        // 九十九 is jiu3-shi2-jiu3.
        assert!(out.contains("ʨjou↓ʂɨ↗ʨjou↓"), "99 was not spoken: {out:?}");
    }

    #[test]
    fn cjk_punctuation_becomes_its_ascii_equivalent() {
        assert_eq!(map_punctuation("你好，世界。"), "你好, 世界.");
        assert_eq!(map_punctuation("真的？！"), "真的? !");
    }

    /// Latin text inside a Chinese sentence is passed through rather than
    /// guessed at, matching the reference.
    #[test]
    fn non_han_text_passes_through() {
        let out = phonemize("他说OK。").unwrap();
        assert!(out.contains("OK"), "{out:?}");
    }

    #[test]
    fn empty_input_is_empty_output() {
        assert_eq!(phonemize("").unwrap(), "");
        assert_eq!(phonemize("   ").unwrap(), "");
    }

    /// No combining mark may survive: Kokoro has no token for the non-syllabic
    /// diacritic the syllable table carries.
    /// The claim this module rests on: it reproduces `misaki`, the g2p Kokoro
    /// was trained with. `data/zh-golden.tsv` holds misaki's own answers for a
    /// corpus of natural sentences and the most frequent words in jieba's
    /// dictionary; every one of them must come back identical. Anything else is
    /// a pronunciation the model was never taught.
    #[test]
    fn matches_the_reference_g2p() {
        let golden = include_str!("../data/zh-golden.tsv");
        let mut checked = 0;
        let mut wrong = Vec::new();
        for line in golden.lines() {
            let Some((text, expected)) = line.split_once('\t') else {
                continue;
            };
            let got = phonemize(text).expect("phonemize");
            checked += 1;
            if got != expected {
                wrong.push(format!("  {text}\n    want {expected:?}\n    got  {got:?}"));
            }
        }
        assert!(checked > 400, "fixture only had {checked} cases");
        assert!(
            wrong.is_empty(),
            "{} of {checked} disagree with misaki:\n{}",
            wrong.len(),
            wrong
                .iter()
                .take(15)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn the_non_syllabic_mark_is_stripped() {
        let out = phonemize("你好，世界。").unwrap();
        assert!(!out.contains(NON_SYLLABIC), "{out:?}");
    }
}
