// SPDX-License-Identifier: GPL-3.0-only
//! English G2P: a port of misaki's `en.py`, the pipeline Kokoro's English
//! voices were trained on.
//!
//! misaki's English is not espeak: it is a lexicon-first, POS-aware pipeline
//! — 180k dictionary entries per dialect (790 of them keyed by part of
//! speech, which is how "read" gets past tense and "record" gets noun
//! stress), numbers/years/ordinals/currency verbalized into words,
//! punctuation carried through for prosody, and espeak consulted only as the
//! fallback for words the lexicon does not know. The reference is
//! `misaki==0.9.4`, the version the `kokoro` package pins; the four lexicon
//! files in `data/` are byte-for-byte copies from that wheel, and
//! `data/en-golden.tsv` holds that pipeline's own answers, which the tests
//! replay.
//!
//! The port follows `en.py` function by function, oddities included — where
//! something looks arbitrary (the 0.5-index stress insertion, the
//! "demote the weakest half" rule in [`resolve_tokens`]), it is that way
//! because the reference is that way, and inventing a nicer rule here would
//! change pronunciation relative to what the model was trained on.
//!
//! Tokenization and POS tags come from the caller: misaki gets them from
//! spaCy (`en_core_web_sm`), and reproducing that layer is a separate
//! concern from this file's lexicon/number/stress logic. The golden test
//! feeds this pipeline spaCy's own tokens, captured in `data/en-tokens.tsv`.

use std::collections::HashMap;
use std::sync::OnceLock;

use unicode_normalization::UnicodeNormalization;

const GOLD_US_GZ: &[u8] = include_bytes!("../data/en-gold-us.json.gz");
const SILVER_US_GZ: &[u8] = include_bytes!("../data/en-silver-us.json.gz");
const GOLD_GB_GZ: &[u8] = include_bytes!("../data/en-gold-gb.json.gz");
const SILVER_GB_GZ: &[u8] = include_bytes!("../data/en-silver-gb.json.gz");

const DIPHTHONGS: &str = "AIOQWYʤʧ";
const PUNCTS: &str = ";:,.!?—…\"“”";
const NON_QUOTE_PUNCTS: &str = ";:,.!?—…";
const SUBTOKEN_JUNKS: &str = "',-._‘’/";
const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";
const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";

const PRIMARY_STRESS: char = 'ˈ';
const SECONDARY_STRESS: char = 'ˌ';

/// spaCy tags that mark punctuation, from spaCy's glossary.
const PUNCT_TAGS: [&str; 11] = [
    ".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP",
];
const ORDINALS: [&str; 4] = ["st", "nd", "rd", "th"];

fn punct_tag_phonemes(tag: &str) -> Option<&'static str> {
    match tag {
        "-LRB-" => Some("("),
        "-RRB-" => Some(")"),
        "``" => Some("\u{201c}"),
        "\"\"" | "''" => Some("\u{201d}"),
        _ => None,
    }
}

/// `(singular, sub-unit)` per currency symbol.
fn currency_units(c: char) -> Option<(&'static str, &'static str)> {
    match c {
        '$' => Some(("dollar", "cent")),
        '£' => Some(("pound", "pence")),
        '€' => Some(("euro", "cent")),
        _ => None,
    }
}

fn add_symbol(word: &str) -> Option<&'static str> {
    match word {
        "." => Some("dot"),
        "/" => Some("slash"),
        _ => None,
    }
}

fn symbol(word: &str) -> Option<&'static str> {
    match word {
        "%" => Some("percent"),
        "&" => Some("and"),
        "+" => Some("plus"),
        "@" => Some("at"),
        _ => None,
    }
}

/// The only codepoints lexicon words may contain: `'`, `-`, ASCII letters.
fn in_lexicon_ords(c: char) -> bool {
    c == '\'' || c == '-' || c.is_ascii_alphabetic()
}

fn is_digits(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| c.is_ascii_digit())
}

fn is_vowel(c: char) -> bool {
    VOWELS.contains(c)
}

pub fn stress_weight(ps: &str) -> usize {
    ps.chars()
        .map(|c| if DIPHTHONGS.contains(c) { 2 } else { 1 })
        .sum()
}

/// Python `str.capitalize()`: first char uppercased, the rest lowercased.
fn capitalize(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first
            .to_uppercase()
            .chain(chars.as_str().to_lowercase().chars())
            .collect(),
    }
}

/// Reposition a leading stress mark to just before the first vowel at or
/// after it. misaki does this by giving the stress a half-index and
/// re-sorting; the sort is stable, so equal-keyed characters keep order.
fn restress(ps: &str) -> String {
    let chars: Vec<char> = ps.chars().collect();
    let mut keyed: Vec<(f64, char)> = Vec::with_capacity(chars.len());
    for (i, &c) in chars.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let mut key = i as f64;
        if (c == PRIMARY_STRESS || c == SECONDARY_STRESS)
            && let Some(j) = (i..chars.len()).find(|&j| is_vowel(chars[j]))
        {
            #[allow(clippy::cast_precision_loss)]
            let vowel_key = j as f64;
            key = vowel_key - 0.5;
        }
        keyed.push((key, c));
    }
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("keys are finite"));
    keyed.into_iter().map(|(_, c)| c).collect()
}

/// misaki's `apply_stress`: `stress` is a small signed level (`-2` strips
/// stress, `-0.5`/`-1` demote, `0.5`/`2` used for capitalized words, …).
#[allow(clippy::float_cmp, clippy::collapsible_str_replace)]
pub fn apply_stress(ps: &str, stress: Option<f64>) -> String {
    let Some(stress) = stress else {
        return ps.to_string();
    };
    let has_primary = ps.contains(PRIMARY_STRESS);
    let has_secondary = ps.contains(SECONDARY_STRESS);
    let any_vowel = ps.chars().any(is_vowel);
    if stress < -1.0 {
        ps.replace(PRIMARY_STRESS, "").replace(SECONDARY_STRESS, "")
    } else if stress == -1.0 || ((stress == 0.0 || stress == -0.5) && has_primary) {
        ps.replace(SECONDARY_STRESS, "")
            .replace(PRIMARY_STRESS, &SECONDARY_STRESS.to_string())
    } else if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has_primary && !has_secondary {
        if !any_vowel {
            return ps.to_string();
        }
        restress(&format!("{SECONDARY_STRESS}{ps}"))
    } else if stress >= 1.0 && !has_primary && has_secondary {
        ps.replace(SECONDARY_STRESS, &PRIMARY_STRESS.to_string())
    } else if stress > 1.0 && !has_primary && !has_secondary {
        if !any_vowel {
            return ps.to_string();
        }
        restress(&format!("{PRIMARY_STRESS}{ps}"))
    } else {
        ps.to_string()
    }
}

/// A lexicon entry: a plain pronunciation, or one keyed by POS tag with a
/// `DEFAULT`. Tag values may be `null` in the data, meaning "fall through
/// to letter-by-letter NNP spelling".
#[derive(Clone)]
enum Entry {
    Ps(String),
    Tagged(HashMap<String, Option<String>>),
}

pub struct Lexicon {
    british: bool,
    golds: HashMap<String, Entry>,
    silvers: HashMap<String, Entry>,
}

fn parse_lexicon(gz: &[u8]) -> HashMap<String, Entry> {
    use std::io::Read;
    let mut json = String::new();
    flate2::read::GzDecoder::new(gz)
        .read_to_string(&mut json)
        .expect("embedded lexicon decompresses");
    let raw: HashMap<String, serde_json::Value> =
        serde_json::from_str(&json).expect("embedded lexicon parses");
    let mut out = HashMap::with_capacity(raw.len() * 2);
    let mut originals = HashMap::with_capacity(raw.len());
    for (k, v) in raw {
        let entry = match v {
            serde_json::Value::String(s) => Entry::Ps(s),
            serde_json::Value::Object(m) => Entry::Tagged(
                m.into_iter()
                    .map(|(tag, v)| match v {
                        serde_json::Value::String(s) => (tag, Some(s)),
                        serde_json::Value::Null => (tag, None),
                        other => panic!("unexpected lexicon value {other:?}"),
                    })
                    .collect(),
            ),
            other => panic!("unexpected lexicon value {other:?}"),
        };
        originals.insert(k, entry);
    }
    // misaki's `grow_dictionary`: give lowercase entries a Capitalized twin
    // and vice versa; original keys win over grown ones.
    for (k, v) in &originals {
        if k.chars().count() < 2 {
            continue;
        }
        let lower = k.to_lowercase();
        let cap = capitalize(k);
        if *k == lower {
            if *k != cap {
                out.insert(cap, v.clone());
            }
        } else if *k == capitalize(&lower) {
            out.insert(lower, v.clone());
        }
    }
    out.extend(originals);
    out
}

/// The right-to-left context misaki threads through a sentence: whether the
/// *next* word starts with a vowel (drives "the"/"to"/"a"), and whether it
/// is "to" (drives "used to").
#[derive(Clone, Copy, Default)]
pub struct TokenContext {
    future_vowel: Option<bool>,
    future_to: bool,
}

/// `NOUN`/`VERB`/`ADV`/`ADJ`, or the tag unchanged.
fn parent_tag(tag: &str) -> &str {
    if tag.starts_with("VB") {
        "VERB"
    } else if tag.starts_with("NN") {
        "NOUN"
    } else if tag.starts_with("ADV") || tag.starts_with("RB") {
        "ADV"
    } else if tag.starts_with("ADJ") || tag.starts_with("JJ") {
        "ADJ"
    } else {
        tag
    }
}

type PsRating = (Option<String>, Option<u8>);

impl Lexicon {
    pub fn new(british: bool) -> Self {
        let (gold, silver) = if british {
            (GOLD_GB_GZ, SILVER_GB_GZ)
        } else {
            (GOLD_US_GZ, SILVER_US_GZ)
        };
        Self {
            british,
            golds: parse_lexicon(gold),
            silvers: parse_lexicon(silver),
        }
    }

    pub fn shared(british: bool) -> &'static Self {
        static US: OnceLock<Lexicon> = OnceLock::new();
        static GB: OnceLock<Lexicon> = OnceLock::new();
        if british {
            GB.get_or_init(|| Self::new(true))
        } else {
            US.get_or_init(|| Self::new(false))
        }
    }

    fn gold_str(&self, word: &str) -> Option<&str> {
        match self.golds.get(word) {
            Some(Entry::Ps(s)) => Some(s),
            _ => None,
        }
    }

    /// Spell a word letter by letter from the single-letter gold entries.
    fn get_nnp(&self, word: &str) -> PsRating {
        let mut ps = String::new();
        for c in word.chars().filter(|c| c.is_alphabetic()) {
            let upper: String = c.to_uppercase().collect();
            match self.gold_str(&upper) {
                Some(p) => ps.push_str(p),
                None => return (None, None),
            }
        }
        let ps = apply_stress(&ps, Some(0.0));
        match ps.rfind(SECONDARY_STRESS) {
            Some(i) => {
                let (head, tail) = ps.split_at(i);
                let tail = &tail[SECONDARY_STRESS.len_utf8()..];
                (Some(format!("{head}{PRIMARY_STRESS}{tail}")), Some(3))
            }
            None => (Some(ps), Some(3)),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn get_special_case(
        &self,
        word: &str,
        tag: &str,
        stress: Option<f64>,
        ctx: TokenContext,
    ) -> PsRating {
        if tag == "ADD" && add_symbol(word).is_some() {
            return self.lookup(add_symbol(word).unwrap(), None, Some(-0.5), Some(ctx));
        } else if let Some(sym) = symbol(word) {
            return self.lookup(sym, None, None, Some(ctx));
        } else if word.trim_matches('.').contains('.')
            && word.replace('.', "").chars().all(char::is_alphabetic)
            && word
                .split('.')
                .map(|p| p.chars().count())
                .max()
                .unwrap_or(0)
                < 3
        {
            return self.get_nnp(word);
        } else if word == "a" || word == "A" {
            return if tag == "DT" {
                (Some("ɐ".into()), Some(4))
            } else {
                (Some("ˈA".into()), Some(4))
            };
        } else if word == "am" || word == "Am" || word == "AM" {
            if tag.starts_with("NN") {
                return self.get_nnp(word);
            } else if ctx.future_vowel.is_none() || word != "am" || stress.is_some_and(|s| s > 0.0)
            {
                return (self.gold_str("am").map(String::from), Some(4));
            }
            return (Some("ɐm".into()), Some(4));
        } else if word == "an" || word == "An" || word == "AN" {
            if word == "AN" && tag.starts_with("NN") {
                return self.get_nnp(word);
            }
            return (Some("ɐn".into()), Some(4));
        } else if word == "I" && tag == "PRP" {
            return (Some(format!("{SECONDARY_STRESS}I")), Some(4));
        } else if (word == "by" || word == "By" || word == "BY") && parent_tag(tag) == "ADV" {
            return (Some("bˈI".into()), Some(4));
        } else if word == "to" || word == "To" || (word == "TO" && (tag == "TO" || tag == "IN")) {
            return (
                Some(match ctx.future_vowel {
                    None => self.gold_str("to").expect("gold has 'to'").to_string(),
                    Some(false) => "tə".into(),
                    Some(true) => "tʊ".into(),
                }),
                Some(4),
            );
        } else if word == "in" || word == "In" || (word == "IN" && tag != "NNP") {
            let stress_mark = if ctx.future_vowel.is_none() || tag != "IN" {
                PRIMARY_STRESS.to_string()
            } else {
                String::new()
            };
            return (Some(format!("{stress_mark}ɪn")), Some(4));
        } else if word == "the" || word == "The" || (word == "THE" && tag == "DT") {
            return (
                Some(
                    if ctx.future_vowel == Some(true) {
                        "ði"
                    } else {
                        "ðə"
                    }
                    .into(),
                ),
                Some(4),
            );
        } else if tag == "IN" && {
            let wl = word.to_lowercase();
            wl == "vs" || wl == "vs."
        } {
            return self.lookup("versus", None, None, Some(ctx));
        } else if word == "used" || word == "Used" || word == "USED" {
            let Some(Entry::Tagged(entry)) = self.golds.get("used") else {
                panic!("gold has POS-keyed 'used'");
            };
            if (tag == "VBD" || tag == "JJ") && ctx.future_to {
                return (entry.get("VBD").cloned().flatten(), Some(4));
            }
            return (entry.get("DEFAULT").cloned().flatten(), Some(4));
        }
        (None, None)
    }

    fn is_known(&self, word: &str, _tag: &str) -> bool {
        if self.golds.contains_key(word)
            || symbol(word).is_some()
            || self.silvers.contains_key(word)
        {
            return true;
        }
        if !word.chars().all(char::is_alphabetic) || !word.chars().all(in_lexicon_ords) {
            return false;
        }
        if word.chars().count() == 1 {
            return true;
        }
        if word == word.to_uppercase() && self.golds.contains_key(&word.to_lowercase()) {
            return true;
        }
        let tail: String = word.chars().skip(1).collect();
        tail == tail.to_uppercase()
    }

    fn lookup(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f64>,
        ctx: Option<TokenContext>,
    ) -> PsRating {
        let mut word = word.to_string();
        let mut is_nnp: Option<bool> = None;
        if word == word.to_uppercase() && !self.golds.contains_key(&word) {
            word = word.to_lowercase();
            is_nnp = Some(tag == Some("NNP"));
        }
        let mut rating = Some(4);
        let mut entry = self.golds.get(&word);
        if entry.is_none() && is_nnp != Some(true) {
            entry = self.silvers.get(&word);
            rating = Some(3);
        }
        let mut ps: Option<String> = None;
        if let Some(entry) = entry {
            ps = match entry {
                Entry::Ps(s) => Some(s.clone()),
                Entry::Tagged(m) => {
                    let mut key = tag.unwrap_or("None");
                    if ctx.is_some_and(|c| c.future_vowel.is_none()) && m.contains_key("None") {
                        key = "None";
                    } else if !m.contains_key(key) {
                        key = parent_tag(key);
                    }
                    m.get(key)
                        .cloned()
                        .unwrap_or_else(|| m.get("DEFAULT").cloned().expect("DEFAULT exists"))
                }
            };
        }
        if ps.is_none()
            || (is_nnp == Some(true) && !ps.as_deref().unwrap_or("").contains(PRIMARY_STRESS))
        {
            let (nnp_ps, nnp_rating) = self.get_nnp(&word);
            if nnp_ps.is_some() {
                return (nnp_ps, nnp_rating);
            }
        }
        (ps.map(|p| apply_stress(&p, stress)), rating)
    }

    /// <https://en.wiktionary.org/wiki/-s>
    fn suffix_s(&self, stem: &str) -> Option<String> {
        let last = stem.chars().last()?;
        Some(if "ptkfθ".contains(last) {
            format!("{stem}s")
        } else if "szʃʒʧʤ".contains(last) {
            format!("{stem}{}z", if self.british { 'ɪ' } else { 'ᵻ' })
        } else {
            format!("{stem}z")
        })
    }

    /// <https://en.wiktionary.org/wiki/-ed>
    fn suffix_ed(&self, stem: &str) -> Option<String> {
        let chars: Vec<char> = stem.chars().collect();
        let last = *chars.last()?;
        Some(if "pkfθʃsʧ".contains(last) {
            format!("{stem}t")
        } else if last == 'd' {
            format!("{stem}{}d", if self.british { 'ɪ' } else { 'ᵻ' })
        } else if last != 't' {
            format!("{stem}d")
        } else if self.british || chars.len() < 2 {
            format!("{stem}ɪd")
        } else if US_TAUS.contains(chars[chars.len() - 2]) {
            let head: String = chars[..chars.len() - 1].iter().collect();
            format!("{head}ɾᵻd")
        } else {
            format!("{stem}ᵻd")
        })
    }

    /// <https://en.wiktionary.org/wiki/-ing>
    fn suffix_ing(&self, stem: &str) -> Option<String> {
        let chars: Vec<char> = stem.chars().collect();
        let last = *chars.last()?;
        if self.british {
            if last == 'ə' || last == 'ː' {
                return None;
            }
        } else if chars.len() > 1 && last == 't' && US_TAUS.contains(chars[chars.len() - 2]) {
            let head: String = chars[..chars.len() - 1].iter().collect();
            return Some(format!("{head}ɾɪŋ"));
        }
        Some(format!("{stem}ɪŋ"))
    }

    fn stem_s(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f64>,
        ctx: Option<TokenContext>,
    ) -> PsRating {
        let n = word.chars().count();
        if n < 3 || !word.ends_with('s') {
            return (None, None);
        }
        let tag_str = tag.unwrap_or("None");
        let stem = if !word.ends_with("ss") && self.is_known(&word[..word.len() - 1], tag_str) {
            word[..word.len() - 1].to_string()
        } else if (word.ends_with("'s")
            || (n > 4 && word.ends_with("es") && !word.ends_with("ies")))
            && self.is_known(&word[..word.len() - 2], tag_str)
        {
            word[..word.len() - 2].to_string()
        } else if n > 4
            && word.ends_with("ies")
            && self.is_known(&format!("{}y", &word[..word.len() - 3]), tag_str)
        {
            format!("{}y", &word[..word.len() - 3])
        } else {
            return (None, None);
        };
        let (stem_ps, rating) = self.lookup(&stem, tag, stress, ctx);
        (stem_ps.and_then(|p| self.suffix_s(&p)), rating)
    }

    fn stem_ed(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f64>,
        ctx: Option<TokenContext>,
    ) -> PsRating {
        let n = word.chars().count();
        if n < 4 || !word.ends_with('d') {
            return (None, None);
        }
        let tag_str = tag.unwrap_or("None");
        let stem = if !word.ends_with("dd") && self.is_known(&word[..word.len() - 1], tag_str) {
            word[..word.len() - 1].to_string()
        } else if n > 4
            && word.ends_with("ed")
            && !word.ends_with("eed")
            && self.is_known(&word[..word.len() - 2], tag_str)
        {
            word[..word.len() - 2].to_string()
        } else {
            return (None, None);
        };
        let (stem_ps, rating) = self.lookup(&stem, tag, stress, ctx);
        (stem_ps.and_then(|p| self.suffix_ed(&p)), rating)
    }

    fn stem_ing(
        &self,
        word: &str,
        tag: Option<&str>,
        stress: Option<f64>,
        ctx: Option<TokenContext>,
    ) -> PsRating {
        let n = word.chars().count();
        if n < 5 || !word.ends_with("ing") {
            return (None, None);
        }
        let tag_str = tag.unwrap_or("None");
        let doubled = {
            // en.py: r'([bcdgklmnprstvxz])\1ing$|cking$'
            let chars: Vec<char> = word.chars().collect();
            let len = chars.len();
            (len > 4
                && chars[len - 4] == chars[len - 5]
                && "bcdgklmnprstvxz".contains(chars[len - 4]))
                || word.ends_with("cking")
        };
        let stem = if n > 5 && self.is_known(&word[..word.len() - 3], tag_str) {
            word[..word.len() - 3].to_string()
        } else if self.is_known(&format!("{}e", &word[..word.len() - 3]), tag_str) {
            format!("{}e", &word[..word.len() - 3])
        } else if n > 5 && doubled && self.is_known(&word[..word.len() - 4], tag_str) {
            word[..word.len() - 4].to_string()
        } else {
            return (None, None);
        };
        let (stem_ps, rating) = self.lookup(&stem, tag, stress, ctx);
        (stem_ps.and_then(|p| self.suffix_ing(&p)), rating)
    }

    fn get_word(&self, word: &str, tag: &str, stress: Option<f64>, ctx: TokenContext) -> PsRating {
        let (ps, rating) = self.get_special_case(word, tag, stress, ctx);
        if ps.is_some() {
            return (ps, rating);
        }
        let mut word = word.to_string();
        let wl = word.to_lowercase();
        let tail: String = word.chars().skip(1).collect();
        if word.chars().count() > 1
            && word.replace('\'', "").chars().all(char::is_alphabetic)
            && word != wl
            && (tag != "NNP" || word.chars().count() > 7)
            && !self.golds.contains_key(&word)
            && !self.silvers.contains_key(&word)
            && (word == word.to_uppercase() || tail == tail.to_lowercase())
            && (self.golds.contains_key(&wl)
                || self.silvers.contains_key(&wl)
                || [Self::stem_s, Self::stem_ed, Self::stem_ing]
                    .iter()
                    .any(|f| f(self, &wl, Some(tag), stress, Some(ctx)).0.is_some()))
        {
            word = wl;
        }
        if self.is_known(&word, tag) {
            return self.lookup(&word, Some(tag), stress, Some(ctx));
        } else if word.ends_with("s'")
            && self.is_known(&format!("{}'s", &word[..word.len() - 2]), tag)
        {
            return self.lookup(
                &format!("{}'s", &word[..word.len() - 2]),
                Some(tag),
                stress,
                Some(ctx),
            );
        } else if word.ends_with('\'') && self.is_known(&word[..word.len() - 1], tag) {
            return self.lookup(&word[..word.len() - 1], Some(tag), stress, Some(ctx));
        }
        let (s, rating) = self.stem_s(&word, Some(tag), stress, Some(ctx));
        if s.is_some() {
            return (s, rating);
        }
        let (ed, rating) = self.stem_ed(&word, Some(tag), stress, Some(ctx));
        if ed.is_some() {
            return (ed, rating);
        }
        let (ing, rating) = self.stem_ing(&word, Some(tag), Some(stress.unwrap_or(0.5)), Some(ctx));
        if ing.is_some() {
            return (ing, rating);
        }
        (None, None)
    }
}

// ---------------------------------------------------------------------------
// Numbers

/// Split num2words output into words the way `re.split(r'[^a-z]+')` does.
fn number_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_lowercase())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// `num2words(float(word))`: "12.75" -> "twelve point seven five".
fn float_words(word: &str) -> Option<String> {
    let value: f64 = word.parse().ok()?;
    // Python derives precision from str(float(word)), which drops trailing
    // zeros; Rust's shortest-roundtrip Display does the same.
    let canon = format!("{value}");
    let (pre, post) = canon.split_once('.').unwrap_or((&canon, ""));
    let mut out = en_numbers::cardinal(pre.parse().ok()?);
    if !post.is_empty() {
        out.push_str(" point");
        for d in post.chars() {
            out.push(' ');
            out.push_str(&en_numbers::cardinal(u128::from(d.to_digit(10)?)));
        }
    }
    Some(out)
}

use crate::en_numbers;

impl Lexicon {
    fn is_currency_amount(word: &str) -> bool {
        if !word.contains('.') {
            return true;
        }
        if word.matches('.').count() > 1 {
            return false;
        }
        let cents = word.split('.').nth(1).unwrap_or("");
        // en.py compares `set(cents) == {0}` — a set of *characters* against
        // a set holding the *integer* zero, which is never true. The length
        // check is therefore the only live condition, and this port keeps
        // that behaviour rather than "fixing" it into a new one.
        cents.chars().count() < 3
    }

    #[allow(clippy::too_many_lines)]
    fn get_number(
        &self,
        word: &str,
        currency: Option<char>,
        is_head: bool,
        num_flags: &str,
    ) -> PsRating {
        let suffix: String = {
            let tail: Vec<char> = word
                .chars()
                .rev()
                .take_while(|&c| c.is_ascii_lowercase() || c == '\'')
                .collect();
            tail.into_iter().rev().collect()
        };
        let word = &word[..word.len() - suffix.len()];
        let mut result: Vec<(String, u8)> = Vec::new();
        let push = |r: &mut Vec<(String, u8)>, pr: PsRating| {
            let (ps, rating) = pr;
            r.push((ps.unwrap_or_default(), rating.unwrap_or(4)));
        };
        let word = if let Some(rest) = word.strip_prefix('-') {
            push(&mut result, self.lookup("minus", None, None, None));
            rest
        } else {
            word
        };

        let extend_num = |lex: &Self, result: &mut Vec<(String, u8)>, text: &str, first: bool| {
            let splits = number_words(text);
            let n = splits.len();
            for (i, w) in splits.iter().enumerate() {
                if w != "and" || num_flags.contains('&') {
                    if first && i == 0 && n > 1 && w == "one" && num_flags.contains('a') {
                        result.push(("ə".to_string(), 4));
                    } else {
                        let stress = if w == "point" { Some(-2.0) } else { None };
                        let (ps, rating) = lex.lookup(w, None, stress, None);
                        result.push((ps.unwrap_or_default(), rating.unwrap_or(4)));
                    }
                } else if w == "and"
                    && num_flags.contains('n')
                    && let Some(last) = result.last_mut()
                {
                    last.0.push_str("ən");
                }
            }
        };
        let extend_int = |lex: &Self, result: &mut Vec<(String, u8)>, num: &str, first: bool| {
            let Ok(n) = num.parse::<u128>() else { return };
            extend_num(lex, result, &en_numbers::cardinal(n), first);
        };

        let is_ordinal_suffix = ORDINALS.contains(&suffix.as_str());
        if is_digits(word) && is_ordinal_suffix {
            let n: u128 = word.parse().expect("digits parse");
            extend_num(self, &mut result, &en_numbers::ordinal(n), true);
        } else if result.is_empty()
            && word.chars().count() == 4
            && currency.and_then(currency_units).is_none()
            && is_digits(word)
        {
            let n: u128 = word.parse().expect("digits parse");
            extend_num(self, &mut result, &en_numbers::year(n), true);
        } else if !is_head && !word.contains('.') {
            let num = word.replace(',', "");
            if num.starts_with('0') || num.chars().count() > 3 {
                for c in num.chars() {
                    extend_int(self, &mut result, &c.to_string(), false);
                }
            } else if num.chars().count() == 3 && !num.ends_with("00") {
                let digits: Vec<char> = num.chars().collect();
                extend_int(self, &mut result, &digits[0].to_string(), true);
                if digits[1] == '0' {
                    push(&mut result, self.lookup("O", None, Some(-2.0), None));
                    extend_int(self, &mut result, &digits[2].to_string(), false);
                } else {
                    let tail: String = digits[1..].iter().collect();
                    extend_int(self, &mut result, &tail, false);
                }
            } else {
                extend_int(self, &mut result, &num, true);
            }
        } else if word.matches('.').count() > 1 || !is_head {
            let mut first = true;
            for num in word.replace(',', "").split('.') {
                if num.is_empty() {
                } else if num.starts_with('0')
                    || (num.chars().count() != 2 && num.chars().skip(1).any(|c| c != '0'))
                {
                    for c in num.chars() {
                        extend_int(self, &mut result, &c.to_string(), false);
                    }
                } else {
                    extend_int(self, &mut result, num, first);
                }
                first = false;
            }
        } else if let Some(units) = currency.and_then(currency_units) {
            if Self::is_currency_amount(word) {
                let cleaned = word.replace(',', "");
                let mut pairs: Vec<(u128, &str)> = cleaned
                    .split('.')
                    .zip([units.0, units.1])
                    .map(|(num, unit)| (num.parse().unwrap_or(0), unit))
                    .collect();
                if pairs.len() > 1 {
                    if pairs[1].0 == 0 {
                        pairs.truncate(1);
                    } else if pairs[0].0 == 0 {
                        pairs.remove(0);
                    }
                }
                for (i, (num, unit)) in pairs.iter().enumerate() {
                    if i > 0 {
                        push(&mut result, self.lookup("and", None, None, None));
                    }
                    extend_num(self, &mut result, &en_numbers::cardinal(*num), i == 0);
                    let unit_ps = if *num != 1 && *unit != "pence" {
                        self.stem_s(&format!("{unit}s"), None, None, None)
                    } else {
                        self.lookup(unit, None, None, None)
                    };
                    push(&mut result, unit_ps);
                }
            }
        } else {
            let text = if is_digits(word) {
                word.parse::<u128>().ok().map(en_numbers::cardinal)
            } else if !word.contains('.') {
                let n = word.replace(',', "").parse::<u128>().ok();
                n.map(|n| {
                    if is_ordinal_suffix {
                        en_numbers::ordinal(n)
                    } else {
                        en_numbers::cardinal(n)
                    }
                })
            } else {
                let cleaned = word.replace(',', "");
                if let Some(frac) = cleaned.strip_prefix('.') {
                    let digits: Option<Vec<String>> = frac
                        .chars()
                        .map(|c| c.to_digit(10).map(|d| en_numbers::cardinal(u128::from(d))))
                        .collect();
                    digits.map(|d| format!("point {}", d.join(" ")))
                } else {
                    float_words(&cleaned)
                }
            };
            if let Some(text) = text {
                extend_num(self, &mut result, &text, true);
            }
        }

        if result.is_empty() {
            return (None, None);
        }
        let rating = result.iter().map(|(_, r)| *r).min();
        let joined = result
            .iter()
            .map(|(p, _)| p.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let ps = match suffix.as_str() {
            "s" | "'s" => self.suffix_s(&joined),
            "ed" | "'d" => self.suffix_ed(&joined),
            "ing" => self.suffix_ing(&joined),
            _ => Some(joined),
        };
        (ps, rating)
    }

    fn append_currency(&self, ps: &str, currency: Option<char>) -> String {
        let Some(units) = currency.and_then(currency_units) else {
            return ps.to_string();
        };
        match self.stem_s(&format!("{}s", units.0), None, None, None).0 {
            Some(cur) => format!("{ps} {cur}"),
            None => ps.to_string(),
        }
    }

    fn is_number(word: &str, is_head: bool) -> bool {
        if word.chars().all(|c| !c.is_ascii_digit()) {
            return false;
        }
        let mut word = word;
        for s in ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"] {
            if word.ends_with(s) {
                word = &word[..word.len() - s.len()];
                break;
            }
        }
        word.chars().enumerate().all(|(i, c)| {
            c.is_ascii_digit() || c == ',' || c == '.' || (is_head && i == 0 && c == '-')
        })
    }

    /// misaki's `Lexicon.__call__`: word normalization, capitalization
    /// stress, then the word/number paths.
    fn call_token(&self, tk: &MToken, ctx: TokenContext) -> PsRating {
        let word = tk.alias.as_deref().unwrap_or(&tk.text);
        let word = word.replace(['\u{2018}', '\u{2019}'], "'");
        let word: String = word.nfkc().collect();
        // en.py maps every digit through unicodedata.numeric; for ASCII
        // digits that is the identity, and non-ASCII digits do not survive
        // spaCy tokenization as CD tokens in our inputs.
        let stress = if word == word.to_lowercase() {
            None
        } else if word == word.to_uppercase() {
            Some(2.0)
        } else {
            Some(0.5)
        };
        let (ps, rating) = self.get_word(&word, &tk.tag, stress, ctx);
        if let Some(ps) = ps {
            return (
                Some(apply_stress(
                    &self.append_currency(&ps, tk.currency),
                    tk.stress,
                )),
                rating,
            );
        }
        if Self::is_number(&word, tk.is_head) {
            let (ps, rating) = self.get_number(&word, tk.currency, tk.is_head, &tk.num_flags);
            return (ps.map(|p| apply_stress(&p, tk.stress)), rating);
        }
        if !word.chars().all(in_lexicon_ords) {
            return (None, None);
        }
        (None, None)
    }
}

// ---------------------------------------------------------------------------
// Tokens and the G2P pipeline

/// misaki's `MToken`, flattened (the `_` underscore fields inlined).
#[derive(Clone)]
pub struct MToken {
    pub text: String,
    pub tag: String,
    pub whitespace: String,
    pub phonemes: Option<String>,
    pub is_head: bool,
    pub alias: Option<String>,
    pub stress: Option<f64>,
    pub currency: Option<char>,
    pub num_flags: String,
    pub prespace: bool,
    pub rating: Option<u8>,
}

impl MToken {
    pub fn new(text: &str, tag: &str, whitespace: &str) -> Self {
        Self {
            text: text.to_string(),
            tag: tag.to_string(),
            whitespace: whitespace.to_string(),
            phonemes: None,
            is_head: true,
            alias: None,
            stress: None,
            currency: None,
            num_flags: String::new(),
            prespace: false,
            rating: None,
        }
    }
}

#[allow(clippy::float_cmp)]
fn merge_tokens(tokens: &[MToken], unk: Option<&str>) -> MToken {
    let mut stresses: Vec<f64> = tokens.iter().filter_map(|t| t.stress).collect();
    stresses.dedup_by(|a, b| a == b);
    stresses.sort_by(|a, b| a.partial_cmp(b).expect("stress is finite"));
    stresses.dedup_by(|a, b| a == b);
    let currency = tokens.iter().filter_map(|t| t.currency).max();
    let ratings: Vec<Option<u8>> = tokens.iter().map(|t| t.rating).collect();
    let rating = if ratings.iter().any(Option::is_none) {
        None
    } else {
        ratings.iter().map(|r| r.expect("checked")).min()
    };
    let phonemes = unk.map(|unk| {
        let mut ps = String::new();
        for tk in tokens {
            if tk.prespace
                && !ps.is_empty()
                && !ps.ends_with(char::is_whitespace)
                && tk.phonemes.as_deref().is_some_and(|p| !p.is_empty())
            {
                ps.push(' ');
            }
            ps.push_str(tk.phonemes.as_deref().unwrap_or(unk));
        }
        ps
    });
    let mut text = String::new();
    for tk in &tokens[..tokens.len() - 1] {
        text.push_str(&tk.text);
        text.push_str(&tk.whitespace);
    }
    text.push_str(&tokens[tokens.len() - 1].text);
    let tag = tokens
        .iter()
        .max_by_key(|tk| {
            tk.text
                .chars()
                .map(|c| {
                    let lower: String = c.to_lowercase().collect();
                    if lower == c.to_string() { 1usize } else { 2 }
                })
                .sum::<usize>()
        })
        .expect("at least one token")
        .tag
        .clone();
    let mut num_flags: Vec<char> = tokens.iter().flat_map(|t| t.num_flags.chars()).collect();
    num_flags.sort_unstable();
    num_flags.dedup();
    MToken {
        text,
        tag,
        whitespace: tokens[tokens.len() - 1].whitespace.clone(),
        phonemes,
        is_head: tokens[0].is_head,
        alias: None,
        stress: if stresses.len() == 1 {
            Some(stresses[0])
        } else {
            None
        },
        currency,
        num_flags: num_flags.into_iter().collect(),
        prespace: tokens[0].prespace,
        rating,
    }
}

fn subtokenize(word: &str) -> Vec<String> {
    static RE: OnceLock<fancy_regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        fancy_regex::Regex::new(
            r"^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$",
        )
        .expect("subtoken regex compiles")
    });
    re.find_iter(word)
        .map(|m| m.expect("regex match succeeds").as_str().to_string())
        .collect()
}

enum Word {
    One(MToken),
    Group(Vec<MToken>),
}

/// misaki's `retokenize`: split spaCy tokens into subtokens, resolve
/// punctuation/currency/dash right away, and group subtokens that are not
/// separated by whitespace so they can be re-merged greedily later.
#[allow(clippy::too_many_lines)] // one function per en.py function, kept intact
fn retokenize(tokens: &[MToken]) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut currency: Option<char> = None;
    let n_tokens = tokens.len();
    for (i, token) in tokens.iter().enumerate() {
        let mut tks: Vec<MToken> = if token.alias.is_none() && token.phonemes.is_none() {
            subtokenize(&token.text)
                .into_iter()
                .map(|t| MToken {
                    text: t,
                    whitespace: String::new(),
                    phonemes: None,
                    is_head: true,
                    alias: None,
                    prespace: false,
                    rating: None,
                    currency: None,
                    ..token.clone()
                })
                .collect()
        } else {
            vec![token.clone()]
        };
        if let Some(last) = tks.last_mut() {
            last.whitespace.clone_from(&token.whitespace);
        }
        let n_tks = tks.len();
        for (j, tk) in tks.into_iter().enumerate() {
            let mut tk = tk;
            let single_char = tk.text.chars().count() == 1;
            let first_char = tk.text.chars().next();
            if tk.alias.is_some() || tk.phonemes.is_some() {
                // resolved already
            } else if tk.tag == "$"
                && single_char
                && first_char.is_some_and(|c| currency_units(c).is_some())
            {
                currency = first_char;
                tk.phonemes = Some(String::new());
                tk.rating = Some(4);
            } else if tk.tag == ":" && (tk.text == "-" || tk.text == "–") {
                tk.phonemes = Some("—".to_string());
                tk.rating = Some(3);
            } else if PUNCT_TAGS.contains(&tk.tag.as_str())
                && !tk
                    .text
                    .chars()
                    .all(|c| c.to_lowercase().all(|l| l.is_ascii_lowercase()))
            {
                tk.phonemes = Some(match punct_tag_phonemes(&tk.tag) {
                    Some(p) => p.to_string(),
                    None => tk.text.chars().filter(|c| PUNCTS.contains(*c)).collect(),
                });
                tk.rating = Some(4);
            } else if let Some(cur) = currency {
                if tk.tag != "CD" {
                    currency = None;
                } else if j + 1 == n_tks && (i + 1 == n_tokens || tokens[i + 1].tag != "CD") {
                    tk.currency = Some(cur);
                }
            } else if j > 0 && j < n_tks - 1 && tk.text == "2" {
                // "2" wedged between letters is "to" (e.g. "B2B").
                // The neighbours are subtokens of the same source token, so
                // re-derive them cheaply from the original text.
                let subs = subtokenize(&token.text);
                let prev_last = subs[j - 1].chars().last();
                let next_first = subs[j + 1].chars().next();
                if prev_last.is_some_and(char::is_alphabetic)
                    && next_first.is_some_and(char::is_alphabetic)
                {
                    tk.alias = Some("to".to_string());
                }
            }
            if tk.alias.is_some() || tk.phonemes.is_some() {
                words.push(Word::One(tk));
            } else if let Some(Word::Group(group)) = words.last_mut() {
                if group.last().is_some_and(|g| g.whitespace.is_empty()) {
                    tk.is_head = false;
                    group.push(tk);
                } else if tk.whitespace.is_empty() {
                    words.push(Word::Group(vec![tk]));
                } else {
                    words.push(Word::One(tk));
                }
            } else if tk.whitespace.is_empty() {
                words.push(Word::Group(vec![tk]));
            } else {
                words.push(Word::One(tk));
            }
        }
    }
    words
        .into_iter()
        .map(|w| match w {
            Word::Group(g) if g.len() == 1 => Word::One(g.into_iter().next().expect("len checked")),
            other => other,
        })
        .collect()
}

fn token_context(ctx: TokenContext, ps: Option<&str>, token: &MToken) -> TokenContext {
    let mut vowel = ctx.future_vowel;
    if let Some(ps) = ps
        && !ps.is_empty()
    {
        {
            for c in ps.chars() {
                let in_vowels = VOWELS.contains(c);
                let in_consonants = CONSONANTS.contains(c);
                let in_puncts = NON_QUOTE_PUNCTS.contains(c);
                if in_vowels || in_consonants || in_puncts {
                    vowel = if in_puncts { None } else { Some(in_vowels) };
                    break;
                }
            }
        }
    }
    let future_to = token.text == "to"
        || token.text == "To"
        || (token.text == "TO" && (token.tag == "TO" || token.tag == "IN"));
    TokenContext {
        future_vowel: vowel,
        future_to,
    }
}

/// misaki's `resolve_tokens`: inside a whitespace-free compound, give
/// trailing punctuation its phonemes, blank out junk, and demote the
/// stress of the weaker half of the subwords.
fn resolve_tokens(tokens: &mut [MToken]) {
    let mut text = String::new();
    for tk in &tokens[..tokens.len() - 1] {
        text.push_str(&tk.text);
        text.push_str(&tk.whitespace);
    }
    text.push_str(&tokens[tokens.len() - 1].text);
    let classes: std::collections::HashSet<u8> = text
        .chars()
        .filter(|c| !SUBTOKEN_JUNKS.contains(*c))
        .map(|c| {
            if c.is_alphabetic() {
                0
            } else if c.is_ascii_digit() {
                1
            } else {
                2
            }
        })
        .collect();
    let prespace = text.contains(' ') || text.contains('/') || classes.len() > 1;
    let n = tokens.len();
    for (i, tk) in tokens.iter_mut().enumerate() {
        if tk.phonemes.is_none() {
            if i == n - 1
                && tk.text.chars().count() == 1
                && tk
                    .text
                    .chars()
                    .next()
                    .is_some_and(|c| NON_QUOTE_PUNCTS.contains(c))
            {
                tk.phonemes = Some(tk.text.clone());
                tk.rating = Some(3);
            } else if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                tk.phonemes = Some(String::new());
                tk.rating = Some(3);
            }
        } else if i > 0 {
            tk.prespace = prespace;
        }
    }
    if prespace {
        return;
    }
    let indices: Vec<(bool, usize, usize)> = tokens
        .iter()
        .enumerate()
        .filter(|(_, tk)| tk.phonemes.as_deref().is_some_and(|p| !p.is_empty()))
        .map(|(i, tk)| {
            let ps = tk.phonemes.as_deref().expect("filtered");
            (ps.contains(PRIMARY_STRESS), stress_weight(ps), i)
        })
        .collect();
    if indices.len() == 2 && tokens[indices[0].2].text.chars().count() == 1 {
        let i = indices[1].2;
        let ps = tokens[i].phonemes.as_deref().expect("has phonemes");
        tokens[i].phonemes = Some(apply_stress(ps, Some(-0.5)));
        return;
    }
    let primaries = indices.iter().filter(|(p, _, _)| *p).count();
    if indices.len() < 2 || primaries <= indices.len().div_ceil(2) {
        return;
    }
    let mut sorted = indices;
    sorted.sort_unstable();
    let take = sorted.len() / 2;
    for (_, _, i) in &sorted[..take] {
        let ps = tokens[*i].phonemes.as_deref().expect("has phonemes");
        tokens[*i].phonemes = Some(apply_stress(ps, Some(-0.5)));
    }
}

/// A fallback phonemizes a token the lexicon could not (misaki uses espeak).
pub type Fallback<'a> = &'a dyn Fn(&MToken) -> PsRating;

/// The main pipeline, following `G2P.__call__` after spaCy: fold, split into
/// words, walk right-to-left resolving each against the lexicon (so "the"
/// and "to" can see whether a vowel follows), then stitch phonemes back
/// together with the original whitespace.
#[allow(clippy::too_many_lines)] // G2P.__call__, ported whole so it reads against en.py
pub fn phonemize_tokens(
    tokens: Vec<MToken>,
    lexicon: &Lexicon,
    fallback: Option<Fallback>,
) -> String {
    // fold_left: merge feature-continuation tokens into their head.
    let mut folded: Vec<MToken> = Vec::with_capacity(tokens.len());
    for tk in tokens {
        if !tk.is_head
            && let Some(prev) = folded.pop()
        {
            folded.push(merge_tokens(&[prev, tk], Some("")));
            continue;
        }
        folded.push(tk);
    }
    let mut words = retokenize(&folded);

    let mut ctx = TokenContext::default();
    for word in words.iter_mut().rev() {
        match word {
            Word::One(tk) => {
                if tk.phonemes.is_none() {
                    let (ps, rating) = lexicon.call_token(tk, ctx);
                    tk.phonemes = ps;
                    tk.rating = rating;
                }
                if tk.phonemes.is_none()
                    && let Some(fallback) = fallback
                {
                    let (ps, rating) = fallback(tk);
                    tk.phonemes = ps;
                    tk.rating = rating;
                }
                ctx = token_context(ctx, tk.phonemes.as_deref(), tk);
            }
            Word::Group(w) => {
                let (mut left, mut right) = (0usize, w.len());
                let mut should_fallback = false;
                while left < right {
                    let merged = if w[left..right]
                        .iter()
                        .any(|t| t.alias.is_some() || t.phonemes.is_some())
                    {
                        None
                    } else {
                        Some(merge_tokens(&w[left..right], None))
                    };
                    let (ps, rating) = match &merged {
                        None => (None, None),
                        Some(tk) => lexicon.call_token(tk, ctx),
                    };
                    if let Some(ps) = ps {
                        ctx = token_context(
                            ctx,
                            Some(&ps),
                            merged.as_ref().expect("ps implies merged"),
                        );
                        w[left].phonemes = Some(ps);
                        w[left].rating = rating;
                        for x in &mut w[left + 1..right] {
                            x.phonemes = Some(String::new());
                        }
                        right = left;
                        left = 0;
                    } else if left + 1 < right {
                        left += 1;
                    } else {
                        right -= 1;
                        let tk = &mut w[right];
                        if tk.phonemes.is_none() {
                            if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                                tk.phonemes = Some(String::new());
                                tk.rating = Some(3);
                            } else if fallback.is_some() {
                                should_fallback = true;
                                break;
                            }
                        }
                        left = 0;
                    }
                }
                if should_fallback {
                    let merged = merge_tokens(w, None);
                    let (ps, rating) = fallback.expect("checked")(&merged);
                    w[0].phonemes = ps;
                    w[0].rating = rating;
                    for x in &mut w[1..] {
                        x.phonemes = Some(String::new());
                    }
                } else {
                    resolve_tokens(w);
                }
            }
        }
    }

    let final_tokens: Vec<MToken> = words
        .into_iter()
        .map(|w| match w {
            Word::One(tk) => tk,
            Word::Group(g) => merge_tokens(&g, Some("")),
        })
        .collect();
    let mut result = String::new();
    for tk in &final_tokens {
        // Kokoro v1.0 vocab: the flap is `T`, the glottal stop is plain `t`.
        let ps = tk
            .phonemes
            .as_deref()
            .unwrap_or("")
            .replace('ɾ', "T")
            .replace('ʔ', "t");
        result.push_str(&ps);
        result.push_str(&tk.whitespace);
    }
    result
}

/// English G2P from raw text: tokenize like spaCy, tag with the perceptron,
/// then run the misaki pipeline with the bundled-espeak fallback.
pub fn phonemize_text(text: &str, british: bool) -> String {
    // G2P.preprocess lstrips; its [text](feature) link syntax is not
    // supported here — daemon input is plain text.
    let text = text.trim_start();
    let pairs = crate::en_tokenizer::Tokenizer::shared().tokenize(text);
    let words: Vec<String> = pairs.iter().map(|(w, _)| w.clone()).collect();
    let tags = crate::en_tagger::Tagger::shared().tag(&words);
    let tokens: Vec<MToken> = pairs
        .iter()
        .zip(&tags)
        .map(|((word, ws), tag)| MToken::new(word, tag, ws))
        .collect();
    let fallback = move |tk: &MToken| espeak_fallback(&tk.text, british);
    phonemize_tokens(tokens, Lexicon::shared(british), Some(&fallback))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spacy_tokens() -> HashMap<String, Vec<MToken>> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/en-tokens.tsv");
        let data = std::fs::read_to_string(path).expect("token fixture exists");
        data.lines()
            .map(|line| {
                let (text, json) = line.split_once('\t').expect("two columns");
                let raw: Vec<(String, String, String)> =
                    serde_json::from_str(json).expect("token json parses");
                let tokens = raw
                    .into_iter()
                    .map(|(text, tag, ws)| MToken::new(&text, &tag, &ws))
                    .collect();
                (text.to_string(), tokens)
            })
            .collect()
    }

    /// End to end from raw text: our tokenizer and our perceptron tagger,
    /// not spaCy's. The perceptron will not match spaCy tag for tag, so
    /// this asserts a divergence *budget*; every divergence is printed.
    #[test]
    fn end_to_end_matches_the_reference_within_budget() {
        let golden = concat!(env!("CARGO_MANIFEST_DIR"), "/data/en-golden.tsv");
        let golden = std::fs::read_to_string(golden).expect("golden fixture exists");
        let mut diverged: Vec<String> = Vec::new();
        let mut total = 0usize;
        for line in golden.lines() {
            let mut cols = line.split('\t');
            let (lang, group, text, expected) = (
                cols.next().expect("lang"),
                cols.next().expect("group"),
                cols.next().expect("text"),
                cols.next().expect("phonemes"),
            );
            total += 1;
            let actual = phonemize_text(text, lang == "en-gb");
            if actual != expected {
                diverged.push(format!(
                    "[{lang}/{group}] {text:?}\n  expected: {expected}\n  actual  : {actual}"
                ));
            }
        }
        let budget = total / 100; // 1%; measured 14/1666, all POS-tag ties on adversarial heteronym sentences
        for d in &diverged {
            println!("{d}");
        }
        assert!(
            diverged.len() <= budget,
            "{} of {total} rows diverge end-to-end (budget {budget}):\n{}",
            diverged.len(),
            diverged.join("\n")
        );
        println!(
            "end-to-end: {}/{total} rows match ({} within budget {budget})",
            total - diverged.len(),
            diverged.len(),
        );
    }

    /// Replays misaki 0.9.4's own answers for every golden row that does not
    /// need the espeak fallback, feeding the pipeline spaCy's tokens.
    #[test]
    fn matches_the_reference_g2p() {
        let golden = concat!(env!("CARGO_MANIFEST_DIR"), "/data/en-golden.tsv");
        let golden = std::fs::read_to_string(golden).expect("golden fixture exists");
        let tokens = spacy_tokens();
        let mut mismatches: Vec<String> = Vec::new();
        let mut counts: HashMap<&str, (usize, usize)> = HashMap::new();
        for line in golden.lines() {
            let mut cols = line.split('\t');
            let (lang, group, text, expected) = (
                cols.next().expect("lang"),
                cols.next().expect("group"),
                cols.next().expect("text"),
                cols.next().expect("phonemes"),
            );
            let british = lang == "en-gb";
            let lexicon = Lexicon::shared(british);
            let toks = tokens
                .get(text)
                .expect("token fixture covers corpus")
                .clone();
            let fallback = move |tk: &MToken| espeak_fallback(&tk.text, british);
            let actual = phonemize_tokens(toks, lexicon, Some(&fallback));
            let entry = counts.entry(group).or_insert((0, 0));
            entry.1 += 1;
            if actual == expected {
                entry.0 += 1;
            } else if mismatches.len() < 60 {
                mismatches.push(format!(
                    "[{lang}/{group}] {text:?}\n  expected: {expected}\n  actual  : {actual}"
                ));
            }
        }
        let mut summary: Vec<_> = counts.iter().collect();
        summary.sort();
        let report: Vec<String> = summary
            .iter()
            .map(|(g, (ok, total))| format!("{g}: {ok}/{total}"))
            .collect();
        let total_ok: usize = counts.values().map(|(ok, _)| ok).sum();
        let total: usize = counts.values().map(|(_, t)| t).sum();
        assert_eq!(
            total_ok,
            total,
            "{} of {} golden rows diverge\n{}\n\nfirst mismatches:\n{}",
            total - total_ok,
            total,
            report.join("  "),
            mismatches.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// The espeak fallback, for words the lexicon cannot answer.

/// misaki's `EspeakFallback.E2M`, in the exact order Python applies it:
/// sorted by key length, longest first, insertion order within a length.
/// Sequential global replacements, so order is behaviour.
const E2M: [(&str, &str); 20] = [
    ("ʔˌn\u{329}", "ʔn"),
    ("ʔn\u{329}", "ʔn"),
    ("a^ɪ", "I"),
    ("a^ʊ", "W"),
    ("d^ʒ", "ʤ"),
    ("e^ɪ", "A"),
    ("t^ʃ", "ʧ"),
    ("ɔ^ɪ", "Y"),
    ("ə^l", "ᵊl"),
    ("ʲo", "jo"),
    ("ʲə", "jə"),
    ("e", "A"),
    ("ʲ", ""),
    ("ɚ", "əɹ"),
    ("r", "ɹ"),
    ("x", "k"),
    ("ç", "k"),
    ("ɐ", "ə"),
    ("ɬ", "l"),
    ("\u{303}", ""),
];

/// The punctuation phonemizer hides from espeak and restores afterwards.
const PHONEMIZER_MARKS: &str = ";:,.!?¡¿—…\"«»“”(){}[]";

#[derive(PartialEq, Eq, Clone, Copy)]
enum MarkPos {
    Begin,
    End,
    Inner,
    Alone,
}

/// phonemizer's `Punctuation._preserve_line`: split a line on runs of
/// punctuation (with surrounding spaces attached to the mark), remembering
/// where each run sat so it can be stitched back.
fn preserve_marks(line: &str) -> (Vec<String>, Vec<(String, MarkPos)>) {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        let class = regex::escape(PHONEMIZER_MARKS);
        regex::Regex::new(&format!(r"(\s*[{class}]+\s*)+")).expect("marks regex compiles")
    });
    let matches: Vec<(usize, usize)> = re.find_iter(line).map(|m| (m.start(), m.end())).collect();
    if matches.is_empty() {
        return (vec![line.to_string()], Vec::new());
    }
    if matches.len() == 1 && matches[0] == (0, line.len()) {
        return (Vec::new(), vec![(line.to_string(), MarkPos::Alone)]);
    }
    let mut chunks = Vec::new();
    let mut marks = Vec::new();
    let mut last = 0;
    for (i, &(start, end)) in matches.iter().enumerate() {
        let pos = if i == 0 && start == 0 {
            MarkPos::Begin
        } else if i == matches.len() - 1 && end == line.len() {
            MarkPos::End
        } else {
            MarkPos::Inner
        };
        chunks.push(line[last..start].to_string());
        marks.push((line[start..end].to_string(), pos));
        last = end;
    }
    chunks.push(line[last..].to_string());
    (chunks, marks)
}

/// One chunk through espeak in phonemizer's tie mode, normalized the way
/// phonemizer normalizes it: whitespace collapsed, a trailing word
/// separator, U+0361 swapped for the `^` tie misaki's rules expect.
fn espeak_chunk(chunk: &str, voice: &str) -> Option<String> {
    if chunk.trim().is_empty() {
        return Some(String::new());
    }
    let ps = crate::espeak::phonemize_tied(chunk, voice).ok()?;
    let words: Vec<&str> = ps.split_whitespace().collect();
    if words.is_empty() {
        return Some(String::new());
    }
    let mut line = words.join(" ");
    line.push(' ');
    Some(line.replace('\u{361}', "^"))
}

/// phonemizer's `Punctuation.restore` for a single input line.
fn restore_marks(mut chunks: Vec<String>, marks: Vec<(String, MarkPos)>) -> String {
    let mut out = String::new();
    let mut marks = marks.into_iter().peekable();
    loop {
        if chunks.is_empty() {
            for (mark, _) in marks {
                out.push_str(&mark);
            }
            return out;
        }
        let Some((mark, pos)) = marks.next() else {
            for line in chunks {
                out.push_str(&line);
                if !line.ends_with(' ') && !line.is_empty() {
                    out.push(' ');
                }
            }
            return out;
        };
        if let Some(first) = chunks.first_mut()
            && first.ends_with(' ')
        {
            first.pop();
        }
        match pos {
            MarkPos::Begin => {
                chunks[0] = format!("{mark}{}", chunks[0]);
            }
            MarkPos::End => {
                let first = chunks.remove(0);
                out.push_str(&first);
                out.push_str(&mark);
                if !mark.ends_with(' ') {
                    out.push(' ');
                }
            }
            MarkPos::Alone => {
                out.push_str(&mark);
            }
            MarkPos::Inner => {
                let second = if chunks.len() > 1 {
                    chunks.remove(1)
                } else {
                    String::new()
                };
                chunks[0] = format!("{}{mark}{second}", chunks[0]);
            }
        }
    }
}

/// misaki's `EspeakFallback`: the bundled espeak-ng, punctuation preserved,
/// output rewritten into Kokoro's vocabulary. Rating 2 — the lowest, which
/// is why the lexicon always wins when it has an answer.
pub fn espeak_fallback(text: &str, british: bool) -> PsRating {
    // espeak's British voice is the file `gmw/en`: its *name* is "en" and
    // "en-gb" is only its language code, which `SetVoiceByName` does not
    // resolve (the CLI falls back to a by-properties search; we just use
    // the name, as `phonemes.rs` does for the `b*` Kokoro voices).
    let voice = if british { "en" } else { "en-us" };
    let (chunks, marks) = preserve_marks(text);
    let phonemized: Option<Vec<String>> = chunks.iter().map(|c| espeak_chunk(c, voice)).collect();
    let Some(phonemized) = phonemized else {
        return (None, None);
    };
    let mut ps = restore_marks(phonemized, marks);
    ps = ps.trim().to_string();
    if ps.is_empty() {
        return (None, None);
    }
    for (old, new) in E2M {
        ps = ps.replace(old, new);
    }
    // `re.sub(r'(\S)̩', r'ᵊ\1', ps)` then drop any leftover U+0329.
    {
        static RE: OnceLock<regex::Regex> = OnceLock::new();
        let re = RE.get_or_init(|| regex::Regex::new(r"(\S)\u{329}").expect("compiles"));
        ps = re.replace_all(&ps, "ᵊ$1").into_owned();
        ps = ps.replace('\u{329}', "");
    }
    if british {
        ps = ps.replace("e^ə", "ɛː");
        ps = ps.replace("iə", "ɪə");
        ps = ps.replace("ə^ʊ", "Q");
    } else {
        ps = ps.replace("o^ʊ", "O");
        ps = ps.replace("ɜːɹ", "ɜɹ");
        ps = ps.replace("ɜː", "ɜɹ");
        ps = ps.replace("ɪə", "iə");
        ps = ps.replace('ː', "");
    }
    ps = ps.replace('o', "ɔ");
    ps = ps.replace('ɾ', "T").replace('ʔ', "t");
    (Some(ps.replace('^', "")), Some(2))
}
