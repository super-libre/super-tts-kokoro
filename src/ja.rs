// SPDX-License-Identifier: GPL-3.0-only
//! Japanese text to phonemes.
//!
//! Like Mandarin, Japanese cannot go through `espeak-ng` here, and like
//! Mandarin the target is not "correct Japanese IPA" in the abstract but
//! *what `misaki` emits*, because that is what Kokoro was trained on.
//!
//! # Which misaki
//!
//! `misaki.ja.JAG2P` has two paths and defaults to **cutlet** (`fugashi` +
//! `UniDic`), not to `pyopenjtalk`. The obvious guess is the wrong one, and they
//! do not agree:
//!
//! | | こんにちは、世界。 |
//! |---|---|
//! | cutlet (default) | `koɲɲiʨiβa, sekai.` |
//! | pyopenjtalk | `koɴniʨiwa, sekai.__-------jj^^___j` |
//!
//! The pyopenjtalk branch appends its pitch string to the phonemes and emits
//! `ᶄ`, which has no token in Kokoro's vocabulary. This module follows cutlet.
//!
//! # The dictionary
//!
//! Japanese needs a morphological analyser, and its dictionary is far too large
//! to embed the way the espeak and Mandarin data are: `UniDic` unpacks to 191 MB.
//! It is declared in `backend.toml` and fetched by the daemon like the model
//! weights, then unpacked once and memory-mapped, which is why loading it costs
//! milliseconds rather than the seconds a 191 MB read would.
//!
//! It is also unpacked *lazily*, on the first Japanese utterance. A user who
//! only ever speaks English or Mandarin never pays for it.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail};

/// Kana (and two-kana digraphs) to IPA, plus the punctuation cutlet rewrites.
const KANA_GZ: &[u8] = include_bytes!("../data/ja-kana.tsv.gz");
/// Katakana Phonetic Extensions to their full-size forms (ㇰ -> ク).
const KATA_EXT_GZ: &[u8] = include_bytes!("../data/ja-kata-ext.tsv.gz");
/// Words cutlet re-joins adjacent tokens into.
const WORDS_GZ: &[u8] = include_bytes!("../data/ja-words.tsv.gz");

/// Where the daemon puts the dictionary archive, relative to the backend dir.
pub const DICTIONARY_ARCHIVE: &str = "dictionaries/lindera-unidic-6.0.0.zip";
/// The directory the archive expands to.
const DICTIONARY_DIR: &str = "lindera-unidic";

/// `UniDic` field index of `pron`. `lindera` exposes no field-name map for this
/// schema, so the position is what identifies it. The layout is
/// `pos1,pos2,pos3,pos4,cType,cForm,lForm,lemma,orth,pron,...`.
const PRON: usize = 9;

/// Small kana that attach to the preceding mora rather than standing alone.
const SUTEGANA: &str = "ゃゅょぁぃぅぇぉ";
/// Iteration marks, which repeat the previous kana.
const ODORI: &str = "〃々ゝゞヽ";

/// `MeCab` character classes, as `char.def` numbers them. Only the ones cutlet
/// distinguishes are named; the rest fall through to [`OTHER`].
const SYMBOL: u8 = 3;
const HIRAGANA: u8 = 6;
const KATAKANA: u8 = 7;
const OTHER: u8 = 0;

struct Tables {
    kana: HashMap<String, String>,
    kata_ext: HashMap<char, char>,
    words: HashSet<String>,
    /// Longest entry in `words`, in characters, bounding the join scan.
    longest_word: usize,
}

fn gunzip(bytes: &[u8]) -> Result<String> {
    let mut out = String::new();
    flate2::read::GzDecoder::new(bytes)
        .read_to_string(&mut out)
        .context("decompressing an embedded Japanese table")?;
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
    let mut kana = HashMap::new();
    for line in gunzip(KANA_GZ)?.lines() {
        if let Some((k, v)) = line.split_once('\t') {
            kana.insert(k.to_string(), v.to_string());
        }
    }
    let mut kata_ext = HashMap::new();
    for line in gunzip(KATA_EXT_GZ)?.lines() {
        let Some((k, v)) = line.split_once('\t') else {
            continue;
        };
        if let (Some(a), Some(b)) = (k.chars().next(), v.chars().next()) {
            kata_ext.insert(a, b);
        }
    }
    let mut words = HashSet::new();
    for line in gunzip(WORDS_GZ)?.lines() {
        let word = line.split_once('\t').map_or(line, |(k, _)| k);
        if !word.is_empty() {
            words.insert(word.to_string());
        }
    }
    let longest_word = words.iter().map(|w| w.chars().count()).max().unwrap_or(1);
    Ok(Tables {
        kana,
        kata_ext,
        words,
        longest_word,
    })
}

// ---------------------------------------------------------------------------
// The dictionary
// ---------------------------------------------------------------------------

/// The backend directory, recorded at load so the phonemizer can find the
/// dictionary without threading a path through every call.
static BACKEND_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Record where the backend's files live. Called at model load.
///
/// A directory without the dictionary is ignored rather than recorded. It
/// carries no information — Japanese cannot load from it either way — and
/// forgetting a directory that *did* have one in order to remember a directory
/// that does not is strictly worse. This also makes the order of calls
/// irrelevant, which matters because the analyzer below is built once per
/// process and so only ever honours one directory anyway.
pub fn set_backend_dir(dir: &Path) {
    if !dir.join(DICTIONARY_ARCHIVE).is_file() {
        log::debug!(
            "no Japanese dictionary under {}; leaving any recorded directory alone",
            dir.display()
        );
        return;
    }
    *BACKEND_DIR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(dir.to_path_buf());
}

struct Analyzer {
    segmenter: lindera::segmenter::Segmenter,
}

/// Serializes the segmenter, which is not `Sync`.
static ANALYZER: OnceLock<Result<Mutex<Analyzer>, String>> = OnceLock::new();

fn analyzer() -> Result<&'static Mutex<Analyzer>> {
    ANALYZER
        .get_or_init(|| load_analyzer().map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{e}"))
}

fn load_analyzer() -> Result<Mutex<Analyzer>> {
    let backend_dir = BACKEND_DIR
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .context("the backend directory was never recorded; Japanese cannot load")?;

    let archive = backend_dir.join(DICTIONARY_ARCHIVE);
    if !archive.is_file() {
        bail!(
            "the Japanese dictionary is missing from {}. It is declared in \
             backend.toml and fetched by the daemon; a backend directory \
             populated before Japanese was added will not have it.",
            archive.display()
        );
    }
    let dir = unpack_dictionary(&archive)?;

    let uri = format!("file://{}", dir.display());
    let dictionary = lindera::dictionary::load_dictionary(&uri)
        .map_err(|e| anyhow::anyhow!("loading the UniDic dictionary from {uri}: {e}"))?;
    let segmenter =
        lindera::segmenter::Segmenter::new(lindera::mode::Mode::Normal, dictionary, None);
    log::info!("Japanese dictionary ready ({})", dir.display());
    Ok(Mutex::new(Analyzer { segmenter }))
}

/// Marker written once extraction finishes, so an abandoned unpack is not
/// mistaken for a complete one. See `crate::espeak` for the same pattern and
/// why a file from the archive itself cannot play this role.
const COMPLETE_MARKER: &str = ".complete";

/// Expand the dictionary archive into a temporary directory, once.
fn unpack_dictionary(archive: &Path) -> Result<PathBuf> {
    let root = std::env::temp_dir().join("super-tts-unidic-6.0.0");
    let dir = root.join(DICTIONARY_DIR);
    if root.join(COMPLETE_MARKER).is_file() {
        return Ok(dir);
    }

    let scratch = root.with_extension(format!("tmp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).with_context(|| format!("creating {}", scratch.display()))?;

    let file =
        std::fs::File::open(archive).with_context(|| format!("opening {}", archive.display()))?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("reading {}", archive.display()))?;
    zip.extract(&scratch).with_context(|| {
        format!(
            "extracting {} into {}",
            archive.display(),
            scratch.display()
        )
    })?;
    std::fs::write(scratch.join(COMPLETE_MARKER), [])
        .with_context(|| format!("marking {} complete", scratch.display()))?;

    if root.exists() && !root.join(COMPLETE_MARKER).is_file() {
        let _ = std::fs::remove_dir_all(&root);
    }
    match std::fs::rename(&scratch, &root) {
        Ok(()) => {}
        Err(_) if root.join(COMPLETE_MARKER).is_file() => {
            let _ = std::fs::remove_dir_all(&scratch);
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&scratch);
            return Err(e).with_context(|| format!("installing {}", root.display()));
        }
    }
    Ok(dir)
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Classify a character the way `MeCab`'s `char.def` does.
///
/// Only used for tokens the dictionary did not recognise: cutlet folds every
/// known word, and all katakana, into [`HIRAGANA`], so this decides the fate of
/// unknown text alone.
fn char_type(c: char) -> u8 {
    match c {
        '\u{3041}'..='\u{309F}' => HIRAGANA,
        '\u{30A1}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}' | '\u{FF66}'..='\u{FF9F}' => KATAKANA,
        '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' | '々' | '〆' | '〇' => 2, // KANJI
        '0'..='9' | '\u{FF10}'..='\u{FF19}' => 4,                                    // NUMERIC
        'A'..='Z' | 'a'..='z' => 5,                                                  // ALPHA
        c if c.is_whitespace() => 1,                                                 // SPACE
        c if c.is_ascii_punctuation() => SYMBOL,
        '\u{3000}'..='\u{303F}' | '\u{FF01}'..='\u{FF0F}' | '\u{FF1A}'..='\u{FF20}' => SYMBOL,
        _ => OTHER,
    }
}

/// Katakana to hiragana, which is a fixed offset for the main block.
fn kata_to_hira(text: &str) -> String {
    text.chars()
        .map(|c| {
            if ('\u{30A1}'..='\u{30F6}').contains(&c) {
                char::from_u32(c as u32 - 0x60).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// Normalize the variations that are meaningless for pronunciation.
///
/// `misaki` reaches this through `unicodedata.normalize('`NFKC`')` followed by
/// two `mojimoji` calls — full-width Latin to half-width, then half-width
/// katakana to full-width. `NFKC` already performs both of those mappings, so
/// only `NFKC` is applied here.
fn normalize(t: &Tables, text: &str) -> String {
    // A wave dash before a digit is a range, and is read as "from".
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if (c == '〜' || c == '～') && chars.get(i + 1).is_some_and(char::is_ascii_digit) {
            out.push_str("から");
        } else if let Some(&full) = t.kata_ext.get(&c) {
            out.push(full);
        } else {
            out.push(c);
        }
    }
    let normalized: String =
        unicode_normalization::UnicodeNormalization::nfkc(out.as_str()).collect();
    spell_out_digits(&normalized)
}

/// Replace every run of digits with its kana reading.
///
/// The leading space is `misaki`'s, and it matters: it keeps the number from
/// being glued to the preceding word and re-analysed as one token.
fn spell_out_digits(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut digits = String::new();
    for c in text.chars() {
        if c.is_ascii_digit() {
            digits.push(c);
            continue;
        }
        if !digits.is_empty() {
            flush_digits(&mut out, &mut digits);
        }
        out.push(c);
    }
    if !digits.is_empty() {
        flush_digits(&mut out, &mut digits);
    }
    out
}

fn flush_digits(out: &mut String, digits: &mut String) {
    if let Some(kana) = crate::ja_numbers::to_kana(digits) {
        out.push(' ');
        out.push_str(&kana);
    } else {
        out.push_str(digits);
    }
    digits.clear();
}

// ---------------------------------------------------------------------------
// cutlet
// ---------------------------------------------------------------------------

struct Word {
    surface: String,
    hira: String,
    char_type: u8,
}

struct Tok {
    surface: String,
    space: bool,
}

/// Convert Japanese `text` to `misaki`-compatible IPA.
///
/// # Errors
/// Returns an error if the embedded tables or the `UniDic` dictionary cannot be
/// loaded.
pub fn phonemize(text: &str) -> Result<String> {
    if text.trim().is_empty() {
        return Ok(String::new());
    }
    let t = tables()?;
    let text = normalize(t, text);

    let words = {
        let analyzer = analyzer()?;
        let guard = analyzer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut tokens = guard
            .segmenter
            .segment(Cow::Owned(text))
            .map_err(|e| anyhow::anyhow!("segmenting Japanese: {e}"))?;
        let mut words = Vec::new();
        for tok in &mut tokens {
            let unknown = tok.word_id.is_unknown();
            let surface = tok.surface.to_string();
            let pron = tok
                .get_detail(PRON)
                .filter(|p| !p.is_empty() && *p != "*")
                .map_or_else(|| surface.clone(), str::to_string);
            let ct = surface.chars().next().map_or(OTHER, char_type);
            words.push(Word {
                hira: kata_to_hira(&pron),
                // A known word, or any katakana, is treated as ordinary kana.
                char_type: if ct == KATAKANA || !unknown {
                    HIRAGANA
                } else {
                    ct
                },
                surface,
            });
        }
        words
    };

    Ok(assemble(t, &join_known_words(t, &words)))
}

/// Re-join adjacent tokens that spell a known word.
///
/// The tokenizer splits 日本語 into 日本 + 語; pronouncing those separately
/// would put a word boundary inside a word. cutlet scans runs of one character
/// class for the longest join present in its word list, which is what this
/// reproduces.
fn join_known_words(t: &Tables, words: &[Word]) -> Vec<Word> {
    let mut out: Vec<Word> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        // The run of tokens sharing this token's character class.
        let run_end = words[i + 1..]
            .iter()
            .position(|w| w.char_type != words[i].char_type)
            .map_or(words.len(), |p| i + 1 + p);

        let mut joined = None;
        let limit = run_end.min(i + t.longest_word);
        for j in (i + 1..=limit).rev() {
            if j == i + 1 {
                continue;
            }
            let surface: String = words[i..j].iter().map(|w| w.surface.as_str()).collect();
            if t.words.contains(&surface) {
                joined = Some((j, surface));
                break;
            }
        }
        if let Some((j, surface)) = joined {
            let hira: String = words[i..j].iter().map(|w| w.hira.as_str()).collect();
            out.push(Word {
                surface,
                hira,
                char_type: words[i].char_type,
            });
            i = j;
        } else {
            out.push(words[i].clone_word());
            i += 1;
        }
    }
    out
}

impl Word {
    fn clone_word(&self) -> Word {
        Word {
            surface: self.surface.clone(),
            hira: self.hira.clone(),
            char_type: self.char_type,
        }
    }
}

/// Turn words into the final phoneme string, applying cutlet's spacing rules.
fn assemble(t: &Tables, words: &[Word]) -> String {
    let mut out: Vec<Tok> = Vec::with_capacity(words.len());
    for word in words {
        let roma = romaji_word(t, word);
        let mut tok = Tok {
            surface: roma.clone(),
            space: false,
        };
        let opening =
            matches!(word.surface.as_str(), "「" | "『" | "«") || roma == "(" || roma == "[";
        let closing = matches!(word.surface.as_str(), "」" | "』" | "»")
            || matches!(roma.as_str(), "]" | ")" | "." | "," | "?" | "!" | ":");
        if opening {
            if let Some(prev) = out.last_mut() {
                prev.space = true;
            }
        } else if closing {
            if let Some(prev) = out.last_mut() {
                prev.space = false;
            }
            tok.space = true;
        } else {
            // A word standing on its own is followed by a space; the
            // interpunct cutlet maps to a bare space is not doubled.
            tok.space = roma != " ";
        }
        out.push(tok);
    }

    let mut joined = String::new();
    for tok in &out {
        // Any sokuon that survived to here is not pronounceable on its own.
        joined.push_str(&tok.surface.replace('っ', ""));
        if tok.space {
            joined.push(' ');
        }
    }
    finish(&joined)
}

/// Collapse whitespace, rewrite parentheses, and close up the spaces around a
/// sokuon, which is a held consonant rather than a word of its own.
fn finish(text: &str) -> String {
    // Characters after which a space before `ʔ` is kept, and before which a
    // space after `ʔ` is kept.
    const KEEP_BEFORE: &str = "!\",.:;?»—…”";
    const KEEP_AFTER: &str = "\"«“";

    let mut collapsed = String::with_capacity(text.len());
    let mut in_space = false;
    for c in text.trim().chars() {
        if c.is_whitespace() {
            in_space = true;
        } else {
            if in_space {
                collapsed.push(' ');
            }
            in_space = false;
            collapsed.push(match c {
                '(' => '«',
                ')' => '»',
                other => other,
            });
        }
    }

    // `misaki` does this with two lookarounds, which the regex crate has no
    // equivalent for, so it is a scan: drop a space before `ʔ` unless the
    // character before it ends a clause, and drop a space after `ʔ` unless a
    // quotation mark follows.
    let chars: Vec<char> = collapsed.chars().collect();
    let mut out = String::with_capacity(collapsed.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == ' ' {
            let before = i.checked_sub(1).and_then(|j| chars.get(j)).copied();
            let after = chars.get(i + 1).copied();
            if after == Some('ʔ') && !before.is_some_and(|b| KEEP_BEFORE.contains(b)) {
                continue;
            }
            if before == Some('ʔ') && !after.is_some_and(|a| KEEP_AFTER.contains(a)) {
                continue;
            }
        }
        out.push(c);
    }
    out
}

/// Pronounce a single word.
fn romaji_word(t: &Tables, word: &Word) -> String {
    if word.surface.is_ascii() {
        return word.surface.clone();
    }
    if word.char_type == SYMBOL {
        return word
            .surface
            .chars()
            .map(|c| {
                t.kana
                    .get(&c.to_string())
                    .cloned()
                    .unwrap_or_else(|| c.to_string())
            })
            .collect();
    }
    if word.char_type != HIRAGANA {
        // cutlet silently drops what it cannot read rather than guessing.
        return String::new();
    }
    let hira: Vec<char> = word.hira.chars().collect();
    let mut out = String::new();
    for i in 0..hira.len() {
        let prev = i.checked_sub(1).map(|j| hira[j]);
        let next = hira.get(i + 1).copied();
        out.push_str(&single_mapping(t, prev, hira[i], next));
    }
    out
}

/// Map one kana to IPA, given its neighbours.
fn single_mapping(t: &Tables, prev: Option<char>, kk: char, next: Option<char>) -> String {
    let get = |s: &str| t.kana.get(s).cloned();

    if ODORI.contains(kk) {
        // Iteration marks repeat the previous kana, voiced for ゞ/ヾ.
        if kk == 'ゝ' || kk == 'ヽ' {
            return prev.map(|p| p.to_string()).unwrap_or_default();
        }
        if kk == 'ゞ' || kk == 'ヾ' {
            return prev
                .and_then(add_dakuten)
                .and_then(|v| get(&v.to_string()))
                .unwrap_or_default();
        }
        // 々 (kanji) and 〃 (symbols) cannot have their span inferred here.
        return String::new();
    }

    // A digraph is written on its second kana, so the first contributes
    // nothing and the second carries both.
    if let Some(v) = prev.and_then(|p| get(&format!("{p}{kk}"))) {
        return v;
    }
    if let Some(n) = next {
        if t.kana.contains_key(&format!("{kk}{n}")) {
            return String::new();
        }
        if SUTEGANA.contains(n) {
            if kk == 'っ' {
                return String::new();
            }
            // The small kana replaces the base kana's vowel.
            let base = get(&kk.to_string()).unwrap_or_default();
            let small = get(&n.to_string()).unwrap_or_default();
            let mut merged: String = base.chars().collect::<Vec<_>>()
                [..base.chars().count().saturating_sub(1)]
                .iter()
                .collect();
            merged.push_str(&small);
            return merged;
        }
    }
    if SUTEGANA.contains(kk) {
        return String::new();
    }
    if kk == 'ー' {
        return "ː".to_string();
    }
    if kk == 'っ' {
        return "ʔ".to_string();
    }
    if kk == 'ん' {
        // The moraic nasal takes the place of what follows it.
        if let Some(following) = next.and_then(|n| get(&n.to_string())) {
            let first = following.chars().next().unwrap_or(' ');
            if "mpb".contains(first) {
                return "m".to_string();
            }
            if "kɡ".contains(first) {
                return "ŋ".to_string();
            }
            if ["ɲ", "ʨ", "ʥ"].iter().any(|p| following.starts_with(p)) {
                return "ɲ".to_string();
            }
            if "ntdɾz".contains(first) {
                return "n".to_string();
            }
        }
        return "ɴ".to_string();
    }
    get(&kk.to_string()).unwrap_or_default()
}

/// Add a dakuten to a kana, for the voiced iteration mark.
fn add_dakuten(kk: char) -> Option<char> {
    const PLAIN: &str = "かきくけこさしすせそたちつてとはひふへほ";
    const VOICED: &str = "がぎぐげござじずぜぞだぢづでどばびぶべぼ";
    PLAIN
        .chars()
        .position(|c| c == kk)
        .and_then(|i| VOICED.chars().nth(i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tables_load() {
        let t = tables().expect("tables load");
        assert!(t.kana.len() > 180, "kana: {}", t.kana.len());
        assert!(t.words.len() > 140_000, "words: {}", t.words.len());
        assert_eq!(t.kata_ext.len(), 16);
    }

    #[test]
    fn katakana_becomes_hiragana() {
        assert_eq!(kata_to_hira("セカイ"), "せかい");
        assert_eq!(kata_to_hira("コンニチワ"), "こんにちわ");
    }

    #[test]
    fn the_moraic_nasal_assimilates_to_what_follows() {
        let t = tables().expect("tables");
        // ん before k -> ŋ, before p/m/b -> m, otherwise ɴ
        assert_eq!(single_mapping(t, None, 'ん', Some('こ')), "ŋ");
        assert_eq!(single_mapping(t, None, 'ん', Some('ぱ')), "m");
        assert_eq!(single_mapping(t, None, 'ん', None), "ɴ");
    }

    /// Where the dictionary lives when running the tests. It is fetched by the
    /// daemon in production, so it is not in the repository and the tests that
    /// need it say so rather than passing vacuously.
    fn test_backend_dir() -> Option<PathBuf> {
        let dir = PathBuf::from(std::env::var("SUPER_TTS_BACKEND_DIR").ok()?);
        dir.join(DICTIONARY_ARCHIVE).is_file().then_some(dir)
    }

    /// Inputs where this backend and `misaki` disagree because their `UniDic`
    /// builds disagree, not because the port does.
    ///
    /// `misaki` runs against `unidic-lite`; this runs against the
    /// `unidic-mecab-2.1.2` that `lindera` ships. Their lexicons differ in a few
    /// rare entries, and the difference is visible before any phoneme is
    /// produced — the tokenizers return different readings:
    ///
    /// | input | unidic-lite | unidic-mecab-2.1.2 |
    /// |---|---|---|
    /// | 抓れる | ツネレ + ル | ツメレ + ル |
    /// | 観心 | カンシン | カンジン |
    /// | 一、二、三。 | `二、三` is one entry | three tokens |
    ///
    /// They are listed rather than dropped from the fixture so the count
    /// cannot quietly grow: a port regression would add cases here, and the
    /// assertion below refuses that.
    const DICTIONARY_DIVERGENCES: &[&str] = &["一、二、三。", "抓れる", "観心"];

    /// The claim this module rests on: it reproduces `misaki`, the g2p Kokoro
    /// was trained with. `data/ja-golden.tsv` holds misaki's own answers, and
    /// every one must come back identical — anything else is a pronunciation
    /// the model was never taught.
    #[test]
    fn matches_the_reference_g2p() {
        let Some(dir) = test_backend_dir() else {
            eprintln!(
                "skipping: no {DICTIONARY_ARCHIVE} under $SUPER_TTS_BACKEND_DIR. \
                 The Japanese dictionary is fetched by the daemon, so set that \
                 variable to a populated backend directory to run this test."
            );
            return;
        };
        set_backend_dir(&dir);

        let golden = include_str!("../data/ja-golden.tsv");
        let mut checked = 0;
        let mut wrong = Vec::new();
        for line in golden.lines() {
            let Some((text, expected)) = line.split_once('\t') else {
                continue;
            };
            let got = phonemize(text).expect("phonemize");
            checked += 1;
            if got != expected && !DICTIONARY_DIVERGENCES.contains(&text) {
                wrong.push(format!("  {text}\n    want {expected:?}\n    got  {got:?}"));
            }
        }
        assert!(checked > 700, "fixture only had {checked} cases");
        // Under 1% of the fixture. A port regression shows up as new entries
        // here long before it reaches that ceiling.
        assert!(
            DICTIONARY_DIVERGENCES.len() * 100 <= checked,
            "{} known divergences is too many for a {checked}-case fixture",
            DICTIONARY_DIVERGENCES.len()
        );
        assert!(
            wrong.is_empty(),
            "{} of {checked} disagree with misaki:\n{}",
            wrong.len(),
            wrong
                .iter()
                .take(12)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// Digits used to reach the phoneme table, which has no token for them,
    /// and vanish — 価格は99円 was spoken as "the price is ___ yen".
    #[test]
    fn numbers_are_spoken_rather_than_dropped() {
        let Some(dir) = test_backend_dir() else {
            eprintln!("skipping: no Japanese dictionary; see matches_the_reference_g2p");
            return;
        };
        set_backend_dir(&dir);
        let out = phonemize("価格は99円です。").unwrap();
        assert!(
            !out.contains('9'),
            "digits survived into the phonemes: {out:?}"
        );
        // きゅうじゅうきゅう -> kʲɨː ʥɨː kʲɨː
        assert!(out.contains("ʥɨː"), "the じゅう of 99 is missing: {out:?}");
        assert_eq!(
            out.matches("kʲɨː").count(),
            2,
            "99 needs two きゅう: {out:?}"
        );
    }

    #[test]
    fn empty_input_is_empty_output() {
        assert_eq!(phonemize("").unwrap(), "");
        assert_eq!(phonemize("  ").unwrap(), "");
    }
}
