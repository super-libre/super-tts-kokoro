// SPDX-License-Identifier: GPL-3.0-only
//! Text to Kokoro token ids.
//!
//! Kokoro is a `StyleTTS2` model: it consumes IPA phoneme tokens, not graphemes.
//! The reference implementation phonemizes with `misaki`, a Python g2p. This
//! backend uses `espeak-ng`, which is what every other non-Python Kokoro port
//! does — but linked in and carried inside the binary rather than shelled out
//! to, so no system package stands between an install and speech. See
//! [`crate::espeak`].
//!
//! That substitution is safe in a way worth recording: `espeak-ng`'s IPA output
//! for the languages routed to it lands entirely inside Kokoro's 115-symbol
//! vocabulary, so no lossy transliteration table stands between the two. It was
//! verified per language, and [`tests::espeak_output_stays_inside_the_vocab`]
//! keeps it verified.
//!
//! # One phonemizer per language, not one for all of them
//!
//! Which engine phonemizes a language is a property of the language, so
//! [`Language`] names it and [`g2p_for`] resolves it. That indirection is not
//! speculative generality: `espeak-ng` cannot serve Japanese or Mandarin here.
//! Kokoro ships voices for both, and its vocabulary was plainly built for them
//! — tokens 169-173 are the Mandarin tone arrows `↓ → ↗ ↘` that `misaki` emits,
//! and `ʣ ʥ ʦ ʨ` are affricates those two languages need. But `espeak-ng`
//! writes Mandarin with numeric tone marks (`ni2 hao3`), which have no token
//! here, so routing those voices through it would synthesize from a stream of
//! dropped symbols. They need engines that match what Kokoro was trained on,
//! and until those land the voices are withheld rather than mispronounced.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};

/// Longest token sequence the model accepts, excluding the two pad tokens.
///
/// The style vectors are stored as 510 rows indexed by token count, so this is
/// a property of the voice files rather than a tuning choice.
pub const MAX_TOKENS: usize = 510;

/// Kokoro's phoneme vocabulary: IPA symbol to token id.
///
/// Generated from `tokenizer.json` in `onnx-community/Kokoro-82M-v1.0-ONNX`.
/// Ids are sparse — the model's embedding table is wider than the symbol set —
/// so they are listed rather than derived from position.
const VOCAB: &[(char, i64)] = &[
    ('$', 0),
    (';', 1),
    (':', 2),
    (',', 3),
    ('.', 4),
    ('!', 5),
    ('?', 6),
    ('—', 9),
    ('…', 10),
    ('"', 11),
    ('(', 12),
    (')', 13),
    ('“', 14),
    ('”', 15),
    (' ', 16),
    ('̃', 17),
    ('ʣ', 18),
    ('ʥ', 19),
    ('ʦ', 20),
    ('ʨ', 21),
    ('ᵝ', 22),
    ('ꭧ', 23),
    ('A', 24),
    ('I', 25),
    ('O', 31),
    ('Q', 33),
    ('S', 35),
    ('T', 36),
    ('W', 39),
    ('Y', 41),
    ('ᵊ', 42),
    ('a', 43),
    ('b', 44),
    ('c', 45),
    ('d', 46),
    ('e', 47),
    ('f', 48),
    ('h', 50),
    ('i', 51),
    ('j', 52),
    ('k', 53),
    ('l', 54),
    ('m', 55),
    ('n', 56),
    ('o', 57),
    ('p', 58),
    ('q', 59),
    ('r', 60),
    ('s', 61),
    ('t', 62),
    ('u', 63),
    ('v', 64),
    ('w', 65),
    ('x', 66),
    ('y', 67),
    ('z', 68),
    ('ɑ', 69),
    ('ɐ', 70),
    ('ɒ', 71),
    ('æ', 72),
    ('β', 75),
    ('ɔ', 76),
    ('ɕ', 77),
    ('ç', 78),
    ('ɖ', 80),
    ('ð', 81),
    ('ʤ', 82),
    ('ə', 83),
    ('ɚ', 85),
    ('ɛ', 86),
    ('ɜ', 87),
    ('ɟ', 90),
    ('ɡ', 92),
    ('ɥ', 99),
    ('ɨ', 101),
    ('ɪ', 102),
    ('ʝ', 103),
    ('ɯ', 110),
    ('ɰ', 111),
    ('ŋ', 112),
    ('ɳ', 113),
    ('ɲ', 114),
    ('ɴ', 115),
    ('ø', 116),
    ('ɸ', 118),
    ('θ', 119),
    ('œ', 120),
    ('ɹ', 123),
    ('ɾ', 125),
    ('ɻ', 126),
    ('ʁ', 128),
    ('ɽ', 129),
    ('ʂ', 130),
    ('ʃ', 131),
    ('ʈ', 132),
    ('ʧ', 133),
    ('ʊ', 135),
    ('ʋ', 136),
    ('ʌ', 138),
    ('ɣ', 139),
    ('ɤ', 140),
    ('χ', 142),
    ('ʎ', 143),
    ('ʒ', 147),
    ('ʔ', 148),
    ('ˈ', 156),
    ('ˌ', 157),
    ('ː', 158),
    ('ʰ', 162),
    ('ʲ', 164),
    ('↓', 169),
    ('→', 171),
    ('↗', 172),
    ('↘', 173),
    ('ᵻ', 177),
];

/// Punctuation that survives into the token stream.
///
/// `espeak-ng` drops punctuation from its IPA output and signals a clause break
/// with a newline instead, which loses the distinction between a comma and a
/// full stop. Kokoro has tokens for both and renders them as different pauses,
/// so the text is split on these characters and each one is re-inserted between
/// the phoneme runs around it.
const KEPT_PUNCTUATION: &[char] = &[',', '.', ';', ':', '!', '?', '—', '…', '(', ')', '"'];

/// Characters `espeak-ng` emits that are not phonemes and carry no token.
///
/// French liaison and Mandarin syllable boundaries come out as `-`. It marks a
/// join rather than a sound, so dropping it is not a lossy approximation.
const IGNORED_OUTPUT: &[char] = &['-', '\u{200d}'];

fn vocab() -> &'static HashMap<char, i64> {
    static MAP: OnceLock<HashMap<char, i64>> = OnceLock::new();
    MAP.get_or_init(|| VOCAB.iter().copied().collect())
}

/// The grapheme-to-phoneme engine that serves one language.
///
/// Japanese and Mandarin get their own variants here once their engines land;
/// see the module comment for why they cannot share the `espeak-ng` one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G2p {
    /// Japanese, via [`crate::ja`]. Like Mandarin it cannot go through
    /// `espeak-ng`, and its dictionary is too large to embed, so it is the one
    /// engine that depends on a file the daemon fetches.
    Japanese,
    /// Mandarin, via [`crate::zh`]. `espeak-ng` cannot serve it: it writes the
    /// tones as digits (`ni2 hao3`), which have no token here, where Kokoro was
    /// trained on `misaki`'s tone arrows.
    Mandarin,
    /// English, via [`crate::en`]: the misaki pipeline (lexicon, POS-aware
    /// heteronyms, verbalized numbers) with the bundled espeak-ng only as
    /// its out-of-vocabulary fallback — the same shape as Kokoro's training
    /// input. `british` selects the gb lexicon and accent.
    English {
        /// British lexicon and espeak accent when set; American otherwise.
        british: bool,
    },
    /// The bundled `espeak-ng`, addressed by the voice name it resolves.
    ///
    /// These are voice *file* names, which is not always the tag one would
    /// reach for: the linked `espeak_ng_SetVoiceByName` matches a file, while
    /// the `espeak-ng` command's `-v` also accepts language tags that alias
    /// one. So British English is `en` (the file `lang/gmw/en`) and not
    /// `en-gb`, and French is `fr` and not `fr-fr`. Both spellings produce
    /// identical phonemes; only the resolvable one is written here.
    Espeak(&'static str),
}

/// A language this backend can pronounce, and the engine that phonemizes it.
///
/// The BCP-47 tag is what the daemon sends on `POST /v1/synthesize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Language {
    /// BCP-47 tag as it appears in the manifest and on the wire.
    pub tag: &'static str,
    /// How this language is turned into phonemes.
    pub g2p: G2p,
}

/// Every language the backend declares, in manifest order.
///
/// `en` maps to American English because the default voice is American; a
/// British voice overrides it through [`g2p_for`].
pub const LANGUAGES: &[Language] = &[
    Language {
        tag: "en",
        g2p: G2p::English { british: false },
    },
    Language {
        tag: "es",
        g2p: G2p::Espeak("es"),
    },
    Language {
        tag: "fr",
        g2p: G2p::Espeak("fr"),
    },
    Language {
        tag: "it",
        g2p: G2p::Espeak("it"),
    },
    Language {
        tag: "pt",
        g2p: G2p::Espeak("pt-br"),
    },
    Language {
        tag: "hi",
        g2p: G2p::Espeak("hi"),
    },
    Language {
        tag: "zh",
        g2p: G2p::Mandarin,
    },
    Language {
        tag: "ja",
        g2p: G2p::Japanese,
    },
];

/// Resolve the phonemizer for a request.
///
/// A Kokoro voice id encodes its own accent in the first character (`a`
/// American, `b` British, `e` Spanish, and so on), and that is more specific
/// than the request's language tag: `en` with `bf_emma` means British English.
/// The voice therefore wins, and `language` only decides when the voice's
/// prefix says nothing useful.
#[must_use]
pub fn g2p_for(voice: &str, language: Option<&str>) -> G2p {
    let from_voice = match voice.as_bytes().first() {
        Some(b'a') => Some(G2p::English { british: false }),
        Some(b'b') => Some(G2p::English { british: true }),
        Some(b'e') => Some(G2p::Espeak("es")),
        Some(b'f') => Some(G2p::Espeak("fr")),
        Some(b'i') => Some(G2p::Espeak("it")),
        Some(b'p') => Some(G2p::Espeak("pt-br")),
        Some(b'h') => Some(G2p::Espeak("hi")),
        Some(b'z') => Some(G2p::Mandarin),
        Some(b'j') => Some(G2p::Japanese),
        _ => None,
    };
    from_voice
        .or_else(|| {
            language.and_then(|tag| {
                let base = tag.split('-').next().unwrap_or(tag);
                LANGUAGES.iter().find(|l| l.tag == base).map(|l| l.g2p)
            })
        })
        .unwrap_or(G2p::English { british: false })
}

/// Split `text` into alternating phonemizable runs and kept punctuation.
///
/// Returns `(run, trailing_punctuation)` pairs. Either half may be empty: the
/// run is empty for consecutive punctuation, the punctuation is empty for the
/// final run.
fn segment(text: &str) -> Vec<(String, Option<char>)> {
    let mut out = Vec::new();
    let mut run = String::new();
    for ch in text.chars() {
        if KEPT_PUNCTUATION.contains(&ch) {
            out.push((std::mem::take(&mut run), Some(ch)));
        } else {
            run.push(ch);
        }
    }
    if !run.trim().is_empty() {
        out.push((run, None));
    }
    out
}

/// Confirm the phonemizer works before the model is declared ready.
///
/// Called at load, not at first synthesis. Without it the failure lands in the
/// worst possible place: the weights parse, `/v1/status` reports `ready`, the
/// settings UI shows a loaded model — and then every single utterance fails.
/// A broken phonemizer should stop the load it makes useless, where the daemon
/// surfaces the reason, rather than surfacing as broken speech later.
///
/// This no longer checks for an installed program — espeak-ng is linked into
/// this binary — but it still earns its place: the bundled data is unpacked
/// lazily, and unpacking is the step that can fail on a full or read-only
/// temporary directory.
///
/// # Errors
/// Returns an error if the bundled engine cannot initialize or answers a known
/// word with nothing.
pub fn preflight(backend_dir: &std::path::Path) -> Result<()> {
    // Japanese needs a dictionary from the backend directory rather than an
    // embedded table; record where to find it before anything asks.
    crate::ja::set_backend_dir(backend_dir);

    let ipa = phonemize_run("hello", G2p::Espeak("en-us"))
        .context("the bundled espeak-ng could not phonemize")?;
    if ipa.trim().is_empty() {
        bail!("the bundled espeak-ng ran but produced no phonemes for a known word");
    }
    // Also warms the English pipeline: parsing the embedded lexicons and
    // tagger takes a moment, better spent at load than on the first
    // utterance — and a failure surfaces here, where the daemon reports it.
    let en = crate::en::phonemize_text("hello", false);
    if en.trim().is_empty() {
        bail!("the English g2p produced no phonemes for a known word");
    }
    log::info!("phonemizer preflight ok ({} -> {:?})", "hello", en.trim());
    Ok(())
}

/// Phonemize one run of text with the engine `g2p` names.
///
/// # Errors
/// Returns an error if that engine fails on this input.
fn phonemize_run(text: &str, g2p: G2p) -> Result<String> {
    match g2p {
        G2p::English { british } => Ok(crate::en::phonemize_text(text, british)),
        G2p::Espeak(voice) => crate::espeak::phonemize(text, voice),
        G2p::Mandarin => crate::zh::phonemize(text),
        G2p::Japanese => crate::ja::phonemize(text),
    }
}

/// Convert IPA text to token ids, dropping symbols with no token.
///
/// Unknown symbols are dropped rather than refused: `espeak-ng` occasionally
/// emits a marker for an input it could not pronounce, and losing one symbol is
/// better than failing an utterance the user can otherwise hear.
fn to_tokens(ipa: &str, out: &mut Vec<i64>) {
    let vocab = vocab();
    for ch in ipa.chars() {
        if IGNORED_OUTPUT.contains(&ch) {
            continue;
        }
        if let Some(&id) = vocab.get(&ch) {
            // Collapse runs of whitespace: Kokoro renders a doubled space as a
            // doubled pause, and joining phoneme runs naturally produces them.
            if id == 16 && out.last() == Some(&16) {
                continue;
            }
            out.push(id);
        } else {
            log::debug!("dropping symbol {ch:?} with no Kokoro token");
        }
    }
}

/// Whether a token stream contains anything that is actually pronounced.
///
/// Punctuation and spaces are real tokens — Kokoro renders them as pauses — but
/// a stream of nothing else would synthesize silence. Distinguishing the two is
/// left to the caller rather than decided here, because the right answer
/// differs: a whole request of punctuation is a failed utterance, while one
/// punctuation-only segment inside a longer utterance is just a pause to skip.
#[must_use]
pub fn is_pronounceable(tokens: &[i64]) -> bool {
    tokens.iter().any(|id| !PUNCTUATION_IDS.contains(id))
}

/// Token ids that produce a pause rather than a sound: the kept punctuation
/// plus the space.
const PUNCTUATION_IDS: &[i64] = &[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16];

/// Phonemize `text` and return Kokoro token ids, without the pad tokens.
///
/// Returns an empty vector when `text` has nothing to phonemize at all. It is
/// not an error here: the daemon chunks a long utterance, and a chunk that came
/// out as punctuation should cost that chunk rather than the whole request.
/// Use [`is_pronounceable`] to tell silence from speech.
///
/// The result is truncated to [`MAX_TOKENS`]. The daemon already chunks to the
/// model's `max_input_chars`, so truncation here is a backstop rather than the
/// normal path.
///
/// # Errors
/// Returns an error if the phonemizer for this voice fails.
pub fn phonemize(text: &str, voice: &str, language: Option<&str>) -> Result<Vec<i64>> {
    let g2p = g2p_for(voice, language);
    let mut tokens = Vec::new();

    // The misaki English pipeline carries punctuation through itself — that
    // is part of what it was trained on — so it gets the text whole rather
    // than pre-split by `segment`.
    if let G2p::English { british } = g2p {
        let ipa = crate::en::phonemize_text(text, british);
        to_tokens(ipa.trim(), &mut tokens);
        tokens.truncate(MAX_TOKENS);
        return Ok(tokens);
    }

    for (run, punct) in segment(text) {
        if !run.trim().is_empty() {
            let ipa = phonemize_run(run.trim(), g2p)?;
            to_tokens(ipa.trim(), &mut tokens);
        }
        if let Some(p) = punct {
            to_tokens(&p.to_string(), &mut tokens);
            // A pause reads as a pause only with air after it.
            to_tokens(" ", &mut tokens);
        }
    }

    while tokens.last() == Some(&16) {
        tokens.pop();
    }
    if tokens.len() > MAX_TOKENS {
        log::warn!(
            "truncating {} tokens to the {MAX_TOKENS}-token limit",
            tokens.len()
        );
        tokens.truncate(MAX_TOKENS);
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The claim the whole phonemizer rests on. If `espeak-ng` ever emits a
    /// symbol Kokoro has no token for, this catches it here rather than as
    /// mysteriously slurred audio.
    #[test]
    fn espeak_output_stays_inside_the_vocab() {
        let corpus: &[(&str, &str)] = &[
            (
                "en-us",
                "Hello there, this is a test of the emergency system.",
            ),
            ("en", "Rather good weather today, wouldn't you say?"),
            ("es", "Hola, esto es una prueba del sistema de voz."),
            ("fr", "Bonjour, ceci est un test du systeme vocal."),
            ("it", "Ciao, questo e un test del sistema vocale."),
            ("pt-br", "Ola, este e um teste do sistema de voz."),
            ("hi", "yah aavaaz pranaalee ka pareekshan hai."),
        ];
        let vocab = vocab();
        for (voice, text) in corpus {
            let ipa = phonemize_run(text, G2p::Espeak(voice)).expect("bundled espeak-ng runs");
            for ch in ipa.chars() {
                assert!(
                    vocab.contains_key(&ch) || IGNORED_OUTPUT.contains(&ch),
                    "{voice}: {ch:?} has no Kokoro token (from {ipa:?})"
                );
            }
        }
    }

    /// The same guarantee for Mandarin, whose phonemes come from a different
    /// engine: every symbol it emits must have a Kokoro token, or the voice
    /// would synthesize from a stream of dropped symbols.
    #[test]
    fn mandarin_output_stays_inside_the_vocab() {
        let vocab = vocab();
        for text in [
            "你好，世界。",
            "今天天气很好。",
            "他在银行工作。",
            "一二三四五六七八九十。",
            "请问洗手间在哪里？",
        ] {
            let ipa = phonemize_run(text, G2p::Mandarin).expect("mandarin g2p runs");
            for ch in ipa.chars() {
                assert!(
                    vocab.contains_key(&ch) || IGNORED_OUTPUT.contains(&ch),
                    "{ch:?} (U+{:04X}) has no Kokoro token (from {ipa:?})",
                    ch as u32
                );
            }
        }
    }

    /// The same guarantee for Japanese. This reads the reference answers
    /// rather than running the phonemizer, so it holds even where the `UniDic`
    /// dictionary is absent: what it asserts is that the alphabet `misaki`
    /// emits for Japanese is one Kokoro has tokens for.
    #[test]
    fn japanese_output_stays_inside_the_vocab() {
        let vocab = vocab();
        let mut checked = 0;
        for line in include_str!("../data/ja-golden.tsv").lines() {
            let Some((text, ipa)) = line.split_once('\t') else {
                continue;
            };
            checked += 1;
            for ch in ipa.chars() {
                assert!(
                    vocab.contains_key(&ch) || IGNORED_OUTPUT.contains(&ch),
                    "{ch:?} (U+{:04X}) has no Kokoro token (from {text:?} -> {ipa:?})",
                    ch as u32
                );
            }
        }
        assert!(checked > 700, "only checked {checked} cases");
    }

    /// And for Mandarin, over the whole reference fixture rather than a
    /// handful of sentences.
    #[test]
    fn mandarin_reference_output_stays_inside_the_vocab() {
        let vocab = vocab();
        let mut checked = 0;
        for line in include_str!("../data/zh-golden.tsv").lines() {
            let Some((text, ipa)) = line.split_once('\t') else {
                continue;
            };
            checked += 1;
            for ch in ipa.chars() {
                assert!(
                    vocab.contains_key(&ch) || IGNORED_OUTPUT.contains(&ch),
                    "{ch:?} (U+{:04X}) has no Kokoro token (from {text:?} -> {ipa:?})",
                    ch as u32
                );
            }
        }
        assert!(checked > 400, "only checked {checked} cases");
    }

    /// The same guarantee for English, over the misaki reference fixture:
    /// the alphabet the English pipeline emits is one Kokoro has tokens for.
    #[test]
    fn english_reference_output_stays_inside_the_vocab() {
        let vocab = vocab();
        let mut checked = 0;
        for line in include_str!("../data/en-golden.tsv").lines() {
            let mut cols = line.split('\t');
            let (Some(_), Some(_), Some(text), Some(ipa)) =
                (cols.next(), cols.next(), cols.next(), cols.next())
            else {
                continue;
            };
            checked += 1;
            for ch in ipa.chars() {
                assert!(
                    vocab.contains_key(&ch) || IGNORED_OUTPUT.contains(&ch),
                    "{ch:?} (U+{:04X}) has no Kokoro token (from {text:?} -> {ipa:?})",
                    ch as u32
                );
            }
        }
        assert!(checked > 1600, "only checked {checked} cases");
    }

    #[test]
    fn punctuation_survives_into_the_token_stream() {
        let comma = vocab()[&','];
        let period = vocab()[&'.'];
        let tokens = phonemize("Wait, stop.", "af_heart", Some("en")).unwrap();
        assert!(tokens.contains(&comma), "comma token missing: {tokens:?}");
        assert!(tokens.contains(&period), "period token missing: {tokens:?}");
    }

    /// The pause has to land where the writer put it, not merely exist.
    #[test]
    fn punctuation_lands_between_the_runs_it_separated() {
        let comma = vocab()[&','];
        let tokens = phonemize("Wait, stop.", "af_heart", Some("en")).unwrap();
        let at = tokens.iter().position(|&t| t == comma).unwrap();
        assert!(at > 0, "comma cannot be the first token: {tokens:?}");
        assert!(at < tokens.len() - 1, "comma cannot be last: {tokens:?}");
    }

    #[test]
    fn a_voice_prefix_outranks_the_language_tag() {
        assert_eq!(
            g2p_for("bf_emma", Some("en")),
            G2p::English { british: true }
        );
        assert_eq!(
            g2p_for("af_heart", Some("en")),
            G2p::English { british: false }
        );
        assert_eq!(g2p_for("ef_dora", Some("en")), G2p::Espeak("es"));
    }

    /// Every voice name the dispatcher can produce has to be one the linked
    /// espeak-ng actually resolves. The command-line tool accepts aliases the
    /// library does not (`en-gb`, `fr-fr`), so a name that looks right can
    /// still fail at synthesis time.
    #[test]
    fn every_routed_voice_name_resolves() {
        // Japanese is excluded: its dictionary is fetched by the daemon rather
        // than embedded, so it is exercised by `ja`'s own tests, which say so
        // when it is absent instead of failing here.
        for lang in LANGUAGES.iter().filter(|l| l.g2p != G2p::Japanese) {
            phonemize_run("test", lang.g2p).unwrap_or_else(|e| {
                panic!("language {:?} routes to {:?}: {e}", lang.tag, lang.g2p)
            });
        }
        for prefix in ["a", "b", "e", "f", "i", "p", "h", "z"] {
            let g2p = g2p_for(&format!("{prefix}f_x"), None);
            phonemize_run("test", g2p)
                .unwrap_or_else(|e| panic!("voice prefix {prefix:?} routes to {g2p:?}: {e}"));
        }
    }

    /// A voice whose prefix says nothing falls back to the request's language,
    /// then to American English. `q` is used rather than `z`, which now names
    /// Kokoro's Mandarin voices.
    #[test]
    fn an_unknown_voice_prefix_falls_back_to_the_language() {
        assert_eq!(g2p_for("qq_unknown", Some("it")), G2p::Espeak("it"));
        // The bare language table maps `en` to American; only a voice
        // prefix expresses British, so `en-GB` without one falls to the
        // default accent, as it did when espeak served English.
        assert_eq!(
            g2p_for("qq_unknown", Some("en-GB")),
            G2p::English { british: false }
        );
        assert_eq!(g2p_for("qq_unknown", None), G2p::English { british: false });
        // and the language tag alone is enough to reach Mandarin
        assert_eq!(g2p_for("qq_unknown", Some("zh")), G2p::Mandarin);
    }

    /// Punctuation is a pause, not a phoneme. Both are tokens, and only the
    /// caller knows whether a pause-only run is a problem.
    #[test]
    fn punctuation_only_text_yields_pauses_that_are_not_pronounceable() {
        let dots = phonemize("...", "af_heart", Some("en")).unwrap();
        assert!(!dots.is_empty(), "punctuation still produces pause tokens");
        assert!(!is_pronounceable(&dots));

        assert!(phonemize("   ", "af_heart", Some("en")).unwrap().is_empty());
        assert!(!is_pronounceable(&[]));
    }

    /// The preflight has to actually exercise the phonemizer, not merely check
    /// that it is linked: unpacking the bundled data is the step that can fail,
    /// and an engine with no data answers with nothing.
    #[test]
    fn preflight_passes_with_the_bundled_engine() {
        preflight(std::path::Path::new("/nonexistent")).expect("the bundled espeak-ng works");
    }

    #[test]
    fn ordinary_text_is_pronounceable() {
        let tokens = phonemize("hello", "af_heart", Some("en")).unwrap();
        assert!(is_pronounceable(&tokens));
    }

    /// Longer text must not silently exceed what the style vectors can index.
    #[test]
    fn tokens_are_capped_at_the_style_vector_row_count() {
        let long = "The quick brown fox jumps over the lazy dog. ".repeat(60);
        let tokens = phonemize(&long, "af_heart", Some("en")).unwrap();
        assert!(tokens.len() <= MAX_TOKENS, "got {}", tokens.len());
    }

    #[test]
    fn repeated_spaces_collapse_to_one_pause() {
        let space = vocab()[&' '];
        let tokens = phonemize("hello     world", "af_heart", Some("en")).unwrap();
        assert!(
            !tokens.windows(2).any(|w| w == [space, space]),
            "doubled pause in {tokens:?}"
        );
    }
}
