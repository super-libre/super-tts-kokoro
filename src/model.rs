// SPDX-License-Identifier: GPL-3.0-only
//! The Kokoro ONNX session.
//!
//! Kokoro takes three inputs — the phoneme token ids, a style vector chosen by
//! token count, and a speed multiplier — and returns a mono waveform at
//! 24 kHz. Inference is synchronous and CPU-bound, and `ort` requires `&mut`
//! on the session because ONNX Runtime is not internally thread safe, so the
//! whole model lives behind one lock and callers run it off the async runtime.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::phonemes;
use crate::voices::{STYLE_DIM, VoiceSet};

/// Kokoro's fixed output rate. Not configurable — it is baked into the decoder.
pub const SAMPLE_RATE: u32 = 24_000;

/// The pad token wrapped around every sequence.
///
/// Kokoro was trained with the sequence bracketed by `$` (id 0); omitting it
/// clips the first and last phoneme.
const PAD: i64 = 0;

/// Longest segment, in characters, that streaming will synthesize in one pass.
///
/// Kokoro produces a whole segment's audio in a single forward pass, so this is
/// what bounds latency to first sound: shorter segments start speaking sooner,
/// longer ones give the model more context for prosody. The daemon has already
/// chunked to `max_input_chars`; this splits again, finer, purely for latency.
const STREAM_SEGMENT_CHARS: usize = 180;

/// A loaded Kokoro model and its voices.
pub struct Kokoro {
    session: Session,
    voices: VoiceSet,
    default_voice: String,
}

// `ort::Session` has no `Debug`, so this is written by hand. `session` is
// named as an opaque field rather than omitted: a `Debug` that silently drops a
// field misrepresents the value it is printing.
impl std::fmt::Debug for Kokoro {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kokoro")
            .field("session", &"<ort::Session>")
            .field("default_voice", &self.default_voice)
            .field("voices", &self.voices.ids().count())
            .finish()
    }
}

/// Locate the ONNX weights inside a backend directory.
///
/// The daemon writes files to the `destination` paths the manifest declared, so
/// this must agree with `backend.toml`. Several filenames are accepted so a
/// user can swap in a quantized build without editing code.
fn find_weights(model_dir: &Path) -> Result<PathBuf> {
    const CANDIDATES: &[&str] = &[
        "model.onnx",
        "model_fp16.onnx",
        "model_quantized.onnx",
        "model_q8f16.onnx",
    ];
    for name in CANDIDATES {
        let path = model_dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    bail!(
        "no Kokoro weights in {} (looked for {})",
        model_dir.display(),
        CANDIDATES.join(", ")
    )
}

impl Kokoro {
    /// Load the model and its voices from a backend directory.
    ///
    /// # Errors
    /// Returns an error if `espeak-ng` is unavailable, the weights or voice
    /// packs are missing, or ONNX Runtime cannot build a session over them.
    pub fn load(backend_dir: &Path, model_name: &str) -> Result<Self> {
        // Before the expensive part: a backend that cannot phonemize cannot
        // speak, and finding that out at load costs a millisecond instead of
        // presenting the user a "ready" model that fails every utterance.
        phonemes::preflight(backend_dir)?;

        let model_dir = backend_dir.join("models").join(model_name);
        let weights = find_weights(&model_dir)?;
        log::info!("loading {}", weights.display());

        // `ort::Error<T>` carries the partially-built value on failure, so it
        // is not a plain `std::error::Error` and `anyhow::Context` will not
        // apply to it. Each step is flattened to a message instead.
        let session = Session::builder()
            .map_err(|e| anyhow!("creating an ONNX Runtime session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("setting the optimization level: {e}"))?
            .commit_from_file(&weights)
            .map_err(|e| anyhow!("loading {}: {e}", weights.display()))?;

        let voices = VoiceSet::load_dir(&model_dir.join("voices"))?;
        let default_voice = voices.ids().next().context("no voices loaded")?.to_string();

        Ok(Self {
            session,
            voices,
            default_voice,
        })
    }

    /// Whether `id` names a loaded voice.
    #[must_use]
    pub fn has_voice(&self, id: &str) -> bool {
        self.voices.get(id).is_some()
    }

    /// The voice used when a request names none.
    #[must_use]
    pub fn default_voice(&self) -> &str {
        &self.default_voice
    }

    /// Synthesize one already-segmented run of text.
    ///
    /// Returns no samples when the segment holds nothing pronounceable, so a
    /// caller can skip it without treating it as a failure.
    ///
    /// # Errors
    /// Returns an error if the voice is unknown or inference fails.
    pub fn synthesize_segment(
        &mut self,
        text: &str,
        voice: &str,
        language: Option<&str>,
        speed: f32,
    ) -> Result<Vec<f32>> {
        let tokens = phonemes::phonemize(text, voice, language)?;
        if !phonemes::is_pronounceable(&tokens) {
            return Ok(Vec::new());
        }
        let style = self
            .voices
            .get(voice)
            .with_context(|| format!("unknown voice {voice}"))?
            .style_for(tokens.len())
            .to_vec();

        // The style row is chosen by the unpadded count, but the model is fed
        // the padded sequence.
        let mut ids = Vec::with_capacity(tokens.len() + 2);
        ids.push(PAD);
        ids.extend_from_slice(&tokens);
        ids.push(PAD);

        let ids = ndarray::Array2::<i64>::from_shape_vec((1, ids.len()), ids)
            .context("shaping input_ids")?;
        let style = ndarray::Array2::<f32>::from_shape_vec((1, STYLE_DIM), style)
            .context("shaping style")?;
        let speed = ndarray::Array1::<f32>::from_vec(vec![speed]);

        let ids_ref = TensorRef::from_array_view(&ids).map_err(|e| anyhow!("input_ids: {e}"))?;
        let style_ref = TensorRef::from_array_view(&style).map_err(|e| anyhow!("style: {e}"))?;
        let speed_ref = TensorRef::from_array_view(&speed).map_err(|e| anyhow!("speed: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => ids_ref,
                "style" => style_ref,
                "speed" => speed_ref,
            ])
            .map_err(|e| anyhow!("running Kokoro inference: {e}"))?;

        let (_, samples) = outputs["waveform"]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("reading the waveform output: {e}"))?;
        Ok(samples.to_vec())
    }
}

/// Split text into segments small enough to keep latency to first sound low.
///
/// Splits on sentence-ending punctuation first and only falls back to a word
/// boundary when a single sentence is longer than [`STREAM_SEGMENT_CHARS`].
/// Breaking mid-sentence costs prosody — Kokoro chooses its style vector per
/// segment — so it is a last resort rather than a fixed window.
#[must_use]
pub fn split_for_streaming(text: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();

    for sentence in split_keeping_terminators(text) {
        if current.chars().count() + sentence.chars().count() <= STREAM_SEGMENT_CHARS {
            current.push_str(&sentence);
            continue;
        }
        if !current.trim().is_empty() {
            segments.push(std::mem::take(&mut current));
        }
        if sentence.chars().count() <= STREAM_SEGMENT_CHARS {
            current = sentence;
        } else {
            segments.extend(split_on_words(&sentence));
        }
    }
    if !current.trim().is_empty() {
        segments.push(current);
    }
    if segments.is_empty() && !text.trim().is_empty() {
        segments.push(text.to_string());
    }
    segments
}

/// Split after `.`, `!`, `?`, or a newline, keeping the terminator attached.
fn split_keeping_terminators(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Break an over-long run at word boundaries.
fn split_on_words(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for word in text.split_inclusive(char::is_whitespace) {
        if !current.is_empty()
            && current.chars().count() + word.chars().count() > STREAM_SEGMENT_CHARS
        {
            out.push(std::mem::take(&mut current));
        }
        current.push_str(word);
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever the split, the reader must hear every character exactly once.
    fn assert_lossless(text: &str) {
        let joined: String = split_for_streaming(text).concat();
        assert_eq!(joined, text, "split lost or duplicated text");
    }

    #[test]
    fn short_text_is_a_single_segment() {
        assert_eq!(split_for_streaming("Hello there."), vec!["Hello there."]);
    }

    #[test]
    fn sentences_accumulate_until_the_segment_is_full() {
        let text = "One. Two. Three.";
        assert_eq!(split_for_streaming(text), vec![text]);
        assert_lossless(text);
    }

    #[test]
    fn a_long_passage_splits_on_sentence_boundaries() {
        let text = "This is a sentence with some length to it. ".repeat(6);
        let segments = split_for_streaming(&text);
        assert!(segments.len() > 1, "expected a split, got {segments:?}");
        for s in &segments {
            assert!(s.chars().count() <= STREAM_SEGMENT_CHARS + 44, "{s:?}");
        }
        assert_lossless(&text);
    }

    /// A sentence longer than the window has no boundary to use, so it falls
    /// back to words rather than growing without bound.
    #[test]
    fn a_single_over_long_sentence_falls_back_to_word_boundaries() {
        let text = "word ".repeat(120);
        let segments = split_for_streaming(&text);
        assert!(segments.len() > 1);
        for s in &segments {
            assert!(s.chars().count() <= STREAM_SEGMENT_CHARS + 5, "{s:?}");
        }
        assert_lossless(&text);
    }

    /// A break must never land inside a word.
    #[test]
    fn word_fallback_never_splits_mid_word() {
        let text = "alpha bravo charlie delta echo foxtrot golf hotel ".repeat(8);
        for s in split_for_streaming(&text) {
            assert!(
                s.starts_with("alpha") || s.starts_with(char::is_alphabetic),
                "segment starts mid-word: {s:?}"
            );
            assert!(!s.trim_end().ends_with("alph"), "truncated word in {s:?}");
        }
        assert_lossless(&text);
    }

    #[test]
    fn empty_and_blank_text_produce_no_segments() {
        assert!(split_for_streaming("").is_empty());
        assert!(split_for_streaming("   \n ").is_empty());
    }

    /// Text with no terminator at all still has to come out whole.
    #[test]
    fn unterminated_text_is_not_dropped() {
        assert_eq!(split_for_streaming("no terminator"), vec!["no terminator"]);
        assert_lossless("no terminator here at all");
    }

    #[test]
    fn missing_weights_are_reported_with_the_directory() {
        let dir = std::env::temp_dir().join(format!("kokoro-noweights-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = find_weights(&dir).unwrap_err().to_string();
        assert!(err.contains("model.onnx"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
