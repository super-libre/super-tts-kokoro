// SPDX-License-Identifier: GPL-3.0-only
//! Kokoro voice packs.
//!
//! A voice is not an embedding but a table of them: `510 x 1 x 256` float32,
//! little-endian, one row per token count. `StyleTTS2` conditions on a style
//! vector chosen by how long the utterance is, so the same voice speaks a
//! two-word phrase and a full sentence with different prosody. Picking the row
//! is therefore part of synthesis, not part of loading.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// Rows in a voice pack — one per possible token count.
pub const STYLE_ROWS: usize = 510;
/// Width of a single style vector.
pub const STYLE_DIM: usize = 256;
/// Exact size of a well-formed voice file.
pub const VOICE_FILE_LEN: usize = STYLE_ROWS * STYLE_DIM * 4;

/// One voice pack: every style row, flattened.
#[derive(Debug, Clone)]
pub struct Voice {
    styles: Vec<f32>,
}

impl Voice {
    /// Parse a voice pack from its raw `.bin` bytes.
    ///
    /// # Errors
    /// Returns an error if the file is not exactly [`VOICE_FILE_LEN`] bytes.
    /// The length is the only integrity check available — the format carries no
    /// header or magic — so it is enforced strictly rather than by truncating a
    /// long file or zero-filling a short one.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != VOICE_FILE_LEN {
            bail!(
                "voice pack is {} bytes, expected {VOICE_FILE_LEN} ({STYLE_ROWS}x{STYLE_DIM} f32)",
                bytes.len()
            );
        }
        let styles = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        Ok(Self { styles })
    }

    /// The style vector for an utterance of `token_count` tokens.
    ///
    /// `token_count` is the unpadded count: the pad tokens the model wants
    /// around the sequence are not part of what selects the style. A count at
    /// or past the table is clamped to the last row, which is the longest
    /// prosody the voice was trained to produce.
    #[must_use]
    pub fn style_for(&self, token_count: usize) -> &[f32] {
        let row = token_count.min(STYLE_ROWS - 1);
        &self.styles[row * STYLE_DIM..(row + 1) * STYLE_DIM]
    }
}

/// Every voice pack found on disk, by id.
#[derive(Debug, Default)]
pub struct VoiceSet {
    voices: BTreeMap<String, Voice>,
}

impl VoiceSet {
    /// Load every `*.bin` in `dir` as a voice, named by its file stem.
    ///
    /// A file that does not parse is skipped with a warning rather than failing
    /// the load: one corrupt pack should cost the user that voice, not the
    /// whole backend.
    ///
    /// # Errors
    /// Returns an error if `dir` cannot be read, or if it yielded no usable
    /// voice — a Kokoro model with no voice can synthesize nothing, so that is
    /// a failed load rather than an empty success.
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let entries =
            std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?;
        let mut voices = BTreeMap::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "bin") {
                continue;
            }
            let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            match std::fs::read(&path)
                .map_err(anyhow::Error::from)
                .and_then(|b| Voice::from_bytes(&b))
            {
                Ok(voice) => {
                    voices.insert(id.to_string(), voice);
                }
                Err(e) => log::warn!("skipping voice {id}: {e:#}"),
            }
        }
        if voices.is_empty() {
            bail!("no usable voice packs in {}", dir.display());
        }
        log::info!("loaded {} voices from {}", voices.len(), dir.display());
        Ok(Self { voices })
    }

    /// Look up a voice by id.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Voice> {
        self.voices.get(id)
    }

    /// Every loaded voice id, sorted.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.voices.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack() -> Vec<u8> {
        // Row r is filled with the value r, so a mis-indexed lookup is visible
        // rather than plausible.
        let mut bytes = Vec::with_capacity(VOICE_FILE_LEN);
        for row in 0..STYLE_ROWS {
            #[allow(clippy::cast_precision_loss)]
            let v = row as f32;
            for _ in 0..STYLE_DIM {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn a_style_row_is_selected_by_token_count() {
        let v = Voice::from_bytes(&pack()).unwrap();
        assert_eq!(v.style_for(0), [0.0; STYLE_DIM]);
        assert_eq!(v.style_for(7), [7.0; STYLE_DIM]);
        assert_eq!(v.style_for(509), [509.0; STYLE_DIM]);
    }

    /// Clamping rather than panicking matters: the token cap and the row count
    /// are the same number today, and a future mismatch should degrade to the
    /// longest style instead of killing the utterance.
    #[test]
    fn a_token_count_past_the_table_clamps_to_the_last_row() {
        let v = Voice::from_bytes(&pack()).unwrap();
        assert_eq!(v.style_for(STYLE_ROWS), [509.0; STYLE_DIM]);
        assert_eq!(v.style_for(100_000), [509.0; STYLE_DIM]);
    }

    #[test]
    fn a_style_vector_is_exactly_one_row_wide() {
        let v = Voice::from_bytes(&pack()).unwrap();
        assert_eq!(v.style_for(3).len(), STYLE_DIM);
    }

    #[test]
    fn a_wrong_sized_pack_is_rejected() {
        assert!(Voice::from_bytes(&[]).is_err());
        assert!(Voice::from_bytes(&vec![0; VOICE_FILE_LEN - 4]).is_err());
        assert!(Voice::from_bytes(&vec![0; VOICE_FILE_LEN + 4]).is_err());
    }

    #[test]
    fn a_directory_with_no_voices_fails_to_load() {
        let dir = std::env::temp_dir().join(format!("kokoro-voices-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(VoiceSet::load_dir(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// One bad pack must not cost the user every other voice.
    #[test]
    fn a_corrupt_pack_is_skipped_and_the_rest_load() {
        let dir = std::env::temp_dir().join(format!("kokoro-voices-mixed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("af_good.bin"), pack()).unwrap();
        std::fs::write(dir.join("af_short.bin"), [0_u8; 16]).unwrap();
        std::fs::write(dir.join("notes.txt"), b"ignored").unwrap();

        let set = VoiceSet::load_dir(&dir).unwrap();
        assert!(set.get("af_good").is_some());
        assert!(set.get("af_short").is_none());
        assert_eq!(set.ids().collect::<Vec<_>>(), ["af_good"]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
