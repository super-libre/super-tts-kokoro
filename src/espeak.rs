// SPDX-License-Identifier: GPL-3.0-only
//! The bundled espeak-ng phonemizer.
//!
//! espeak-ng is built from `vendor/espeak-ng` and linked into this binary (see
//! `build.rs`), rather than executed as a subprocess found on `PATH`. The
//! backend therefore needs no system package: it is the only dependency the
//! manifest could not declare, and a backend that cannot be installed by the
//! daemon alone is not really installable.
//!
//! Only four entry points are declared by hand. Generating bindings would pull
//! `bindgen` and a `libclang` into the build for a surface this small, and the
//! four signatures below are stable ABI that espeak-ng has not changed in
//! years.
//!
//! # Thread safety
//!
//! espeak-ng keeps the selected voice and the clause cursor in process-wide
//! statics, so a concurrent phonemize would interleave two utterances. Every
//! call therefore runs under [`ESPEAK`]. The lock is uncontended in practice —
//! the caller already serializes synthesis behind the model mutex — so it costs
//! nothing and removes the dependency on that remaining true.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, anyhow, bail};

/// The trimmed `espeak-ng-data` for the languages this backend declares,
/// compressed at build time and carried inside the executable.
///
/// espeak-ng reads its dictionaries from a directory, so this is unpacked once
/// at load. Embedding rather than installing beside the binary is deliberate:
/// the backend then carries everything it uses and assumes nothing about what
/// the daemon put on disk.
const ESPEAK_DATA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/espeak-ng-data.tar.gz"));

/// `espeak_Initialize` audio mode. Nothing is ever played — only
/// `espeak_TextToPhonemes` is called — but the parameter is not optional.
const AUDIO_OUTPUT_SYNCHRONOUS: c_int = 0x02;
/// Return errors instead of calling `exit()` from inside the library.
const INITIALIZE_DONT_EXIT: c_int = 0x8000;
/// Ask for IPA rather than espeak's own phoneme mnemonics.
const PHONEME_IPA: c_int = 0x02;
/// The input is UTF-8.
const CHARS_UTF8: c_int = 1;

/// espeak-ng holds its data directory in a fixed 160-byte buffer and silently
/// truncates anything longer, then fails to find data at the truncated path.
const N_PATH_HOME: usize = 160;

unsafe extern "C" {
    fn espeak_Initialize(
        output: c_int,
        buflength: c_int,
        path: *const c_char,
        options: c_int,
    ) -> c_int;
    fn espeak_ng_SetVoiceByName(name: *const c_char) -> c_int;
    fn espeak_TextToPhonemes(
        textptr: *mut *const c_void,
        textmode: c_int,
        phonememode: c_int,
    ) -> *const c_char;
}

/// Serializes access to espeak-ng's process-wide state.
static ESPEAK: Mutex<()> = Mutex::new(());

/// Result of the one-time initialization, so a failure is reported identically
/// to every caller instead of being retried per utterance.
static INIT: OnceLock<Result<(), String>> = OnceLock::new();

/// Unpack the embedded data and initialize espeak-ng, at most once.
///
/// # Errors
/// Returns an error if the data cannot be unpacked or espeak-ng refuses to
/// initialize against it.
pub fn init() -> Result<()> {
    match INIT.get_or_init(|| initialize().map_err(|e| format!("{e:#}"))) {
        Ok(()) => Ok(()),
        Err(e) => Err(anyhow!("{e}")),
    }
}

fn initialize() -> Result<()> {
    let root = unpack_data().context("unpacking the bundled espeak-ng data")?;

    // espeak looks for `<root>/espeak-ng-data`, so `root` is what it is given.
    let path = CString::new(root.as_os_str().as_encoded_bytes())
        .context("espeak-ng data path contains a NUL byte")?;

    let _guard = ESPEAK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // SAFETY: `path` outlives the call, and espeak copies the path it is given.
    let sample_rate = unsafe {
        espeak_Initialize(
            AUDIO_OUTPUT_SYNCHRONOUS,
            0,
            path.as_ptr(),
            INITIALIZE_DONT_EXIT,
        )
    };
    if sample_rate <= 0 {
        bail!(
            "espeak-ng failed to initialize against its bundled data at {} (code {sample_rate})",
            root.display()
        );
    }
    log::info!("bundled espeak-ng initialized from {}", root.display());
    Ok(())
}

/// Marker written once extraction has finished, and checked before the
/// directory is trusted.
///
/// A file from the archive itself cannot play this role. `tar` writes entries
/// in archive order, so the obvious candidate — `phontab` — appears near the
/// *start*, and an abandoned extraction would advertise itself as complete
/// while missing most of its dictionaries. That failure is quiet and remote
/// from its cause: espeak initializes happily and then reports no voice for
/// whichever language did not make it.
const COMPLETE_MARKER: &str = ".complete";

/// Counter making each extraction attempt's scratch directory distinct.
static ATTEMPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Unpack [`ESPEAK_DATA`] into a temporary directory and return its parent.
///
/// The directory is named for a digest of the archive, which makes this
/// idempotent: a second run of the same build reuses what is already there, and
/// a rebuild with different data lands somewhere else instead of reading a
/// stale mixture. Extraction goes to a scratch path and is renamed into place,
/// so a crash midway cannot leave a half-populated directory that later looks
/// complete.
///
/// The scratch path is unique per *attempt*, not per process: two threads
/// racing here would otherwise unpack into one directory and rename it while
/// the other was still writing, publishing a tree that is missing whatever had
/// not been written yet.
fn unpack_data() -> Result<PathBuf> {
    let root = std::env::temp_dir().join(format!("super-tts-espeak-{:016x}", digest(ESPEAK_DATA)));

    let len = root.as_os_str().len();
    if len >= N_PATH_HOME {
        bail!(
            "the temporary directory path is {len} bytes, over espeak-ng's \
             {N_PATH_HOME}-byte limit: {}. Set TMPDIR to a shorter path.",
            root.display()
        );
    }

    if root.join(COMPLETE_MARKER).is_file() {
        return Ok(root);
    }

    let attempt = ATTEMPT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let scratch = root.with_extension(format!("tmp-{}-{attempt}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let staged = scratch.join("espeak-ng-data");
    std::fs::create_dir_all(&staged).with_context(|| format!("creating {}", staged.display()))?;

    let decoder = flate2::read::GzDecoder::new(ESPEAK_DATA);
    tar::Archive::new(decoder)
        .unpack(&staged)
        .with_context(|| format!("extracting into {}", staged.display()))?;
    std::fs::write(scratch.join(COMPLETE_MARKER), [])
        .with_context(|| format!("marking {} complete", scratch.display()))?;

    // A directory here without the marker cannot be in use by anyone — nothing
    // trusts it — so it is an abandoned extraction and is cleared rather than
    // left to block every future load.
    if root.exists() && !root.join(COMPLETE_MARKER).is_file() {
        let _ = std::fs::remove_dir_all(&root);
    }

    match std::fs::rename(&scratch, &root) {
        Ok(()) => {}
        // Someone else published first. Theirs is as good as ours, so long as
        // it is actually finished.
        Err(_) if root.join(COMPLETE_MARKER).is_file() => {
            let _ = std::fs::remove_dir_all(&scratch);
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&scratch);
            return Err(e).with_context(|| format!("installing {}", root.display()));
        }
    }
    Ok(root)
}

/// FNV-1a over the archive. Not a security boundary — it only has to change
/// when the bundled data changes.
fn digest(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// IPA with U+0361 ties inside multi-character phonemes, the mode the
/// `phonemizer` library uses when constructed with `tie=...` — which is how
/// misaki drives espeak. The tie is bit 7 plus the tie codepoint in the high
/// bits; phonemizer then swaps U+0361 for its caller's tie character.
const PHONEME_IPA_TIED: c_int = 0x02 | 0x80 | (0x361 << 8);

/// Phonemize `text` with the espeak-ng voice `voice`, returning IPA.
///
/// Clauses come back one at a time, joined by a space: espeak signals a clause
/// break by ending one string and starting another, and the caller has already
/// split the punctuation that caused it out of `text`.
///
/// # Errors
/// Returns an error if initialization failed, `voice` is not a voice espeak-ng
/// knows, or `text` cannot cross the FFI boundary.
pub fn phonemize(text: &str, voice: &str) -> Result<String> {
    phonemize_mode(text, voice, PHONEME_IPA)
}

/// Like [`phonemize`], but with U+0361 ties inside multi-letter phonemes.
///
/// The English fallback needs ties because misaki's rewrite rules
/// distinguish, say, espeak's tied diphthong `i͡ə` (left alone until the tie
/// is dropped) from the two-vowel sequence `iə` (rewritten per dialect).
///
/// # Errors
/// As [`phonemize`].
pub fn phonemize_tied(text: &str, voice: &str) -> Result<String> {
    phonemize_mode(text, voice, PHONEME_IPA_TIED)
}

fn phonemize_mode(text: &str, voice: &str, mode: c_int) -> Result<String> {
    init()?;

    let voice_c = CString::new(voice).context("voice name contains a NUL byte")?;
    let text_c = CString::new(text).context("text contains a NUL byte")?;

    let _guard = ESPEAK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // SAFETY: `voice_c` outlives the call.
    if unsafe { espeak_ng_SetVoiceByName(voice_c.as_ptr()) } != 0 {
        bail!("espeak-ng has no voice {voice:?}");
    }

    let mut out = String::new();
    let mut ptr = text_c.as_ptr().cast::<c_void>();
    while !ptr.is_null() {
        // SAFETY: espeak advances `ptr` through `text_c` and NULLs it when
        // done; the returned pointer is a static buffer valid until the next
        // call, which cannot happen while the lock is held.
        let clause = unsafe {
            let res = espeak_TextToPhonemes(&raw mut ptr, CHARS_UTF8, mode);
            if res.is_null() {
                break;
            }
            CStr::from_ptr(res).to_string_lossy().into_owned()
        };
        if clause.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&clause);
    }
    Ok(out)
}

/// Whether `dir` holds a complete unpacked data directory. Test-only helper.
#[cfg(test)]
pub(crate) fn data_is_unpacked(dir: &std::path::Path) -> bool {
    dir.join(COMPLETE_MARKER).is_file() && dir.join("espeak-ng-data/phontab").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bundled data must actually be there. A build that silently embedded
    /// an empty archive would fail much later, as unpronounceable text.
    #[test]
    fn the_embedded_data_is_not_empty() {
        assert!(
            ESPEAK_DATA.len() > 100_000,
            "embedded espeak data is only {} bytes",
            ESPEAK_DATA.len()
        );
    }

    #[test]
    fn unpacking_produces_a_usable_data_directory() {
        let root = unpack_data().expect("unpack");
        assert!(data_is_unpacked(&root), "no phontab in {}", root.display());
    }

    /// Unpacking twice must land in the same place and not re-extract, since
    /// every load calls it.
    #[test]
    fn unpacking_is_idempotent() {
        let a = unpack_data().expect("unpack");
        let b = unpack_data().expect("unpack again");
        assert_eq!(a, b);
    }

    #[test]
    fn a_known_word_phonemizes() {
        let ipa = phonemize("hello", "en-us").expect("phonemize");
        assert!(!ipa.trim().is_empty(), "empty IPA for a known word");
    }

    #[test]
    fn an_unknown_voice_is_an_error_not_silence() {
        assert!(phonemize("hello", "not-a-voice").is_err());
    }
}
