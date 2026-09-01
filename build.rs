// SPDX-License-Identifier: GPL-3.0-only
//! Builds the vendored espeak-ng and embeds the phoneme data it needs.
//!
//! The backend used to shell out to a system `espeak-ng`, which made it the one
//! backend that could not be installed by the daemon alone: the manifest can
//! declare files to download, but it cannot declare `pacman -S espeak-ng`. Here
//! espeak-ng is built from `vendor/espeak-ng` (a submodule pinned to
//! `1.52.0-6-g88e8ddb4`) and linked statically, so the shipped binary carries
//! its own grapheme-to-phoneme engine.
//!
//! Two decisions worth recording:
//!
//! * **Everything audio is compiled out.** espeak-ng can synthesize speech; we
//!   want only its phonemizer. `USE_LIBPCAUDIO`/`USE_SPEECHPLAYER`/`USE_MBROLA`
//!   /`USE_LIBSONIC`/`USE_ASYNC` are all off, which drops every optional system
//!   library. Leaving `USE_LIBPCAUDIO` on is not merely wasteful — it links
//!   against a system `libpcaudio` and reintroduces exactly the kind of
//!   host dependency this file exists to remove.
//! * **The phoneme data is trimmed and embedded**, not installed beside the
//!   binary. A backend should carry everything it uses, so the data for the
//!   languages this backend declares is compressed into the executable and
//!   unpacked at load. See `src/espeak.rs`.
//!
//! # Compiling the data
//!
//! espeak-ng has no data files to install: dictionaries, `phontab` and
//! `phondata` are *compiled*, by running the `espeak-ng` binary that was just
//! built (`--compile-intonations`, `--compile-phonemes`, `--compile=<lang>`).
//! Its `cmake/data.cmake` does that for all ~130 languages, and its top-level
//! `CMakeLists.txt` skips the whole file under `NOT CMAKE_CROSSCOMPILING` —
//! unconditionally, so `CMAKE_CROSSCOMPILING_EMULATOR` does not rescue it.
//!
//! So `COMPILE_INTONATIONS` is off in both native and cross builds and
//! [`compile_espeak_data`] drives those same commands directly, for the six
//! languages in [`ESPEAK_DICTS`] rather than all of them.
//!
//! Under a cross build the binary being run is a foreign one, and the way to
//! run a foreign binary is Cargo's own `CARGO_TARGET_<TRIPLE>_RUNNER`. `cross`
//! sets it in every image (`/linux-runner <arch>`, which picks qemu-user when
//! the binary is not native and `exec`s it directly when it is), so
//! `cross build --target aarch64-unknown-linux-gnu` needs nothing else, and a
//! plain host build sees no runner and executes the binary itself.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{fs, io};

/// The languages this backend phonemizes with espeak-ng, as `<dict>_dict` stems.
///
/// Japanese and Mandarin are deliberately not here: Kokoro speaks both, but
/// espeak-ng's output for them falls outside Kokoro's token vocabulary, so they
/// are phonemized by their own engines instead. See `src/phonemes.rs`.
const ESPEAK_DICTS: &[&str] = &["en", "es", "fr", "it", "pt", "hi"];

/// Voice-definition directories those languages need, under `espeak-ng-data/lang`.
///
/// espeak-ng groups voices by language family: `gmw` West Germanic (English),
/// `roa` Romance (Spanish, French, Italian, Portuguese), `inc` Indo-Aryan
/// (Hindi).
const ESPEAK_LANG_DIRS: &[&str] = &["gmw", "roa", "inc"];

/// Data files that are not language-specific and are always required.
const ESPEAK_CORE_FILES: &[&str] = &[
    "phontab",
    "phondata",
    "phonindex",
    "phondata-manifest",
    "intonations",
];

/// espeak-ng stores its data directory in a fixed-size buffer (`N_PATH_HOME`,
/// 160 bytes). A longer path is silently truncated, the truncated path is not a
/// directory, and espeak falls back to a compiled-in default that does not
/// exist — surfacing as a baffling "No such file or directory" mid-build.
/// Failing loudly here is worth the few lines.
const N_PATH_HOME: usize = 160;

/// espeak-ng's data compiler holds its data directory in a fixed-size buffer,
/// so a deep build path is silently truncated and it then fails to find data it
/// has just written. Failing here, with the reason, is worth the few lines.
fn check_path_length(data_dir: &Path) {
    assert!(
        data_dir.as_os_str().len() < N_PATH_HOME,
        "the build path is {} bytes, over espeak-ng's {N_PATH_HOME}-byte limit:\n  {}\n\
         espeak-ng would silently fail to find its own data. Build from a \
         shorter path (e.g. set CARGO_TARGET_DIR=/tmp/kokoro-target).",
        data_dir.as_os_str().len(),
        data_dir.display(),
    );
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = manifest.join("vendor/espeak-ng");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    assert!(
        vendor.join("CMakeLists.txt").exists(),
        "vendor/espeak-ng is empty. It is a git submodule holding the \
         grapheme-to-phoneme engine; fetch it with:\n\n    \
         git submodule update --init --recursive\n"
    );

    let install = cmake::Config::new(&vendor)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("ENABLE_TESTS", "OFF")
        // Off always: the data is compiled below, for the six languages this
        // backend ships instead of all ~130. See the module comment.
        .define("COMPILE_INTONATIONS", "OFF")
        // Every optional system library, off. See the module comment.
        .define("USE_LIBPCAUDIO", "OFF")
        .define("USE_SPEECHPLAYER", "OFF")
        .define("USE_MBROLA", "OFF")
        .define("USE_LIBSONIC", "OFF")
        .define("USE_ASYNC", "OFF")
        // `USE_LIBSONIC=OFF` keeps libsonic out of the binary but does not stop
        // it being downloaded: `cmake/deps.cmake` git-clones
        // github.com/waywardgeek/sonic whenever the *system* has no libsonic,
        // and it does so before `USE_LIBSONIC` is even declared. That is
        // upstream espeak-ng#2273 (PR #2274, open since 2025-09-03), and it
        // silently made this build need the network — which it must not, since
        // the point of vendoring is that a checkout builds on its own.
        // Pre-seeding the two cache variables that fetch is guarded on skips
        // it. Nothing reads the values: `USE_LIBSONIC=OFF` gates the only use
        // (`src/libespeak-ng/CMakeLists.txt`), leaving just a status line.
        .define("SONIC_LIB", "unused-see-build-rs")
        .define("SONIC_INC", "unused-see-build-rs")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .profile("Release")
        .build();

    println!(
        "cargo:rustc-link-search=native={}",
        install.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=espeak-ng");
    // ucd-tools is espeak-ng's own Unicode character-database helper. It is
    // built as a separate archive and is not installed, so it is linked from
    // the build tree.
    println!(
        "cargo:rustc-link-search=native={}",
        out.join("build/src/ucd-tools").display()
    );
    println!("cargo:rustc-link-lib=static=ucd");

    let data_src = compile_espeak_data(&vendor, &out, &install.join("bin/espeak-ng"));
    pack_espeak_data(&data_src, &out.join("espeak-ng-data.tar.gz"));

    println!("cargo:rerun-if-changed=vendor/espeak-ng");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed={}", runner_var());
}

/// The Cargo variable naming the wrapper that can execute a binary built for
/// the target: `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER` and friends.
fn runner_var() -> String {
    let target = std::env::var("TARGET").unwrap();
    format!(
        "CARGO_TARGET_{}_RUNNER",
        target.to_uppercase().replace('-', "_")
    )
}

/// How to invoke a binary built for the target.
///
/// Empty on a host build. Under `cross` this is `/linux-runner <arch>`, which
/// runs the binary under `qemu-user`. The value is a command line, not a single
/// path, so it is split on whitespace the way Cargo splits it.
///
/// A target of a different architecture with no runner cannot work — the data
/// compiler would be handed to the host and fail with `Exec format error`, or
/// worse, a shell's `Syntax error: ")" unexpected` — so say what is missing.
fn target_runner() -> Vec<String> {
    let runner: Vec<String> = std::env::var(runner_var())
        .unwrap_or_default()
        .split_whitespace()
        .map(str::to_owned)
        .collect();

    let target = std::env::var("TARGET").unwrap();
    let host = std::env::var("HOST").unwrap();
    let arch = |triple: &str| triple.split('-').next().unwrap_or_default().to_owned();
    assert!(
        !runner.is_empty() || arch(&target) == arch(&host),
        "building for {target} on {host} needs a way to run {} binaries, \
         because espeak-ng compiles its dictionaries by running itself.\n\
         Build with `cross build --target {target}`, which supplies one, or set \
         {}=\"qemu-{} -L <sysroot>\".",
        arch(&target),
        runner_var(),
        arch(&target),
    );
    runner
}

/// Compile the phoneme tables and dictionaries, returning the data directory.
///
/// A transcription of `cmake/data.cmake` restricted to [`ESPEAK_DICTS`]: lay
/// out the source data espeak-ng expects to find under `ESPEAK_DATA_PATH`, then
/// run the compiler over it. Doing it here rather than in cmake is what makes
/// the cross build work at all — see the module comment.
fn compile_espeak_data(vendor: &Path, out: &Path, espeak_bin: &Path) -> PathBuf {
    let root = out.join("espeak-data");
    let dist = root.join("espeak-ng-data");
    let dict_tmp = root.join("dictsource");
    let dict_src = vendor.join("dictsource");
    check_path_length(&dist);

    // A stale tree would let a removed language survive as a `*_dict` file.
    if root.exists() {
        fs::remove_dir_all(&root).unwrap_or_else(|e| panic!("clearing {}: {e}", root.display()));
    }
    fs::create_dir_all(&dict_tmp)
        .unwrap_or_else(|e| panic!("creating {}: {e}", dict_tmp.display()));
    copy_dir(&vendor.join("espeak-ng-data/lang"), &dist.join("lang"));
    copy_dir(
        &vendor.join("espeak-ng-data/voices/!v"),
        &dist.join("voices/!v"),
    );
    // The compiler reads `phonemes` and everything it includes from the working
    // directory. It is the copy, not the submodule, so a build never writes
    // into a checked-out tree.
    let phsource = root.join("phsource");
    copy_dir(&vendor.join("phsource"), &phsource);

    let runner = target_runner();
    espeak(
        &runner,
        espeak_bin,
        &root,
        &phsource,
        "--compile-intonations",
    );
    espeak(&runner, espeak_bin, &root, &phsource, "--compile-phonemes");

    for lang in ESPEAK_DICTS {
        // `_rules` and `_list` are required; `_listx` (preferring the `extra`
        // variant, as data.cmake does) and `_emoji` only exist for some.
        let mut sources = vec![
            dict_src.join(format!("{lang}_rules")),
            dict_src.join(format!("{lang}_list")),
        ];
        let listx = dict_src.join(format!("extra/{lang}_listx"));
        let listx = if listx.is_file() {
            listx
        } else {
            dict_src.join(format!("{lang}_listx"))
        };
        sources.push(listx);
        sources.push(dict_src.join(format!("{lang}_emoji")));

        for src in &sources {
            if !src.is_file() {
                continue;
            }
            let target = dict_tmp.join(src.file_name().unwrap());
            fs::copy(src, &target).unwrap_or_else(|e| panic!("copying {}: {e}", src.display()));
        }
        espeak(
            &runner,
            espeak_bin,
            &root,
            &dict_tmp,
            &format!("--compile={lang}"),
        );
    }

    dist
}

/// Run the freshly built espeak-ng, through the target runner if there is one.
///
/// `ESPEAK_DATA_PATH` points at the *parent* of `espeak-ng-data`; that is where
/// the compiled tables are written and where later steps read them back.
fn espeak(runner: &[String], bin: &Path, data_root: &Path, cwd: &Path, arg: &str) {
    let (program, prefix) = match runner.split_first() {
        Some((program, rest)) => (Path::new(program), rest),
        None => (bin, &[] as &[String]),
    };
    let mut cmd = Command::new(program);
    cmd.args(prefix);
    if !runner.is_empty() {
        cmd.arg(bin);
    }
    cmd.arg(arg)
        .current_dir(cwd)
        .env("ESPEAK_DATA_PATH", data_root);
    run(&mut cmd, &format!("espeak-ng {arg}"));
}

fn run(cmd: &mut Command, what: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("{what}: could not run: {e}"));
    assert!(status.success(), "{what}: exited with {status}");
}

/// Recursive directory copy. `fs::copy` is the only thing std offers.
fn copy_dir(src: &Path, dest: &Path) {
    copy_dir_inner(src, dest).unwrap_or_else(|e| {
        panic!("copying {} to {}: {e}", src.display(), dest.display());
    });
}

fn copy_dir_inner(src: &Path, dest: &Path) -> io::Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let to = dest.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_inner(&entry.path(), &to)?;
        } else {
            fs::copy(entry.path(), &to)?;
        }
    }
    Ok(())
}

/// Compress the subset of `espeak-ng-data` this backend actually uses.
///
/// The full data directory is 8.7 MB across 364 files, almost all of it
/// dictionaries for languages Kokoro has no voice for. Keeping only the
/// declared languages takes it to ~1.5 MB, and ~674 KB compressed — small
/// enough to live inside the executable.
fn pack_espeak_data(src: &Path, dest: &Path) {
    let tar_gz =
        fs::File::create(dest).unwrap_or_else(|e| panic!("creating {}: {e}", dest.display()));
    let encoder = flate2::write::GzEncoder::new(tar_gz, flate2::Compression::best());
    let mut tar = tar::Builder::new(encoder);

    for name in ESPEAK_CORE_FILES {
        let path = src.join(name);
        assert!(path.is_file(), "espeak-ng data is missing {name}");
        tar.append_path_with_name(&path, name)
            .unwrap_or_else(|e| panic!("packing {name}: {e}"));
    }
    for lang in ESPEAK_DICTS {
        let name = format!("{lang}_dict");
        let path = src.join(&name);
        assert!(
            path.is_file(),
            "espeak-ng compiled no dictionary for the declared language {lang:?}"
        );
        tar.append_path_with_name(&path, &name)
            .unwrap_or_else(|e| panic!("packing {name}: {e}"));
    }
    for dir in ESPEAK_LANG_DIRS {
        let path = src.join("lang").join(dir);
        assert!(path.is_dir(), "espeak-ng data is missing lang/{dir}");
        tar.append_dir_all(format!("lang/{dir}"), &path)
            .unwrap_or_else(|e| panic!("packing lang/{dir}: {e}"));
    }

    tar.into_inner()
        .and_then(flate2::write::GzEncoder::finish)
        .unwrap_or_else(|e| panic!("finishing {}: {e}", dest.display()));
}
