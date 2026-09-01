# SPDX-License-Identifier: GPL-3.0-only
# Task runner for the standalone Kokoro backend. Mirrors the recipe names used
# by the main super-tts repo and by the other out-of-tree backends, so `just
# check` means the same thing everywhere.
#
# Two things are specific to this backend:
#
#   * espeak-ng is built from `vendor/espeak-ng`, so a checkout without
#     submodules cannot build. `just submodules` fixes that, and every build
#     recipe depends on it being done.
#   * The Japanese fixture needs the UniDic dictionary the daemon normally
#     fetches. `just test` runs without it and the Japanese test says it is
#     skipping; `just test-ja` fetches it once and runs the whole suite.

unidic_url := "https://github.com/lindera/lindera/releases/download/v6.0.0/lindera-unidic-6.0.0.zip"
unidic_dir := justfile_directory() / "target/test-backend"

# Default: build release
default: build-release

# Fetch the espeak-ng submodule. Safe to re-run.
submodules:
    git submodule update --init --recursive

# Compiles with debug profile. Usage: just build-debug [args]
build-debug *args:
    cargo build {{ args }}

# Compiles with release profile. Usage: just build-release [args]
build-release *args:
    cargo build --release --locked {{ args }}

# Cargo already names the artifact as `backend.toml`'s entrypoint, and the
# release workflow tarballs it under the same name, so a local install and a
# published one stage the same bytes.
#
# Build and stage the binary for Import-from-dir. Usage: just stage [args]
stage *args: (build-release args)
    cp target/release/super-tts-backend-kokoro super-tts-backend-kokoro
    @echo "staged super-tts-backend-kokoro — this directory is now installable with Import from dir"

# Runs a clippy check — mirrors super-tts's lint.
check *args:
    cargo clippy --all-targets {{ args }} -- -W clippy::pedantic -D warnings -D unused_must_use

# Runs a clippy check with JSON message format (consumed by clippy-sarif in CI)
check-json: (check '--message-format=json')

# Apply rustfmt to the whole crate
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# The Japanese reference test skips unless the UniDic dictionary is present;
# see `test-ja`.
#
# Run the test suite. Usage: just test [--verbose]
test *args:
    cargo test --locked {{ args }}

# The daemon fetches this file in production; it is 46 MB and deliberately not
# in the repository.
#
# Fetch the UniDic dictionary so the Japanese fixture runs instead of skipping.
fetch-unidic:
    #!/usr/bin/env bash
    set -euo pipefail
    dest="{{ unidic_dir }}/dictionaries/lindera-unidic-6.0.0.zip"
    if [ -f "$dest" ]; then echo "already present: $dest"; exit 0; fi
    mkdir -p "$(dirname "$dest")"
    curl -fL --retry 3 -o "$dest" "{{ unidic_url }}"
    echo "fetched $dest"

# Run the whole suite including the Japanese reference fixture.
test-ja *args: fetch-unidic
    SUPER_TTS_BACKEND_DIR="{{ unidic_dir }}" cargo test --locked {{ args }}

# cross-rs builds inside a container, so no local cmake or C toolchain is
# needed. aarch64 works because build.rs runs the espeak-ng it just built
# through `CARGO_TARGET_<TRIPLE>_RUNNER`, which cross points at qemu-user — see
# build.rs's comment on compiling the data.
#
# Cross-compile for a target. Usage: just cross-build <target>
cross-build target="x86_64-unknown-linux-gnu":
    cross build --release --locked --target {{ target }}

# --remap-path-prefix keeps report paths relative (src/...), and tests/ is
# excluded so only product code is counted.
#
# Measure coverage, requires cargo-llvm-cov. Usage: just coverage [--html]
coverage *args:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' {{ args }}

# Coverage for CI: write lcov.info and print a summary.
coverage-lcov:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' --lcov --output-path lcov.info
    cargo llvm-cov report --summary-only --ignore-filename-regex 'tests/'

# Needs uv; see the docstrings in tools/ for why the Python versions are
# pinned. Only necessary when upstream pronunciation data changes, since the
# outputs are committed.
#
# Regenerate the Mandarin and Japanese tables from misaki.
regen-tables:
    #!/usr/bin/env bash
    set -euo pipefail
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    uv venv --python 3.12 "$work/.venv"
    VIRTUAL_ENV="$work/.venv" uv pip install --quiet \
        pypinyin==0.55.0 jieba==0.42.1 cn2an==0.5.24 ordered-set==4.1.0 \
        addict==2.4.0 fugashi==1.5.1 unidic-lite==1.0.8 jaconv==0.4.0 mojimoji==0.0.13
    git clone --quiet --depth 1 https://github.com/hexgrad/misaki "$work/misaki"
    PYTHONPATH="$work/misaki" "$work/.venv/bin/python" tools/gen-zh-tables.py
    PYTHONPATH="$work/misaki" "$work/.venv/bin/python" tools/gen-ja-tables.py

# No doctests: this is a binary-only crate, so `cargo test --doc` has no lib
# target.
#
# Full local CI gate: format, lint, build, test.
ci: fmt-check check build-release test
