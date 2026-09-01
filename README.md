# Super TTS — Kokoro backend

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) as a subprocess backend
for [Super TTS](https://github.com/jorge-menjivar/super-tts). 82M parameters,
24 kHz output, 54 preset voices across eight languages, running on the CPU in
faster than real time, with no system dependencies at all.

## What it is

The Super TTS daemon does not compile model inference in-tree. It discovers
backends on disk and drives each one over a `/v1` HTTP contract on a Unix
socket. This is one such backend: a single static binary that loads Kokoro's
ONNX weights and answers `POST /v1/synthesize` with framed PCM.

```
daemon ──POST /v1/synthesize──▶ this binary ──▶ espeak-ng (phonemes, linked in)
                                            └──▶ kokoro.onnx (waveform)
       ◀── [audio][audio][mark][done] ──────────┘
```

## Requirements

**None at runtime.** The binary carries everything it uses.

espeak-ng used to be a hard dependency that had to come from the system package
manager, which made this the one backend the daemon could not finish installing
on its own: a manifest can declare files to download, it cannot declare
`pacman -S espeak-ng`. It is now built from `vendor/espeak-ng` (a submodule
pinned to 1.52.0) and linked statically, and the phoneme data for the languages
below is compressed into the executable and unpacked at load. Swapping the
subprocess for the linked library left the audio byte-identical across all six
languages.

ONNX Runtime is likewise statically linked, so there is no `libonnxruntime.so`
to install or keep in step with the binary — but it is a C++ library, so the
binary does link `libstdc++`. Present on any Linux desktop; worth knowing for a
minimal container.

### Building

```sh
git clone --recursive https://github.com/jorge-menjivar/super-tts-kokoro
just build-release          # or: cargo build --release
```

The submodule is not optional — `build.rs` fails with instructions if it is
missing (`just submodules` fetches it). Building espeak-ng needs **cmake** and
a **C compiler**; both are build-time only and neither is needed to run the
result. `just cross-build` avoids even those by building in a container.

`just test` runs the suite; the Japanese reference fixture skips unless the
UniDic dictionary is present, so `just test-ja` fetches it once and runs
everything. `just ci` is the full local gate.

**Cross-compiling needs two cmake passes**, and the reason is worth knowing
before touching `build.rs`. espeak-ng compiles its dictionaries by *running* the
`espeak-ng` it has just built, so a cross build hands a foreign ELF to the build
host's shell and dies with `Syntax error: ")" unexpected`. `build.rs` therefore
builds espeak-ng for the host purely to obtain its data, and for the target with
`COMPILE_INTONATIONS=OFF` to obtain the library. That option is why the
submodule is pinned six commits past 1.52.0 — the release includes the data step
unconditionally. Host-built data is safe for either target because dictionaries
and `phontab` are little-endian byte formats, not machine code.

Both release targets therefore build in cross's stock images, and the x86_64
artifact was verified to produce byte-identical audio to a native build.

> One sharp edge: espeak-ng keeps its data directory in a fixed 160-byte buffer
> and silently truncates anything longer, so a very deep build path makes it
> fail to find data it just compiled. `build.rs` checks this and tells you to
> set `CARGO_TARGET_DIR` rather than letting it surface as a confusing
> mid-build "No such file or directory".

### What this is written in

Not pure Rust, and the distinction is worth stating plainly:

| Component | Language | How |
|-----------|----------|-----|
| Server, framing, phonemizer, voice packs | Rust | this repo, ~1,700 lines |
| `ort` bindings | Rust | crate |
| **ONNX Runtime** — all inference | **C++** | statically linked (~21k symbols) |
| **espeak-ng** — grapheme-to-phoneme | **C** | statically linked, 4 hand-written FFI decls |

The Rust is the plumbing: the `/v1` server, the frame codec, tokenization,
segmentation, and the style-vector lookup. Both computational cores — the
neural network and the pronunciation lookup — are C or C++.

A genuinely pure-Rust Kokoro backend would need the `StyleTTS2` + ISTFTNet
architecture ported to Candle or burn (no such port exists) and a Rust g2p to
replace espeak-ng. Both are possible and neither is a weekend.

The Rust g2p half was measured rather than assumed, and the candidates did not
survive it. `piper-plus-g2p` 0.4.0 covers neither Italian nor Hindi, emits
private-use characters outside Kokoro's vocabulary for French and Portuguese,
and silently drops any word its dictionary lacks — 16% of an ordinary technical
sentence. `varna` is a phoneme-inventory database, not a g2p at all. The
dictionary was supposed to buy correct heteronyms, and it does not: CMUdict
stores one pronunciation per word, so it cannot disambiguate them, and it gets
`record` wrong where espeak-ng gets it right. `misaki-rs` is the interesting
one to revisit — it is a port of Kokoro's own reference g2p with a POS tagger,
but it is English-only today.

### Why a second speech engine is involved at all

It is not. `espeak-ng` never synthesizes here — only `espeak_TextToPhonemes` is
called, and every audio path in it is compiled out of the vendored build
(`USE_LIBPCAUDIO`, `USE_SPEECHPLAYER`, `USE_MBROLA`, `USE_LIBSONIC` and
`USE_ASYNC` are all off, which is also what keeps it from linking system audio
libraries). It is used as a pronunciation dictionary, not a speech engine.

That step is unavoidable because **Kokoro has no text input**. Its three model
inputs are `input_ids`, `style`, and `speed`, and `input_ids` are phoneme
tokens. Every Kokoro deployment is therefore a model plus a separate
grapheme-to-phoneme stage; in the reference stack that stage is `misaki`, which
is its own package rather than part of Kokoro.

### Where this differs from the reference stack

`misaki` consults a pronunciation dictionary first and falls back to
`espeak-ng` only for words the dictionary does not have. This backend uses
`espeak-ng` for everything.

The practical difference is homographs and proper nouns — the cases a
dictionary exists to settle. `espeak-ng` handles some of them and not others:

| Input | `espeak-ng` | Correct? |
|-------|-------------|----------|
| "Play the record." | `ɹˈɛkɚd` | yes |
| "He will record it." | `ɹᵻkˈɔːɹd` | yes |
| "I read it yesterday." | `ɹˈiːd` | no — should be `ɹˈɛd` |
| "The pipe is made of lead." | `lˈiːd` | no — should be `lˈɛd` |

Stress-shifting noun/verb pairs come out right; homographs that need to know
the sentence's tense or meaning do not.

If a word comes out *mispronounced* rather than merely spoken poorly, that is
this stage and not the model. The two are cleanly separated behind
`phonemes::phonemize`, so a dictionary lookup could be layered in front of
`espeak-ng` — the same shape `misaki` uses — without touching synthesis.

## Install

```sh
cargo build --release
DIR=~/.local/share/super-tts/backends/app.super-tts.kokoro
mkdir -p "$DIR"
cp target/release/super-tts-backend-kokoro backend.toml "$DIR/"
```

Then select `kokoro-82m` in the Super TTS settings app, or set it in
`~/.config/super-tts/daemon.toml`:

```toml
[synthesis]
    preferred_model  = "kokoro-82m"
    preferred_source = "github.com/jorge-menjivar/super-tts-kokoro"
    active_backend   = "app.super-tts.kokoro"
```

The daemon downloads the weights, voice packs and the Japanese dictionary
itself, verifying each against the SHA-256 pinned in `backend.toml` — 325 MB
for the model, 510 KB per voice and 46 MB for UniDic, about 400 MB in total. The backend never downloads anything: it runs with
`PrivateNetwork=yes` and a read-only backend directory.

## Languages

| Language | phonemizer | Kokoro voices |
|----------|------------|---------------|
| English (American) | espeak-ng `en-us` | 20 |
| English (British) | espeak-ng `en` | 8 |
| **Mandarin** | **`src/zh.rs`** | **8** |
| **Japanese** | **`src/ja.rs`** | **5** |
| Hindi | espeak-ng `hi` | 4 |
| Spanish | espeak-ng `es` | 3 |
| Portuguese (Brazilian) | espeak-ng `pt-br` | 3 |
| Italian | espeak-ng `it` | 2 |
| French | espeak-ng `fr` | 1 |

> British English is `en` and French is `fr`, which is not a typo. The linked
> `espeak_ng_SetVoiceByName` resolves voice *files*, while the `espeak-ng`
> command's `-v` also accepts language tags aliasing them, so `en-gb` and
> `fr-fr` work on the command line and fail in-process. Both spellings produce
> identical phonemes; `every_routed_voice_name_resolves` keeps the table honest.

### Mandarin does not go through espeak-ng

`espeak-ng` writes Mandarin with numeric tone marks (`ni2 hao3`), which have no
token in Kokoro's vocabulary — routing the Mandarin voices through it would
synthesize from a stream of dropped symbols. Kokoro was trained on `misaki`,
whose tones are the arrows `↓ → ↗ ↘` (tokens 169-173), so `src/zh.rs`
reproduces `misaki` instead.

Reproducing it, not approximating it, is the whole point: a phoneme the model
was never trained on is a confident mispronunciation. `data/zh-golden.tsv`
holds 449 of misaki's own answers — natural sentences, the most frequent words
in jieba's dictionary, and a sample of its phrase table — and
`matches_the_reference_g2p` asserts this backend agrees with every one.

Two things that had to be right, and were found by that fixture rather than by
reasoning:

- **Segmentation runs with HMM on**, matching `jieba.lcut(..., cut_all=False)`,
  whose `HMM` parameter defaults to `True`. With it off the segmenter produces
  shorter words, shorter words miss phrase readings, and 音乐会 turns 乐 from
  `yue4` into `le4`.
- **Every phrase is carried, not only the ones that differ** from
  character-by-character reading. Dropping the "redundant" 80% looks free and
  is not: pypinyin resolves a word by longest match over the whole phrase
  dictionary, so 好好学习 has to be able to outrank the shorter 好好 — otherwise
  `hao3hao3` becomes `hao3hao4`.

Pronunciation is contextual, which is why the phrase table exists at all:
character-by-character lookup misreads **5.7% of running text** by word
frequency, and the words it misreads are the commonest in the language (一个 is
`yi2ge4`, not `yi1ge4`).

### Japanese does not either, and picks a different misaki

Which `misaki` matters here, because the obvious guess is wrong.
`misaki.ja.JAG2P` has two paths and defaults to **cutlet** (`fugashi` +
UniDic), not to `pyopenjtalk`. They do not agree, and only one is usable:

| | こんにちは、世界。 |
|---|---|
| cutlet (default) | `koɲɲiʨiβa, sekai.` |
| pyopenjtalk | `koɴniʨiwa, sekai.__-------jj^^___j` |

The pyopenjtalk branch appends its pitch string to the phonemes and emits `ᶄ`,
which has no Kokoro token. So `src/ja.rs` ports cutlet, over `lindera` for the
UniDic morphology — *not* `jpreprocess`, which is an OpenJTalk rewrite and
therefore the wrong lineage.

`data/ja-golden.tsv` holds 745 of misaki's answers and
`matches_the_reference_g2p` replays every one. 742 match. The three that do not
are named in the test with the reason: `misaki` runs against `unidic-lite` and
this runs against the `unidic-mecab-2.1.2` lindera ships, and their lexicons
differ on a few rare entries — 抓れる is ツネレ in one and ツメレ in the other,
before any phoneme is produced. The port itself is exact; the divergence is
visible in the tokenizer output, and the list is asserted to stay under 1% of
the fixture so a real regression cannot hide in it.

### Numbers are spelled out before anything else looks at the text

`misaki` rewrites digits into words *before* segmenting, and skipping that step
does not mispronounce numbers — it loses them. Digits reach the phoneme table,
which has no token for `9`, and 价格是99元 comes out as "the price is ___ yuan".
`src/zh_numbers.rs` and `src/ja_numbers.rs` fix that, reproducing `cn2an` and
`num2kana` respectively.

Neither is a positional expansion, which is the whole difficulty:

| | |
|---|---|
| Mandarin | 1010 is 一千零一十, 20304 is 二万零三百零四, 100000001 is 一亿零一 — 零 marks an interrupted run of zeros and is omitted elsewhere |
| Mandarin | the reading is contextual: `2024年` is 二零二四年 but `99元` is 九十九元, and `1/4` is 四分之一, `50%` is 百分之五十, `5-10个` is 五到十个 |
| Japanese | 300 is さんびゃく, 3000 is さんぜん, 800 is はっぴゃく |
| Japanese | 1000 is せん alone but いっせん inside any larger number — 1336 is せんさんびゃく… while 11336 is いちまん**いっせん**さんびゃく… |

That last one cost a round of measurement: the rule is not "a group equal to
1000" but "any thousands digit of 1, once the number reaches 万". All of it is
replayed against the reference — 5,029 Japanese readings, 4,531 Mandarin
quantity readings, 4,531 digit readings, and the whole contextual pipeline.

**The dictionary is the one thing this backend does not carry.** UniDic unpacks
to 191 MB, far too large to embed the way the espeak and Mandarin tables are,
so it arrives the way the weights do: declared in `backend.toml`, SHA-256
pinned, fetched by the daemon. `src/ja.rs` expands it once, lazily, on the
first Japanese utterance and memory-maps it — so loading costs milliseconds
rather than a 191 MB read, and a user who never speaks Japanese never pays for
it at all.

## Design notes

**Voices are tables, not vectors.** A Kokoro voice pack is `510 x 1 x 256`
float32 — one style vector per token count. StyleTTS2 conditions on utterance
length, so the row is chosen per synthesis rather than at load.

**Streaming is per segment.** Kokoro produces a whole segment's audio in one
forward pass, so the text is split on sentence boundaries and each segment's
frames are sent as they finish. That is what bounds latency to first sound; the
split falls back to word boundaries only for a sentence longer than the window,
since breaking mid-sentence costs the prosody that the per-segment style vector
would otherwise provide.

**Punctuation is preserved deliberately.** `espeak-ng` drops it and signals a
clause break with a newline, which loses the difference between a comma and a
full stop. Kokoro has tokens for both and renders them as different pauses, so
the text is split on punctuation and each mark is re-inserted between the
phoneme runs it separated.

**Everything runs on the CPU.** The manifest declares `supported_devices =
["cpu"]` because no GPU execution provider is built in. Declaring `gpu` would
offer the user a device this backend cannot honor.

## Development

```sh
cargo test     # 65 tests; the phonemizer ones drive the linked espeak-ng and
               # replay 449 reference answers for Mandarin. The Japanese
               # fixture needs the UniDic dictionary, which the daemon fetches
               # rather than the repo carrying; point SUPER_TTS_BACKEND_DIR at
               # a populated backend directory to run it, or it says it is
               # skipping rather than passing vacuously.
cargo clippy --all-targets -- -W clippy::pedantic -D warnings
```

## License

GPL-3.0-only, matching Super TTS. The Kokoro weights are Apache-2.0 and are
downloaded at runtime rather than vendored here.
