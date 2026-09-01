// SPDX-License-Identifier: GPL-3.0-only
//! spaCy's English tokenizer, ported so the misaki pipeline can run from raw
//! text.
//!
//! misaki tokenizes with spaCy, and tokenization is pronunciation-relevant:
//! "don't" must split into `do` + `n't`, "(quite" into `(` + `quite`, and
//! "$1,250.99" into `$` + `1,250.99`, or everything downstream sees different
//! words. spaCy's tokenizer is entirely rule-driven — no learned weights —
//! so it ports exactly: `data/en-tokenizer.json.gz` carries the 1,347
//! special-case rules and the effective prefix/suffix/infix/url regexes
//! dumped from the pinned `en_core_web_sm`, and this file reimplements the
//! algorithm of `spacy/tokenizer.pyx` (`_tokenize_affixes`,
//! `_split_affixes`, `_attach_tokens`) over them, oddities included — e.g.
//! an infix match at position 0 is skipped, and a suffix is searched on the
//! prefix-stripped string but sliced off the unstripped one.
//!
//! The judge is `data/en-tokens.tsv`: spaCy's own tokens for the whole
//! golden corpus, which the test replays boundary for boundary.

use std::collections::HashMap;
use std::sync::OnceLock;

const TOKENIZER_GZ: &[u8] = include_bytes!("../data/en-tokenizer.json.gz");

pub struct Tokenizer {
    rules: HashMap<String, Vec<String>>,
    prefix: fancy_regex::Regex,
    suffix: fancy_regex::Regex,
    infix: fancy_regex::Regex,
    url: fancy_regex::Regex,
}

/// Python `re` patterns are almost fancy-regex patterns: the `(?u)` flag is
/// redundant here (Unicode is the default) and 8-digit `\U0002CEAF` escapes
/// must become `\u{2CEAF}`.
fn translate_pattern(pattern: &str) -> String {
    static BIG_U: OnceLock<regex::Regex> = OnceLock::new();
    let big_u = BIG_U
        .get_or_init(|| regex::Regex::new(r"\\U([0-9A-Fa-f]{8})").expect("escape regex compiles"));
    let translated = big_u.replace_all(pattern, |c: &regex::Captures| {
        format!(r"\u{{{}}}", c[1].trim_start_matches('0'))
    });
    translated.replace("(?u)", "")
}

impl Tokenizer {
    fn load() -> Self {
        use std::io::Read;
        let mut json = String::new();
        flate2::read::GzDecoder::new(TOKENIZER_GZ)
            .read_to_string(&mut json)
            .expect("embedded tokenizer data decompresses");
        let raw: serde_json::Value = serde_json::from_str(&json).expect("tokenizer data parses");
        let rules = raw["rules"]
            .as_object()
            .expect("rules is an object")
            .iter()
            .map(|(k, v)| {
                let pieces = v
                    .as_array()
                    .expect("rule value is an array")
                    .iter()
                    .map(|p| p.as_str().expect("rule piece is a string").to_string())
                    .collect();
                (k.clone(), pieces)
            })
            .collect();
        let compile = |name: &str| {
            let pattern = raw[name].as_str().expect("pattern is a string");
            fancy_regex::Regex::new(&translate_pattern(pattern))
                .unwrap_or_else(|e| panic!("the {name} pattern compiles: {e}"))
        };
        Self {
            rules,
            prefix: compile("prefix"),
            suffix: compile("suffix"),
            infix: compile("infix"),
            url: compile("url"),
        }
    }

    pub fn shared() -> &'static Self {
        static SHARED: OnceLock<Tokenizer> = OnceLock::new();
        SHARED.get_or_init(Self::load)
    }

    /// `find_prefix`: byte length of the prefix match, 0 for none.
    fn find_prefix(&self, s: &str) -> usize {
        self.prefix
            .find(s)
            .ok()
            .flatten()
            .map_or(0, |m| m.end() - m.start())
    }

    /// `find_suffix`: byte length of the suffix match, 0 for none.
    fn find_suffix(&self, s: &str) -> usize {
        self.suffix
            .find(s)
            .ok()
            .flatten()
            .map_or(0, |m| m.end() - m.start())
    }

    /// Python's `url_match` is `re.Pattern.match`: anchored at the start.
    fn url_match(&self, s: &str) -> bool {
        self.url
            .find(s)
            .ok()
            .flatten()
            .is_some_and(|m| m.start() == 0)
    }

    /// `_split_affixes`: peel prefixes and suffixes, stopping early when the
    /// remainder is a special case.
    fn split_affixes<'a>(
        &self,
        span: &'a str,
        prefixes: &mut Vec<&'a str>,
        suffixes: &mut Vec<&'a str>,
    ) -> &'a str {
        let mut string = span;
        let mut last_size = 0usize;
        while !string.is_empty() && string.len() != last_size {
            if self.rules.contains_key(string) {
                break;
            }
            last_size = string.len();
            let pre_len = self.find_prefix(string);
            let mut minus_pre = "";
            let mut prefix = "";
            if pre_len != 0 {
                prefix = &string[..pre_len];
                minus_pre = &string[pre_len..];
                if !minus_pre.is_empty() && self.rules.contains_key(minus_pre) {
                    prefixes.push(prefix);
                    string = minus_pre;
                    break;
                }
            }
            // The suffix is *searched* on the prefix-stripped string but
            // *sliced* off the unstripped one — tokenizer.pyx does exactly
            // this, and the asymmetry is observable.
            let suf_len = self.find_suffix(&string[pre_len..]);
            let mut minus_suf = "";
            let mut suffix = "";
            if suf_len != 0 {
                suffix = &string[string.len() - suf_len..];
                minus_suf = &string[..string.len() - suf_len];
                if !minus_suf.is_empty() && self.rules.contains_key(minus_suf) {
                    suffixes.push(suffix);
                    string = minus_suf;
                    break;
                }
            }
            if pre_len != 0 && suf_len != 0 && pre_len + suf_len <= string.len() {
                prefixes.push(prefix);
                suffixes.push(suffix);
                string = &string[pre_len..string.len() - suf_len];
            } else if pre_len != 0 {
                prefixes.push(prefix);
                string = minus_pre;
            } else if suf_len != 0 {
                suffixes.push(suffix);
                string = minus_suf;
            }
        }
        string
    }

    /// `_attach_tokens`: emit prefixes, the (special-cased, url, or
    /// infix-split) core, then the suffixes in reverse.
    fn attach_tokens(
        &self,
        out: &mut Vec<(String, String)>,
        string: &str,
        prefixes: &[&str],
        suffixes: &[&str],
    ) {
        for p in prefixes {
            out.push(((*p).to_string(), String::new()));
        }
        if !string.is_empty() {
            if let Some(pieces) = self.rules.get(string) {
                for piece in pieces {
                    out.push((piece.clone(), String::new()));
                }
            } else if self.url_match(string) {
                out.push((string.to_string(), String::new()));
            } else {
                let matches: Vec<(usize, usize)> = self
                    .infix
                    .find_iter(string)
                    .map(|m| {
                        let m = m.expect("infix match succeeds");
                        (m.start(), m.end())
                    })
                    .collect();
                if matches.is_empty() {
                    out.push((string.to_string(), String::new()));
                } else {
                    let mut start = 0usize;
                    for (infix_start, infix_end) in matches {
                        // tokenizer.pyx skips an infix at the position where
                        // the infix scan began (always 0), not at `start`.
                        if infix_start == 0 {
                            continue;
                        }
                        if infix_start != start {
                            out.push((string[start..infix_start].to_string(), String::new()));
                        }
                        if infix_start != infix_end {
                            out.push((string[infix_start..infix_end].to_string(), String::new()));
                        }
                        start = infix_end;
                    }
                    if start < string.len() {
                        out.push((string[start..].to_string(), String::new()));
                    }
                }
            }
        }
        for s in suffixes.iter().rev() {
            out.push(((*s).to_string(), String::new()));
        }
    }

    fn span_tokenize(&self, span: &str, out: &mut Vec<(String, String)>) {
        if let Some(pieces) = self.rules.get(span) {
            for piece in pieces {
                out.push((piece.clone(), String::new()));
            }
            return;
        }
        let mut prefixes = Vec::new();
        let mut suffixes = Vec::new();
        let rest = self.split_affixes(span, &mut prefixes, &mut suffixes);
        self.attach_tokens(out, rest, &prefixes, &suffixes);
    }

    /// Tokenize `text` into `(token, trailing_whitespace)` pairs, following
    /// `_tokenize_affixes`: spans between single spaces are tokenized, a
    /// single space becomes the previous token's whitespace, and any other
    /// whitespace run becomes a token of its own.
    pub fn tokenize(&self, text: &str) -> Vec<(String, String)> {
        let chars: Vec<char> = text.chars().collect();
        let mut out: Vec<(String, String)> = Vec::new();
        if chars.is_empty() {
            return out;
        }
        let mut in_ws = chars[0].is_whitespace();
        let mut start = 0usize;
        for (i, &uc) in chars.iter().enumerate() {
            if uc.is_whitespace() != in_ws {
                if start < i {
                    let span: String = chars[start..i].iter().collect();
                    self.span_tokenize(&span, &mut out);
                }
                if uc == ' ' {
                    if let Some(last) = out.last_mut() {
                        last.1 = " ".to_string();
                    }
                    start = i + 1;
                } else {
                    start = i;
                }
                in_ws = !in_ws;
            }
        }
        if start < chars.len() {
            let span: String = chars[start..].iter().collect();
            self.span_tokenize(&span, &mut out);
            if chars[chars.len() - 1] == ' '
                && !in_ws
                && let Some(last) = out.last_mut()
            {
                last.1 = " ".to_string();
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replays spaCy's own token boundaries and whitespace for the whole
    /// golden corpus.
    #[test]
    fn matches_spacy_tokenization() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/en-tokens.tsv");
        let data = std::fs::read_to_string(path).expect("token fixture exists");
        let tok = Tokenizer::shared();
        let mut bad = Vec::new();
        let mut total = 0;
        for line in data.lines() {
            let (text, json) = line.split_once('\t').expect("two columns");
            let expected: Vec<(String, String, String)> =
                serde_json::from_str(json).expect("token json parses");
            let expected: Vec<(String, String)> =
                expected.into_iter().map(|(t, _, ws)| (t, ws)).collect();
            let actual = tok.tokenize(text);
            total += 1;
            if actual != expected && bad.len() < 5 {
                bad.push(format!(
                    "{text:?}\n  spacy: {expected:?}\n  ours : {actual:?}"
                ));
            }
        }
        assert!(
            bad.is_empty(),
            "tokenization diverges on {} of {total} rows:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }
}
