// SPDX-License-Identifier: GPL-3.0-only
//! Arabic numerals to Japanese kana.
//!
//! `misaki` rewrites every run of digits into kana *before* tokenizing, so a
//! backend that skips this step does not merely mispronounce numbers — it drops
//! them. The digits reach the phoneme table, which has no token for `9`, and
//! 価格は99円 comes out as "the price is ___ yen".
//!
//! The readings are not a straight positional expansion; Japanese contracts
//! several of them, and the contractions are the whole difficulty:
//!
//! | | |
//! |---|---|
//! | 300 | さんび ゃく, not さんひゃく |
//! | 600, 800 | ろっぴゃく, はっぴゃく |
//! | 3000, 8000 | さんぜん, はっせん |
//! | 1000 | せん alone, いっせん inside any larger number |
//! | 10000 | いちまん — 万 keeps its いち where 百 and 千 drop theirs |
//!
//! Every one of those is asserted against `misaki`'s own answers for 5,029
//! numbers in `data/ja-numbers.tsv`.

/// Digits one through nine.
const ONES: [&str; 10] = [
    "",
    "いち",
    "に",
    "さん",
    "よん",
    "ご",
    "ろく",
    "なな",
    "はち",
    "きゅう",
];

/// The largest number `misaki` will read. Beyond this its converter returns an
/// English error message, which it then happily splices into the text; this
/// module reads the digits out one by one instead. See [`to_kana`].
const MAX_DIGITS: usize = 9;

/// Read a group of at most four digits.
///
/// The leading `1` of 十, 百 and 千 is not spoken: 100 is ひゃく, never
/// いちひゃく. One thousand is the exception, and the condition is the number
/// as a whole rather than the group: 1000 is せん and 1336 is せんさんびゃく…,
/// but once anything reaches 万 every thousand contracts, so 11336 is
/// いちまん**いっせん**さんびゃく… and 10,000,000 is **いっせん**まん. `contract`
/// carries that decision in.
fn read_group(n: u32, contract: bool) -> String {
    let mut out = String::new();
    let (thousands, hundreds, tens, ones) = (n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10);

    match thousands {
        0 => {}
        1 => out.push_str(if contract { "いっせん" } else { "せん" }),
        3 => out.push_str("さんぜん"),
        8 => out.push_str("はっせん"),
        d => {
            out.push_str(ONES[d as usize]);
            out.push_str("せん");
        }
    }
    match hundreds {
        0 => {}
        1 => out.push_str("ひゃく"),
        3 => out.push_str("さんびゃく"),
        6 => out.push_str("ろっぴゃく"),
        8 => out.push_str("はっぴゃく"),
        d => {
            out.push_str(ONES[d as usize]);
            out.push_str("ひゃく");
        }
    }
    match tens {
        0 => {}
        1 => out.push_str("じゅう"),
        d => {
            out.push_str(ONES[d as usize]);
            out.push_str("じゅう");
        }
    }
    out.push_str(ONES[ones as usize]);
    out
}

/// Convert a run of digits to kana.
///
/// Returns `None` for input that is not digits.
#[must_use]
pub fn to_kana(digits: &str) -> Option<String> {
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() {
        return Some("ゼロ".to_string());
    }
    // `misaki` gives up past nine digits and returns an English sentence, which
    // it then splices into the text as if it were a reading. Reading the digits
    // out individually is a deliberate divergence: it is wrong in the same
    // place, but wrong as Japanese rather than as an English error message.
    if trimmed.len() > MAX_DIGITS {
        return Some(
            trimmed
                .bytes()
                .map(|b| {
                    if b == b'0' {
                        "ゼロ"
                    } else {
                        ONES[(b - b'0') as usize]
                    }
                })
                .collect(),
        );
    }

    let n: u64 = trimmed.parse().ok()?;
    // Anything that reaches 万 contracts every thousand within it.
    let contract = n >= 10_000;
    let mut out = String::new();
    let oku = u32::try_from(n / 100_000_000).ok()?;
    let man = u32::try_from((n / 10_000) % 10_000).ok()?;
    let rest = u32::try_from(n % 10_000).ok()?;

    if oku > 0 {
        out.push_str(&read_group(oku, contract));
        out.push_str("おく");
    }
    if man > 0 {
        out.push_str(&read_group(man, contract));
        out.push_str("まん");
    }
    if rest > 0 {
        out.push_str(&read_group(rest, contract));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every reading `misaki` produces for 5,029 numbers, replayed. The
    /// contractions are the point: an expansion that looks obviously right
    /// gets 300, 1000 and 10,000,000 wrong.
    #[test]
    fn matches_the_reference_converter() {
        let mut checked = 0;
        let mut wrong = Vec::new();
        for line in include_str!("../data/ja-numbers.tsv").lines() {
            let Some((n, expected)) = line.split_once('\t') else {
                continue;
            };
            checked += 1;
            let got = to_kana(n).expect("digits convert");
            if got != expected {
                wrong.push(format!("  {n}: want {expected:?}, got {got:?}"));
            }
        }
        assert!(checked > 5000, "fixture only had {checked} cases");
        assert!(
            wrong.is_empty(),
            "{} of {checked} readings differ:\n{}",
            wrong.len(),
            wrong
                .iter()
                .take(15)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn zero_and_leading_zeros() {
        assert_eq!(to_kana("0").unwrap(), "ゼロ");
        assert_eq!(to_kana("000").unwrap(), "ゼロ");
        assert_eq!(to_kana("007").unwrap(), "なな");
    }

    #[test]
    fn non_digits_are_rejected() {
        assert!(to_kana("").is_none());
        assert!(to_kana("12a").is_none());
        assert!(to_kana("１２").is_none());
    }

    /// Past what the reference will read, the digits are read out one by one
    /// rather than emitting the reference's English error message.
    #[test]
    fn very_long_numbers_are_read_digit_by_digit() {
        let out = to_kana("1234567890").unwrap();
        assert!(out.starts_with("いちにさん"), "{out}");
        assert!(out.ends_with("ゼロ"), "{out}");
    }
}
