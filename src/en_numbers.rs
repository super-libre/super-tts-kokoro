// SPDX-License-Identifier: GPL-3.0-only
//! English number verbalization, matching Python's `num2words` word for word.
//!
//! misaki's `get_number` does not verbalize numbers itself — it calls
//! `num2words` and splits the result on non-letters, so "1,250" becomes the
//! words `one thousand two hundred and fifty` (the comma num2words prints is
//! split away, the "and" it prints is kept or dropped by flags downstream).
//! Reproducing misaki therefore means reproducing num2words' exact wording:
//! its commas, its hyphens, its British-style "and", and its year style
//! ("nineteen eighty-four", "twenty oh-one"). The test replays
//! `data/en-numwords.tsv`, num2words' own output for 1,025 values.

const LOW: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const TENS: [(u128, &str); 8] = [
    (90, "ninety"),
    (80, "eighty"),
    (70, "seventy"),
    (60, "sixty"),
    (50, "fifty"),
    (40, "forty"),
    (30, "thirty"),
    (20, "twenty"),
];

/// Short-scale thousands groups, largest first. u128 tops out below
/// 10^39, so undecillion (10^36) is the largest group reachable.
const SCALES: [(u128, &str); 12] = [
    (
        1_000_000_000_000_000_000_000_000_000_000_000_000,
        "undecillion",
    ),
    (1_000_000_000_000_000_000_000_000_000_000_000, "decillion"),
    (1_000_000_000_000_000_000_000_000_000_000, "nonillion"),
    (1_000_000_000_000_000_000_000_000_000, "octillion"),
    (1_000_000_000_000_000_000_000_000, "septillion"),
    (1_000_000_000_000_000_000_000, "sextillion"),
    (1_000_000_000_000_000_000, "quintillion"),
    (1_000_000_000_000_000, "quadrillion"),
    (1_000_000_000_000, "trillion"),
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
];

/// `num2words(n)`: "one thousand, nine hundred and eighty-four".
///
/// The joiners come from num2words' merge rules: hyphen inside a tens+unit
/// pair, " and " between a hundreds/thousands group and a remainder under
/// 100, ", " between larger groups.
pub fn cardinal(n: u128) -> String {
    if n < 20 {
        return LOW[usize::try_from(n).expect("n < 20")].to_string();
    }
    if n < 100 {
        let (tens, word) = TENS.iter().find(|(t, _)| n >= *t).expect("20 <= n < 100");
        return if n == *tens {
            (*word).to_string()
        } else {
            format!("{word}-{}", cardinal(n - tens))
        };
    }
    if n < 1_000 {
        let head = format!("{} hundred", cardinal(n / 100));
        return match n % 100 {
            0 => head,
            rem => format!("{head} and {}", cardinal(rem)),
        };
    }
    let (scale, word) = SCALES.iter().find(|(s, _)| n >= *s).expect("n >= 1000");
    let head = format!("{} {word}", cardinal(n / scale));
    match n % scale {
        0 => head,
        rem if rem < 100 => format!("{head} and {}", cardinal(rem)),
        rem => format!("{head}, {}", cardinal(rem)),
    }
}

/// `num2words(n, to='ordinal')`: the last word of the cardinal is replaced
/// by its ordinal form ("eighty-four" -> "eighty-fourth").
pub fn ordinal(n: u128) -> String {
    let words = cardinal(n);
    let (head, last) = words.rsplit_once(' ').unwrap_or(("", &words));
    let (hyphen_head, last) = last.rsplit_once('-').unwrap_or(("", last));
    let last = match last {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    };
    let last = if hyphen_head.is_empty() {
        last
    } else {
        format!("{hyphen_head}-{last}")
    };
    if head.is_empty() {
        last
    } else {
        format!("{head} {last}")
    }
}

/// `num2words(n, to='year')`: "nineteen eighty-four", "twenty hundred",
/// "twenty oh-one"; 00XX, X00X and five-digit years read as cardinals.
pub fn year(n: u128) -> String {
    let (high, low) = (n / 100, n % 100);
    if high == 0 || (high % 10 == 0 && low < 10) || high >= 100 {
        return cardinal(n);
    }
    let low = match low {
        0 => "hundred".to_string(),
        l if l < 10 => format!("oh-{}", cardinal(l)),
        l => cardinal(l),
    };
    format!("{} {low}", cardinal(high))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replays num2words' own output for every value in the committed sweep.
    #[test]
    fn matches_num2words_verbatim() {
        let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/data/en-numwords.tsv");
        let data = std::fs::read_to_string(fixture).expect("fixture exists");
        let mut rows = 0;
        for line in data.lines() {
            let mut cols = line.split('\t');
            let n: u128 = cols.next().unwrap().parse().unwrap();
            let (card, ord, yr) = (
                cols.next().unwrap(),
                cols.next().unwrap(),
                cols.next().unwrap(),
            );
            assert_eq!(cardinal(n), card, "cardinal({n})");
            assert_eq!(ordinal(n), ord, "ordinal({n})");
            assert_eq!(year(n), yr, "year({n})");
            rows += 1;
        }
        assert_eq!(rows, 1025, "the whole sweep was replayed");
    }
}
