// SPDX-License-Identifier: GPL-3.0-only
//! Arabic numerals to Chinese numerals.
//!
//! `misaki` rewrites digits into hanzi *before* segmenting, so a backend that
//! skips the step does not mispronounce numbers, it loses them: the digits
//! reach the phoneme table, which has no token for `9`, and 价格是99元 is spoken
//! as "the price is ___ yuan".
//!
//! Two readings are needed, and which one applies is contextual:
//!
//! | | |
//! |---|---|
//! | [`to_hanzi`] | 2024 → 二千零二十四 — the number as a quantity |
//! | [`digit_by_digit`] | 2024 → 二零二四 — the digits, as in a year |
//!
//! The quantity reading is not a positional expansion. Chinese inserts 零 where
//! a run of zeros interrupts a number and omits it elsewhere, drops the 一 of
//! 一十 at the start, and collapses the zero that would otherwise fall between
//! a myriad and a thousand. 1010 is 一千零一十, 20304 is 二万零三百零四, and
//! 100000001 is 一亿零一. Those are asserted against `cn2an`'s own answers for
//! 4,531 numbers in `data/zh-numbers-low.tsv`.

/// Digits zero through nine.
const NUMERALS: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];

/// The unit for each power of ten, smallest first. Chinese groups by myriads,
/// so 十/百/千 repeat inside every 万 and 亿 band.
const UNITS: [&str; 16] = [
    "", "十", "百", "千", "万", "十", "百", "千", "亿", "十", "百", "千", "万", "十", "百", "千",
];

/// Read `digits` digit by digit, as a year or an identifier is read.
///
/// A decimal point becomes 点; nothing else is interpreted.
#[must_use]
pub fn digit_by_digit(digits: &str) -> Option<String> {
    if digits.is_empty() {
        return None;
    }
    let mut out = String::new();
    for c in digits.chars() {
        match c {
            '.' => out.push('点'),
            d if d.is_ascii_digit() => out.push_str(NUMERALS[(d as u8 - b'0') as usize]),
            _ => return None,
        }
    }
    Some(out)
}

/// Read `digits` as a quantity.
///
/// Handles an optional sign and one decimal point. Returns `None` for anything
/// that is not a number, or for an integer part longer than the 16-position
/// unit ladder.
#[must_use]
pub fn to_hanzi(digits: &str) -> Option<String> {
    let (sign, rest) = digits.strip_prefix('-').map_or(("", digits), |r| ("负", r));
    let (integer, decimal) = match rest.split_once('.') {
        Some((i, d)) => (i, Some(d)),
        None => (rest, None),
    };
    if integer.is_empty() || !integer.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if decimal.is_some_and(|d| d.is_empty() || !d.bytes().all(|b| b.is_ascii_digit())) {
        return None;
    }

    let mut out = String::from(sign);
    out.push_str(&integer_to_hanzi(integer)?);
    if let Some(d) = decimal {
        out.push('点');
        // The fractional part is always read digit by digit.
        for b in d.bytes() {
            out.push_str(NUMERALS[(b - b'0') as usize]);
        }
    }
    Some(out)
}

/// The quantity reading of a whole number.
fn integer_to_hanzi(integer: &str) -> Option<String> {
    let trimmed = integer.trim_start_matches('0');
    let trimmed = if trimmed.is_empty() { "0" } else { trimmed };
    if trimmed.len() > UNITS.len() {
        return None;
    }

    let len = trimmed.len();
    let mut out = String::new();
    for (i, b) in trimmed.bytes().enumerate() {
        let d = (b - b'0') as usize;
        let position = len - i - 1;
        if d != 0 {
            out.push_str(NUMERALS[d]);
            out.push_str(UNITS[position]);
        } else {
            // A myriad boundary keeps its unit even when its digit is zero, so
            // that 100000000 can still say 亿.
            if position % 4 == 0 {
                out.push_str(NUMERALS[0]);
                out.push_str(UNITS[position]);
            }
            // One 零 stands for any run of zeros.
            if i > 0 && !out.ends_with('零') {
                out.push_str(NUMERALS[0]);
            }
        }
    }

    // Collapse the artefacts the loop above leaves behind.
    let mut s = out
        .replace("零零", "零")
        .replace("零万", "万")
        .replace("零亿", "亿")
        .replace("亿万", "亿");
    s = s.trim_matches('零').to_string();
    s = drop_zero_before_thousand(&s);
    // 一十 is said 十 at the start of a number, but not inside one.
    if let Some(rest) = s.strip_prefix("一十") {
        s = format!("十{rest}");
    }
    if s.is_empty() {
        s = NUMERALS[0].to_string();
    }
    Some(s)
}

/// Remove the 零 between a myriad unit and a thousand: 一亿零一千 is 一亿一千.
///
/// `cn2an` does this with a regex; the crate's regex engine has no lookaround
/// and this needs none, so it is a scan.
fn drop_zero_before_thousand(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        let myriad = matches!(chars[i], '万' | '亿');
        let zero_next = chars.get(i + 1) == Some(&'零');
        let numeral = chars
            .get(i + 2)
            .is_some_and(|c| NUMERALS[1..].contains(&c.to_string().as_str()));
        let thousand = matches!(chars.get(i + 3), Some('千' | '仟'));
        if myriad && zero_next && numeral && thousand {
            out.push(chars[i]);
            i += 2; // skip the 零
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Units that mark the number before them as a measured quantity, which is
/// what makes `5-10个` a range rather than a subtraction. Taken from `cn2an`.
const MEASURE_WORDS: &[&str] = &[
    "千克", "公斤", "厘米", "毫米", "公里", "毫升", "分钟", "小时", "斤", "克", "吨", "米", "升",
    "元", "角", "分", "个", "只", "条", "张", "块", "瓶", "杯", "份", "本", "辆", "台", "匹", "头",
    "位", "亩", "秒", "天", "半",
];

fn re(pattern: &str) -> regex::Regex {
    regex::Regex::new(pattern).expect("static pattern")
}

/// Rewrite every number in `text` the way `cn2an`'s `an2cn` transform does.
///
/// The passes run in `cn2an`'s order, and the order is load-bearing: the date
/// pass has to claim `2024年` before the plain-number pass would read it as a
/// quantity, and the percent pass has to claim `50%` before the `%` is left
/// stranded.
#[must_use]
pub fn transform(text: &str) -> String {
    use std::sync::OnceLock;
    static PATTERNS: OnceLock<[regex::Regex; 6]> = OnceLock::new();
    let p = PATTERNS.get_or_init(|| {
        let measures = MEASURE_WORDS.join("|");
        [
            re(&format!(r"\d+(\.\d+)?-\d+(\.\d+)?(?:{measures})")),
            re(r"(\d{2,4}年)?(\d{1,2}月)?(\d{1,2}日)?"),
            re(r"\d+/\d+"),
            re(r"-?(\d+\.)?\d+%"),
            re(r"\d+℃"),
            re(r"-?(\d+\.)?\d+"),
        ]
    });

    let text = p[0].replace_all(text, |c: &regex::Captures| range(&c[0]));
    let text = p[1].replace_all(&text, |c: &regex::Captures| date(&c[0]));
    let text = p[2].replace_all(&text, |c: &regex::Captures| fraction(&c[0]));
    let text = p[3].replace_all(&text, |c: &regex::Captures| percent(&c[0]));
    let text = p[4].replace_all(&text, |c: &regex::Captures| celsius(&c[0]));
    p[5].replace_all(&text, |c: &regex::Captures| {
        to_hanzi(&c[0]).unwrap_or_else(|| c[0].to_string())
    })
    .into_owned()
}

/// `5-10个` -> `五到十个`. The trailing measure word is carried through.
fn range(matched: &str) -> String {
    let digits_end = matched
        .rfind(|c: char| c.is_ascii_digit())
        .map_or(matched.len(), |i| i + 1);
    let (numbers, unit) = matched.split_at(digits_end);
    let Some((start, end)) = numbers.split_once('-') else {
        return matched.to_string();
    };
    match (to_hanzi(start), to_hanzi(end)) {
        (Some(a), Some(b)) => format!("{a}到{b}{unit}"),
        _ => matched.to_string(),
    }
}

/// A year is read digit by digit and the rest of a date as quantities:
/// `2024年3月15日` is 二零二四年三月十五日, not 二千零二十四年.
fn date(matched: &str) -> String {
    if matched.is_empty() {
        return String::new();
    }
    let year = re(r"\d+年");
    let stage = year.replace_all(matched, |c: &regex::Captures| {
        let digits = &c[0][..c[0].len() - "年".len()];
        digit_by_digit(digits).map_or_else(|| c[0].to_string(), |s| format!("{s}年"))
    });
    re(r"\d+")
        .replace_all(&stage, |c: &regex::Captures| {
            to_hanzi(&c[0]).unwrap_or_else(|| c[0].to_string())
        })
        .into_owned()
}

/// `1/4` -> `四分之一`: the denominator is spoken first.
fn fraction(matched: &str) -> String {
    let Some((numerator, denominator)) = matched.split_once('/') else {
        return matched.to_string();
    };
    match (to_hanzi(numerator), to_hanzi(denominator)) {
        (Some(n), Some(d)) => format!("{d}分之{n}"),
        _ => matched.to_string(),
    }
}

fn percent(matched: &str) -> String {
    let body = matched.trim_end_matches('%');
    to_hanzi(body).map_or_else(|| matched.to_string(), |n| format!("百分之{n}"))
}

fn celsius(matched: &str) -> String {
    let body = matched.trim_end_matches('℃');
    to_hanzi(body).map_or_else(|| matched.to_string(), |n| format!("{n}摄氏度"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay(fixture: &str, f: impl Fn(&str) -> Option<String>) {
        let mut checked = 0;
        let mut wrong = Vec::new();
        for line in fixture.lines() {
            let Some((n, expected)) = line.split_once('\t') else {
                continue;
            };
            checked += 1;
            let got = f(n).unwrap_or_default();
            if got != expected {
                wrong.push(format!("  {n}: want {expected:?}, got {got:?}"));
            }
        }
        assert!(checked > 4000, "fixture only had {checked} cases");
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

    /// The quantity reading, against `cn2an`'s own answers. The 零 placement
    /// rules are the point: an obvious positional expansion gets 1010, 20304
    /// and 100000001 wrong.
    #[test]
    fn quantity_readings_match_the_reference() {
        replay(include_str!("../data/zh-numbers-low.tsv"), to_hanzi);
    }

    #[test]
    fn digit_readings_match_the_reference() {
        replay(
            include_str!("../data/zh-numbers-direct.tsv"),
            digit_by_digit,
        );
    }

    #[test]
    fn the_two_readings_differ_where_it_matters() {
        assert_eq!(to_hanzi("2024").unwrap(), "二千零二十四");
        assert_eq!(digit_by_digit("2024").unwrap(), "二零二四");
    }

    /// The whole pipeline, against `cn2an`'s own answers. Order is what these
    /// cases really check: a year must not be read as a quantity, and a
    /// percentage must not lose its sign.
    #[test]
    fn the_transform_matches_the_reference() {
        let mut checked = 0;
        let mut wrong = Vec::new();
        for line in include_str!("../data/zh-transform.tsv").lines() {
            let Some((text, expected)) = line.split_once('\t') else {
                continue;
            };
            checked += 1;
            let got = transform(text);
            if got != expected {
                wrong.push(format!("  {text}\n    want {expected:?}\n    got  {got:?}"));
            }
        }
        assert!(checked > 30, "fixture only had {checked} cases");
        assert!(
            wrong.is_empty(),
            "{} of {checked} differ:\n{}",
            wrong.len(),
            wrong
                .iter()
                .take(12)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn non_numbers_are_rejected() {
        assert!(to_hanzi("").is_none());
        assert!(to_hanzi("12a").is_none());
        assert!(digit_by_digit("x").is_none());
    }
}
