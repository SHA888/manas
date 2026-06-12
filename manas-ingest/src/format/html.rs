use crate::normalizer;

pub fn parse(text: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;

    while i < chars.len() {
        if !in_tag && !in_script && !in_style && i + 6 < chars.len() {
            let lower: String = chars[i..i + 7].iter().collect::<String>().to_lowercase();
            if lower.starts_with("<script") || lower.starts_with("<style") {
                if lower.starts_with("<script") {
                    in_script = true;
                } else {
                    in_style = true;
                }
                while i < chars.len() && chars[i] != '>' {
                    i += 1;
                }
                if i < chars.len() {
                    i += 1;
                }
                continue;
            }
        }

        if in_script || in_style {
            if i + 8 < chars.len() {
                let closing: String = chars[i..].iter().collect::<String>().to_lowercase();
                if in_script && closing.starts_with("</script>") {
                    in_script = false;
                    i += 9;
                    continue;
                }
                if in_style && closing.starts_with("</style>") {
                    in_style = false;
                    i += 8;
                    continue;
                }
            }
            i += 1;
            continue;
        }

        if chars[i] == '<' {
            in_tag = true;
            i += 1;
            continue;
        }

        if in_tag {
            if chars[i] == '>' {
                in_tag = false;
                i += 1;
                result.push(' ');
                continue;
            }
            i += 1;
            continue;
        }

        if chars[i] == '&' {
            if text[i..].starts_with("&amp;") {
                result.push('&');
                i += 5;
                continue;
            }
            if text[i..].starts_with("&lt;") {
                result.push('<');
                i += 4;
                continue;
            }
            if text[i..].starts_with("&gt;") {
                result.push('>');
                i += 4;
                continue;
            }
            if text[i..].starts_with("&quot;") {
                result.push('"');
                i += 6;
                continue;
            }
            if text[i..].starts_with("&#")
                && let Some(semi) = text[i..].find(';')
            {
                i += semi + 1;
                continue;
            }
            if text[i..].starts_with("&nbsp;") {
                result.push(' ');
                i += 6;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }

    normalizer::normalize(&result)
}
