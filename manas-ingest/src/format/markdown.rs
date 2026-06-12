use crate::normalizer;

pub fn parse(text: &str) -> String {
    let mut result = String::new();
    let mut in_code_block = false;

    for line in text.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }

        if in_code_block {
            continue;
        }

        let line = strip_markdown_line(line);
        let line = line.trim();
        if !line.is_empty() {
            result.push_str(line);
            result.push('\n');
        }
    }

    normalizer::normalize(&result)
}

fn strip_markdown_line(line: &str) -> String {
    let mut s = line.to_string();

    if s.starts_with("###") {
        s = s[3..].to_string();
    } else if s.starts_with("##") {
        s = s[2..].to_string();
    } else if s.starts_with('#') {
        s = s[1..].to_string();
    }

    s = strip_images(&s);
    s = strip_links(&s);
    s = strip_inline_code(&s);
    s = strip_bold_italic(&s);

    let trimmed = s.trim();
    if trimmed.starts_with('-') || trimmed.starts_with('*') || trimmed.starts_with('+') {
        s = trimmed[1..].to_string();
    }
    if s.trim()
        .starts_with(|c: char| c.is_ascii_digit() && s.trim().contains('.'))
    {
        let rest = s.trim();
        if let Some(pos) = rest.find('.') {
            s = rest[pos + 1..].to_string();
        }
    }
    if s.trim() == "---" || s.trim() == "***" || s.trim() == "___" {
        s.clear();
    }
    if s.trim().starts_with('>') {
        s = s.trim()[1..].to_string();
    }

    s
}

fn strip_images(s: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if i + 1 < chars.len()
            && chars[i] == '!'
            && chars[i + 1] == '['
            && let Some(close) = s[i + 2..].find(']')
        {
            let alt = &s[i + 2..i + 2 + close];
            result.push_str(alt);
            let after = i + 2 + close + 1;
            if after < s.len()
                && s[after..].starts_with('(')
                && let Some(paren_close) = s[after..].find(')')
            {
                i = after + paren_close + 1;
                continue;
            }
            i = after;
            continue;
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

fn strip_links(s: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if i > 0 && chars[i - 1] == '!' {
            result.push(chars[i]);
            i += 1;
            continue;
        }
        if chars[i] == '['
            && let Some(close) = s[i + 1..].find(']')
        {
            let text = &s[i + 1..i + 1 + close];
            let after = i + 1 + close + 1;
            if after < s.len()
                && s[after..].starts_with('(')
                && let Some(paren_close) = s[after..].find(')')
            {
                result.push_str(text);
                i = after + paren_close + 1;
                continue;
            }
            result.push_str(text);
            i = after;
            continue;
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

fn strip_inline_code(s: &str) -> String {
    let mut result = String::new();
    let mut in_code = false;
    for ch in s.chars() {
        if ch == '`' {
            in_code = !in_code;
            continue;
        }
        if !in_code {
            result.push(ch);
        }
    }
    result
}

fn strip_bold_italic(s: &str) -> String {
    let mut result = String::new();
    let mut in_star = false;
    let mut in_underscore = false;
    for ch in s.chars() {
        match ch {
            '*' => in_star = !in_star,
            '_' => in_underscore = !in_underscore,
            '~' => {}
            _ => result.push(ch),
        }
    }
    result
}
