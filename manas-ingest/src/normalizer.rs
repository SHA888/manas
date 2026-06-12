pub fn normalize(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_space = false;
    for ch in text.chars() {
        if ch.is_control() && ch != '\n' && ch != '\r' && ch != '\t' {
            continue;
        }
        if ch == ' ' || ch == '\t' {
            if prev_space {
                continue;
            }
            prev_space = true;
            result.push(' ');
        } else {
            prev_space = false;
            result.push(ch);
        }
    }
    result.trim().to_string()
}

pub fn strip_control(text: &str) -> String {
    text.chars()
        .filter(|&ch| !ch.is_control() || ch == '\n' || ch == '\r' || ch == '\t')
        .collect()
}
