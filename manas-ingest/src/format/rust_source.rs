use crate::normalizer;

pub fn parse(text: &str) -> String {
    let mut result = String::new();
    let mut in_block_comment = false;

    for line in text.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("/*") {
            in_block_comment = true;
            if trimmed.ends_with("*/") {
                in_block_comment = false;
                let inner = &trimmed[2..trimmed.len() - 2];
                if !inner.is_empty() {
                    result.push_str("note: ");
                    result.push_str(inner.trim());
                    result.push('\n');
                }
            } else if trimmed.len() > 2 {
                let inner = &trimmed[2..];
                if !inner.is_empty() {
                    result.push_str("note: ");
                    result.push_str(inner.trim());
                    result.push('\n');
                }
            }
            continue;
        }
        if in_block_comment {
            if trimmed.ends_with("*/") {
                let inner = &trimmed[..trimmed.len() - 2];
                if !inner.is_empty() {
                    result.push_str("note: ");
                    result.push_str(inner.trim());
                    result.push('\n');
                }
                in_block_comment = false;
            } else {
                let inner = trimmed;
                if !inner.is_empty() {
                    result.push_str("note: ");
                    result.push_str(inner.trim());
                    result.push('\n');
                }
            }
            continue;
        }

        if trimmed.starts_with("///") {
            let doc = &trimmed[3..].trim();
            if !doc.is_empty() {
                result.push_str("doc: ");
                result.push_str(doc);
                result.push('\n');
            }
        } else if trimmed.starts_with("//!") {
            let doc = &trimmed[3..].trim();
            if !doc.is_empty() {
                result.push_str("module doc: ");
                result.push_str(doc);
                result.push('\n');
            }
        } else if trimmed.starts_with("//") {
            continue;
        } else if trimmed.contains("fn ")
            || trimmed.contains("struct ")
            || trimmed.contains("enum ")
            || trimmed.contains("trait ")
            || trimmed.contains("impl ")
            || trimmed.contains("mod ")
        {
            result.push_str("declaration: ");
            result.push_str(trimmed);
            result.push('\n');
        }
    }

    normalizer::normalize(&result)
}
