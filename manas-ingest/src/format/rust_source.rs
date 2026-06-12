use crate::normalizer;

pub fn parse(text: &str) -> String {
    let mut result = String::new();
    let mut in_block_comment = false;

    for line in text.lines() {
        let trimmed = line.trim();

        if let Some(rest) = trimmed.strip_prefix("/*") {
            in_block_comment = true;
            if let Some(inner) = rest.strip_suffix("*/") {
                in_block_comment = false;
                let inner = inner.trim();
                if !inner.is_empty() {
                    result.push_str("note: ");
                    result.push_str(inner);
                    result.push('\n');
                }
            } else {
                let inner = rest.trim();
                if !inner.is_empty() {
                    result.push_str("note: ");
                    result.push_str(inner);
                    result.push('\n');
                }
            }
            continue;
        }
        if in_block_comment {
            if let Some(inner) = trimmed.strip_suffix("*/") {
                let inner = inner.trim();
                if !inner.is_empty() {
                    result.push_str("note: ");
                    result.push_str(inner);
                    result.push('\n');
                }
                in_block_comment = false;
            } else {
                let inner = trimmed.trim();
                if !inner.is_empty() {
                    result.push_str("note: ");
                    result.push_str(inner);
                    result.push('\n');
                }
            }
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("///") {
            let doc = rest.trim();
            if !doc.is_empty() {
                result.push_str("doc: ");
                result.push_str(doc);
                result.push('\n');
            }
        } else if let Some(rest) = trimmed.strip_prefix("//!") {
            let doc = rest.trim();
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
