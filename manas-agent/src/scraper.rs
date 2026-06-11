use manas_core::ManasError;
use scraper::{Html, Selector};
use std::time::Duration;

pub struct Scraper {
    pub timeout_secs: u64,
}

impl Scraper {
    pub fn new() -> Self {
        Scraper { timeout_secs: 10 }
    }

    pub fn new_with_timeout(timeout_secs: u64) -> Self {
        Scraper { timeout_secs }
    }

    pub fn scrape(&self, url: &str) -> Result<String, ManasError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(self.timeout_secs))
            .danger_accept_invalid_certs(false)
            .build()
            .map_err(|e| ManasError::NetworkError(e.to_string()))?;

        let resp = client
            .get(url)
            .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .send()
            .map_err(|e| ManasError::NetworkError(e.to_string()))?;

        let html = resp
            .text()
            .map_err(|e| ManasError::NetworkError(e.to_string()))?;

        extract_readable_text(&html)
    }
}

impl Default for Scraper {
    fn default() -> Self {
        Self::new()
    }
}

fn extract_readable_text(html: &str) -> Result<String, ManasError> {
    let doc = Html::parse_document(html);

    let body_sel = Selector::parse("body").map_err(|e| ManasError::ScraperError(e.to_string()))?;

    let content_selectors = [
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "td",
        "th",
        "blockquote",
        "pre",
        "code",
        "article",
        "section",
        "main",
        "div.content",
        "div.main",
    ];

    let mut text_parts: Vec<String> = Vec::new();

    if let Some(body) = doc.select(&body_sel).next() {
        for tag in &content_selectors {
            if let Ok(sel) = Selector::parse(tag) {
                for element in body.select(&sel) {
                    if is_excluded(&element) {
                        continue;
                    }
                    let text: String = element.text().collect::<Vec<_>>().join(" ");
                    let trimmed = text.trim().to_string();
                    if !trimmed.is_empty() && trimmed.len() > 3 {
                        text_parts.push(trimmed);
                    }
                }
            }
        }
    }

    if text_parts.is_empty() {
        let all_sel =
            Selector::parse("body").map_err(|e| ManasError::ScraperError(e.to_string()))?;
        if let Some(body) = doc.select(&all_sel).next() {
            let text: String = body.text().collect::<Vec<_>>().join(" ");
            let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
            if !cleaned.is_empty() {
                text_parts.push(cleaned);
            }
        }
    }

    Ok(text_parts.join("\n\n"))
}

fn is_excluded(element: &scraper::ElementRef) -> bool {
    if let Some(class) = element.value().attr("class") {
        let class_lower = class.to_lowercase();
        if class_lower.contains("nav")
            || class_lower.contains("sidebar")
            || class_lower.contains("footer")
            || class_lower.contains("menu")
            || class_lower.contains("comment")
            || class_lower.contains("ad-")
        {
            return true;
        }
    }
    if let Some(id) = element.value().attr("id") {
        let id_lower = id.to_lowercase();
        if id_lower.contains("nav")
            || id_lower.contains("sidebar")
            || id_lower.contains("footer")
            || id_lower.contains("menu")
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_simple_html() {
        let html = "<html><body><p>Hello world</p></body></html>";
        let text = extract_readable_text(html).unwrap();
        assert!(text.contains("Hello world"));
    }

    #[test]
    fn test_extract_strips_scripts() {
        let html = "<html><body><p>Hello</p><script>alert('x')</script><p>world</p></body></html>";
        let text = extract_readable_text(html).unwrap();
        assert!(!text.contains("alert"));
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
    }
}
