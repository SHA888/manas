use manas_core::ManasError;
use scraper::{Html, Selector};
use std::time::Duration;

pub struct SearchResult {
    pub url: String,
    pub title: String,
    pub snippet: String,
}

pub struct Searcher {
    pub max_results: usize,
    pub timeout_secs: u64,
}

impl Searcher {
    pub fn new() -> Self {
        Searcher {
            max_results: 5,
            timeout_secs: 10,
        }
    }

    pub fn new_with_params(max_results: usize, timeout_secs: u64) -> Self {
        Searcher {
            max_results,
            timeout_secs,
        }
    }

    pub fn search(&self, query: &str) -> Result<Vec<SearchResult>, ManasError> {
        let encoded: String = url_encode(query);
        let url = format!("https://html.duckduckgo.com/html/?q={}", encoded);

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(self.timeout_secs))
            .danger_accept_invalid_certs(false)
            .build()
            .map_err(|e| ManasError::NetworkError(e.to_string()))?;

        let resp = client
            .get(&url)
            .header("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .send()
            .map_err(|e| ManasError::NetworkError(e.to_string()))?;

        let html = resp
            .text()
            .map_err(|e| ManasError::NetworkError(e.to_string()))?;

        parse_ddg_results(&html, self.max_results)
    }
}

impl Default for Searcher {
    fn default() -> Self {
        Self::new()
    }
}

fn url_encode(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            ' ' => "+".to_string(),
            _ => format!("%{:02X}", c as u8),
        })
        .collect()
}

fn parse_ddg_results(html: &str, max: usize) -> Result<Vec<SearchResult>, ManasError> {
    let doc = Html::parse_document(html);
    let result_selector =
        Selector::parse(".result").map_err(|e| ManasError::ScraperError(e.to_string()))?;
    let title_selector =
        Selector::parse(".result__title a").map_err(|e| ManasError::ScraperError(e.to_string()))?;
    let snippet_selector =
        Selector::parse(".result__snippet").map_err(|e| ManasError::ScraperError(e.to_string()))?;

    let mut results = Vec::new();

    for result_elem in doc.select(&result_selector) {
        if results.len() >= max {
            break;
        }

        let title_elem = result_elem.select(&title_selector).next();
        let snippet_elem = result_elem.select(&snippet_selector).next();

        let title = title_elem
            .map(|e| e.text().collect::<Vec<_>>().join(" "))
            .unwrap_or_default()
            .trim()
            .to_string();

        let url = title_elem
            .and_then(|e| e.attr("href"))
            .map(extract_ddg_url)
            .unwrap_or_default();

        let snippet = snippet_elem
            .map(|e| e.text().collect::<Vec<_>>().join(" "))
            .unwrap_or_default()
            .trim()
            .to_string();

        if !url.is_empty() {
            results.push(SearchResult {
                url,
                title,
                snippet,
            });
        }
    }

    Ok(results)
}

fn extract_ddg_url(redirect_url: &str) -> String {
    if let Some(start) = redirect_url.find("uddg=") {
        let after = &redirect_url[start + 5..];
        if let Some(end) = after.find('&') {
            url_decode(&after[..end])
        } else {
            url_decode(after)
        }
    } else {
        redirect_url.to_string()
    }
}

fn url_decode(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                result.push(byte as char);
            }
        } else if c == '+' {
            result.push(' ');
        } else {
            result.push(c);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_encode() {
        let result = url_encode("hello world");
        assert_eq!(result, "hello+world");
    }

    #[test]
    fn test_url_decode() {
        let result = url_decode("hello+world%21");
        assert_eq!(result, "hello world!");
    }

    #[test]
    fn test_extract_ddg_url() {
        let result = extract_ddg_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=abc");
        assert_eq!(result, "https://example.com");
    }
}
