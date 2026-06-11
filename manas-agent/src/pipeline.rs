use crate::freshness::{FreshnessChecker, FreshnessReport};
use crate::scraper::Scraper;
use crate::searcher::{SearchResult, Searcher};
use manas_core::{ManasError, Network};

pub struct AgentPipeline {
    pub searcher: Searcher,
    pub scraper: Scraper,
    pub freshness: FreshnessChecker,
}

impl AgentPipeline {
    pub fn new() -> Self {
        AgentPipeline {
            searcher: Searcher::new(),
            scraper: Scraper::new(),
            freshness: FreshnessChecker::new(),
        }
    }

    pub fn search(&self, query: &str) -> Result<Vec<SearchResult>, ManasError> {
        self.searcher.search(query)
    }

    pub fn scrape(&self, url: &str) -> Result<String, ManasError> {
        self.scraper.scrape(url)
    }

    pub fn search_and_scrape(
        &self,
        query: &str,
    ) -> Result<Vec<(SearchResult, String)>, ManasError> {
        let results = self.searcher.search(query)?;
        let mut pages = Vec::new();
        for result in &results {
            match self.scraper.scrape(&result.url) {
                Ok(text) => {
                    pages.push((
                        SearchResult {
                            url: result.url.clone(),
                            title: result.title.clone(),
                            snippet: result.snippet.clone(),
                        },
                        text,
                    ));
                }
                Err(_) => continue,
            }
        }
        Ok(pages)
    }

    pub fn find_stale(&self, network: &Network) -> Vec<u64> {
        self.freshness.find_stale(network)
    }

    pub fn refresh_stale(&self, network: &mut Network) -> Result<FreshnessReport, ManasError> {
        self.freshness
            .refresh_all_stale(network, &self.searcher, &self.scraper)
    }
}

impl Default for AgentPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_creates() {
        let pipeline = AgentPipeline::new();
        assert_eq!(pipeline.searcher.max_results, 5);
    }
}
