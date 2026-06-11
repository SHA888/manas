use crate::scraper::Scraper;
use crate::searcher::Searcher;
use manas_core::{ManasError, Network, Neuron};

pub struct FreshnessReport {
    pub total_stale: usize,
    pub refreshed: usize,
    pub failed: usize,
}

pub struct FreshnessChecker;

impl FreshnessChecker {
    pub fn new() -> Self {
        FreshnessChecker
    }

    pub fn find_stale(&self, network: &Network) -> Vec<u64> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut stale = Vec::new();
        for layer in &network.layers {
            for neuron in &layer.neurons {
                if is_stale(neuron, now) {
                    stale.push(neuron.id);
                }
            }
        }
        stale
    }

    pub fn find_stale_by_category(&self, network: &Network, category: u8) -> Vec<u64> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut stale = Vec::new();
        for layer in &network.layers {
            for neuron in &layer.neurons {
                if neuron.freshness_category == category && is_stale(neuron, now) {
                    stale.push(neuron.id);
                }
            }
        }
        stale
    }

    pub fn refresh_neuron(
        &self,
        _neuron_id: u64,
        network: &mut Network,
        _searcher: &Searcher,
        _scraper: &Scraper,
    ) -> Result<(), ManasError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for layer in &mut network.layers {
            for neuron in &mut layer.neurons {
                if neuron.id == _neuron_id {
                    neuron.last_verified = now;
                    return Ok(());
                }
            }
        }

        Err(ManasError::NeuronNotFound(_neuron_id))
    }

    pub fn refresh_all_stale(
        &self,
        network: &mut Network,
        searcher: &Searcher,
        scraper: &Scraper,
    ) -> Result<FreshnessReport, ManasError> {
        let stale_ids = self.find_stale(network);
        let mut refreshed = 0;
        let mut failed = 0;

        for id in &stale_ids {
            match self.refresh_neuron(*id, network, searcher, scraper) {
                Ok(()) => refreshed += 1,
                Err(_) => failed += 1,
            }
        }

        Ok(FreshnessReport {
            total_stale: stale_ids.len(),
            refreshed,
            failed,
        })
    }
}

impl Default for FreshnessChecker {
    fn default() -> Self {
        Self::new()
    }
}

fn is_stale(neuron: &Neuron, now: u64) -> bool {
    let elapsed = now.saturating_sub(neuron.last_verified);
    let threshold = match neuron.freshness_category {
        0 => u64::MAX,
        1 => 30 * 86400,
        2 => 7 * 86400,
        3 => 86400,
        _ => 30 * 86400,
    };
    elapsed > threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use manas_core::Network;

    #[test]
    fn timeless_never_stale() {
        let mut net = Network::new();
        net.grow_layer(1, 4);
        let n = &mut net.layers[0].neurons[0];
        n.freshness_category = 0;
        n.last_verified = 0;

        let checker = FreshnessChecker::new();
        let stale = checker.find_stale(&net);
        assert!(stale.is_empty());
    }

    #[test]
    fn realtime_gets_stale() {
        let mut net = Network::new();
        net.grow_layer(1, 4);
        let n = &mut net.layers[0].neurons[0];
        n.freshness_category = 3;
        n.last_verified = 0;

        let checker = FreshnessChecker::new();
        let stale = checker.find_stale(&net);
        assert!(!stale.is_empty());
    }

    #[test]
    fn refresh_updates_timestamp() {
        let mut net = Network::new();
        net.grow_layer(1, 4);
        let nid = net.layers[0].neurons[0].id;
        net.layers[0].neurons[0].last_verified = 0;

        let checker = FreshnessChecker::new();
        let searcher = Searcher::new();
        let scraper = Scraper::new();

        checker
            .refresh_neuron(nid, &mut net, &searcher, &scraper)
            .unwrap();
        assert!(net.layers[0].neurons[0].last_verified > 0);
    }
}
