use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub vocab_size: u32,
}

impl Tokenizer {
    pub fn new() -> Self {
        Tokenizer {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            vocab_size: 0,
        }
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let text = text.to_lowercase();
        let mut ids = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '-' || ch == '\'' {
                current.push(ch);
            } else if !current.is_empty() {
                ids.push(self.learn_token(&current));
                current.clear();
            }
        }
        if !current.is_empty() {
            ids.push(self.learn_token(&current));
        }
        ids
    }

    pub fn learn_token(&mut self, token: &str) -> u32 {
        if let Some(&id) = self.vocab.get(token) {
            return id;
        }
        let id = self.vocab_size;
        self.vocab.insert(token.to_string(), id);
        self.id_to_token.insert(id, token.to_string());
        self.vocab_size += 1;
        id
    }

    pub fn decode(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    pub fn token_count(&self) -> u32 {
        self.vocab_size
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}
