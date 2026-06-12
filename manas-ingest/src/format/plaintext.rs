use crate::normalizer;

pub fn parse(text: &str) -> String {
    normalizer::strip_control(text)
}
