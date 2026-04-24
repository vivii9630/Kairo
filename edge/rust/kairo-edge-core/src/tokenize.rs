//! Identical tokenizer semantics to the Python side's
//! `kairo_retrieval.local._tokenize`: lowercase, whitespace-split,
//! drop empty tokens. Kept trivial on purpose — any improvement here
//! must land in both runtimes at once or the stores drift.

pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_lowercases_and_splits_whitespace() {
        assert_eq!(
            tokenize("Cats  Purr\nWhen\tContent"),
            vec!["cats", "purr", "when", "content"],
        );
    }

    #[test]
    fn tokenize_empty_string_yields_empty() {
        assert_eq!(tokenize(""), Vec::<String>::new());
    }
}
