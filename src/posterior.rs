use emcee::Guess;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct SaveGuess {
    values: Vec<f32>,
}

impl SaveGuess {
    fn from_guess(guess: Guess) -> Self {
        Self {
            values: guess.values,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Posterior {
    parameter_names: Vec<String>,
    samples: Vec<SaveGuess>,
    dimension: usize,
}

impl Posterior {
    pub fn new(parameter_names: Vec<String>, samples: Vec<Guess>) -> Self {
        let dimension = parameter_names.len();
        Self {
            parameter_names,
            samples: samples
                .into_iter()
                .map(|x| SaveGuess::from_guess(x))
                .collect(),
            dimension,
        }
    }
}
