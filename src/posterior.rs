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
    pub fn to_csv(&self, filename: &str, skip: usize, thinning: usize) -> std::io::Result<()> {
        let mut string_out = "".to_string();
        for parameter_name in self.parameter_names.iter() {
            string_out.push_str(&parameter_name);
            string_out.push(',')
        }
        string_out.pop();
        string_out.push_str("\n");
        for sample in self.samples.iter().skip(skip).step_by(thinning) {
            let mut sample_string: String = sample
                .values
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(",");
            sample_string.push_str("\n");
            string_out.push_str(&sample_string)
        }
        std::fs::write(filename, string_out)?;
        Ok(())
    }
}
