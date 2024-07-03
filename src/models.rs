use emcee::Guess;

pub struct Prediction {
    observables: Vec<f64>,
    errors: Vec<f64>,
    residual_error: f64,
}

impl Prediction {
    pub fn new(observables: Vec<f64>, errors: Vec<f64>, residual_error: f64) -> Self {
        assert_eq!(observables.len(), errors.len());
        Prediction {
            observables,
            errors,
            residual_error,
        }
    }
}

pub trait Model {
    fn predict(&self, proposal: &Guess) -> Prediction;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constructors() {
        let _ = Prediction::new(vec![0.0, 1.0, 3.0], vec![0.0, 0.0, 0.0], 0.0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_constructors() {
        let _ = Prediction::new(vec![0.0], vec![0.0, 1.0], 0.0);
    }
}
