use crate::models::Prediction;

pub trait PartialLikelihood {
    fn loglikelihood(&self, observable: &f64, prediction_error: &f64, residual_error: &f64) -> f64;
}

pub trait Likelihood {
    fn loglikelihood(&self, prediction: Prediction, residual_error: &f64) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constructors() {
        // let _ = Proposal::new(vec![0.0, 1.0, 3.0]);
        let _ = Prediction::new(vec![0.0, 1.0, 3.0], vec![0.0, 0.0, 0.0], 0.0);
    }

    #[test]
    #[should_panic]
    fn test_invalid_constructors() {
        let _ = Prediction::new(vec![0.0], vec![0.0, 1.0], 0.0);
    }
}
