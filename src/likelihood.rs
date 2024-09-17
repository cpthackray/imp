use crate::models::Prediction;

pub trait PartialLikelihood {
    fn loglikelihood(&self, observable: &f64, prediction_error: &f64, residual_error: &f64) -> f64;
}

pub trait Likelihood {
    fn loglikelihood(&self, prediction: Prediction) -> f64;
}

pub struct Observation {
    value: f64,
    error: f64,
}

impl Observation {
    pub fn new(value: f64, error: f64) -> Self {
        Self { value, error }
    }
}

pub struct NondetectObservation {
    detection_limit: f64,
    error: f64,
}

impl NondetectObservation {
    pub fn new(detection_limit: f64, error: f64) -> Self {
        Self {
            detection_limit,
            error,
        }
    }
}

pub struct ObservationSet {
    pub observations: Vec<Box<dyn PartialLikelihood>>,
}

impl ObservationSet {
    pub fn new(observations: Vec<Box<dyn PartialLikelihood>>) -> Self {
        Self { observations }
    }

    pub fn add(&mut self, observation: Box<dyn PartialLikelihood>) {
        self.observations.push(observation)
    }
}
impl Likelihood for ObservationSet {
    fn loglikelihood(&self, prediction: Prediction) -> f64 {
        let mut ll: f64 = 0.0;
        for (i, (o, e)) in prediction
            .observables
            .into_iter()
            .zip(prediction.errors)
            .enumerate()
        {
            ll += self.observations[i].loglikelihood(&o, &e, &prediction.residual_error)
        }
        ll
    }
}

impl PartialLikelihood for Observation {
    fn loglikelihood(&self, observable: &f64, prediction_error: &f64, residual_error: &f64) -> f64 {
        let total_error = &self.error.powi(2) + prediction_error.powi(2) + residual_error.powi(2);
        -(observable - self.value).powi(2) / total_error
    }
}

impl PartialLikelihood for NondetectObservation {
    fn loglikelihood(&self, observable: &f64, prediction_error: &f64, residual_error: &f64) -> f64 {
        if observable <= &self.detection_limit {
            return 0.0;
        } else {
            let total_error =
                &self.error.powi(2) + prediction_error.powi(2) + residual_error.powi(2);
            -(observable - self.detection_limit).powi(2) / total_error
        }
    }
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
