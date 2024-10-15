use emcee::Guess;

pub struct Prediction {
    pub observables: Vec<f64>,
    pub errors: Vec<f64>,
    pub residual_error: f64,
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

pub struct InfluenceFunction {
    weights: Vec<Vec<f64>>,
    relative_error: f64,
}

impl InfluenceFunction {
    pub fn new(weights: Vec<Vec<f64>>, relative_error: f64) -> Self {
        Self {
            weights,
            relative_error,
        }
    }
}

impl Model for InfluenceFunction {
    fn predict(&self, proposal: &Guess) -> Prediction {
        let mut vals = vec![0.0; self.weights.len()];
        for (p, ws) in proposal.values.iter().zip(&self.weights) {
            vals = vals
                .iter()
                .zip(ws)
                .map(|(x, y)| x + *p as f64 * y)
                .collect();
        }
        let errors = vals.iter().map(|x| x * self.relative_error).collect();
        Prediction::new(vals, errors, 0.0)
    }
}

pub struct InfluenceFunctionLog {
    weights: Vec<Vec<f64>>,
    relative_error: f64,
}

impl InfluenceFunctionLog {
    pub fn new(weights: Vec<Vec<f64>>, relative_error: f64) -> Self {
        Self {
            weights,
            relative_error,
        }
    }
}

impl Model for InfluenceFunctionLog {
    fn predict(&self, proposal: &Guess) -> Prediction {
        let mut vals = vec![0.0; self.weights.len()];
        for (p, ws) in proposal.values.iter().zip(&self.weights) {
            vals = vals
                .iter()
                .zip(ws)
                .map(|(x, y)| x + 10.0_f32.powf(*p) as f64 * y)
                .collect();
        }
        // let errors = vals.iter().map(|x| x * self.relative_error).collect();
        let errors = vals.iter().map(|_| self.relative_error).collect();
        Prediction::new(vals.iter().map(|x| x.log10()).collect(), errors, 0.0)
    }
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
