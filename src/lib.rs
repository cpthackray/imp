mod likelihood;
mod models;
mod priors;

use likelihood::Likelihood;
use models::Model;
use priors::Prior;

use emcee::{Guess, Prob};

pub struct InferenceProblem<P: Prior, L: Likelihood, M: Model> {
    prior: P,
    likelihood: L,
    model: M,
}

impl<P: Prior, L: Likelihood, M: Model> Prob for InferenceProblem<P, L, M> {
    fn lnlike(&self, params: &Guess) -> f32 {
        let prediction = self.model.predict(params);
        self.likelihood.loglikelihood(prediction, &0.0) as f32
    }
    fn lnprior(&self, params: &Guess) -> f32 {
        self.prior.logprobability(params) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert!(true);
    }
}
