mod likelihood;
mod models;
mod posterior;
mod priors;

use likelihood::Likelihood;
use models::Model;
use priors::Prior;
use serde::{Deserialize, Serialize};

use emcee::{EnsembleSampler, Guess, Prob};

#[derive(Serialize, Deserialize)]
pub struct InferenceProblem<P: Prior, L: Likelihood, M: Model> {
    pub prior: P,
    pub likelihood: L,
    pub model: M,
    parameter_names: Vec<String>,
    dimension: usize,
}

impl<P: Prior, L: Likelihood, M: Model> InferenceProblem<P, L, M> {
    pub fn new(prior: P, likelihood: L, model: M, parameter_names: Vec<String>) -> Self {
        let dimension = parameter_names.len();
        Self {
            prior,
            likelihood,
            model,
            parameter_names,
            dimension,
        }
    }

    pub fn new_unnamed(prior: P, likelihood: L, model: M, dimension: usize) -> Self {
        let parameter_names = (0..dimension)
            .into_iter()
            .map(|x| format!("p{}", x))
            .collect();
        Self {
            prior,
            likelihood,
            model,
            parameter_names,
            dimension,
        }
    }

    pub fn generate_initial(&self) -> Vec<Guess> {
        self.prior
            .initial_guess()
            .create_initial_guess(self.dimension)
    }
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
    use statrs::assert_almost_eq;

    use super::*;

    #[test]
    fn it_works() {
        assert!(true);
    }

    #[test]
    fn square() {}

    #[test]
    fn sampler() {
        struct Model<'a> {
            x: &'a [f32],
            y: &'a [f32],
        }

        // Linear model y = m * x + c
        impl<'a> Prob for Model<'a> {
            fn lnlike(&self, params: &Guess) -> f32 {
                let m = params[0];
                let c = params[1];

                -0.5 * self
                    .x
                    .iter()
                    .zip(self.y)
                    .map(|(xval, yval)| {
                        let model = m * xval + c;
                        let residual = (yval - model).powf(2.0);
                        residual
                    })
                    .sum::<f32>()
            }

            fn lnprior(&self, _params: &Guess) -> f32 {
                // unimformative priors
                0.0f32
            }
        }
        let initial_guess = Guess::new(&[0.0f32, 0.0f32]);
        let nwalkers = 100;
        let perturbed_guess = initial_guess.create_initial_guess(nwalkers);
        assert_eq!(perturbed_guess.len(), nwalkers);
        let ndim = 2; // m and c

        // Build a linear model y = m * x + c (see above)

        let initial_x = [0.0f32, 1.0f32, 2.0f32];
        let initial_y = [5.0f32, 7.0f32, 9.0f32];

        let model = Model {
            x: &initial_x,
            y: &initial_y,
        };

        let mut sampler =
            emcee::EnsembleSampler::new(nwalkers, ndim, &model).expect("could not create sampler");
        let niterations = 2_000;
        sampler
            .run_mcmc(&perturbed_guess, niterations)
            .expect("error running sampler");
        let samples: Vec<Guess> = sampler.flatchain();
        let m_mean: f32 = samples.iter().map(|g| g.values[0]).sum::<f32>() / samples.len() as f32;
        let b_mean: f32 = samples.iter().map(|g| g.values[1]).sum::<f32>() / samples.len() as f32;
        let tol = 3.0e-1;
        assert_almost_eq!(m_mean as f64, 2.0, tol);
        assert_almost_eq!(b_mean as f64, 5.0, tol);
    }
}
