extern crate letsbayes;
use emcee;
use statrs::distribution::{Normal, Uniform};

fn main() {
    println!("Regression example...");
    // build normal or uninformed priors
    let prior = letsbayes::priors::BasicPrior::new(vec![
        Box::new(letsbayes::priors::IndependentPrior {
            distribution: Normal::new(1.0, 2.0).unwrap(),
        }),
        Box::new(letsbayes::priors::IndependentPrior {
            distribution: Uniform::new(0.0, 10.0).unwrap(),
        }),
        Box::new(letsbayes::priors::IndependentPrior {
            distribution: Uniform::new(0.0, 1.0).unwrap(),
        }),
    ]);
    // make linear model at points x
    struct LM {
        x: Vec<f64>,
    }
    impl letsbayes::models::Model for LM {
        fn predict(&self, proposal: &emcee::Guess) -> letsbayes::models::Prediction {
            let m = proposal.values[1] as f64;
            let b = proposal.values[0] as f64;
            let err = proposal.values[2] as f64;
            letsbayes::models::Prediction::new(
                self.x.iter().map(|x| m * x + b).collect(),
                self.x.iter().map(|_x| 0.0).collect(),
                err,
            )
        }
    }
    let linear_model = LM {
        x: vec![0., 1.0, 2.0, 3.0, 3.0, 1.0, 3.0],
    };
    // regular least squares evidence
    let obs_values = vec![1.0, 3.0, 5.0, 8.0, 6.0, 3.0, 7.0];
    let mut obs: Vec<Box<dyn letsbayes::likelihood::PartialLikelihood>> = vec![];
    for v in obs_values {
        obs.push(Box::new(letsbayes::likelihood::Observation::new(v, 1.0)))
    }
    let obs = letsbayes::likelihood::ObservationSet::new(obs);

    // make inference problem
    let problem = letsbayes::InferenceProblem::new_unnamed(prior, obs, linear_model, 3);

    // sample posterior and write to file
    let posterior = problem.sample(1000, 8);
    posterior
        .to_csv("./examples/regression/regression.csv", 8000, 100)
        .expect("Posterior write failed.")
}
