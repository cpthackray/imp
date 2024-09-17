extern crate letsbayes;
use emcee;
use statrs::distribution::{Normal, Uniform};

fn main() {
    println!("Influence function example...");
    // build normal or uninformed priors
    let prior = letsbayes::priors::BasicPrior::new(vec![
        Box::new(letsbayes::priors::IndependentPrior {
            distribution: Uniform::new(10.0, 200.0).unwrap(),
        }),
        Box::new(letsbayes::priors::IndependentPrior {
            distribution: Uniform::new(0.0, 10.0).unwrap(),
        }),
    ]);

    let influence = letsbayes::models::InfluenceFunction::new(
        vec![
            vec![1.0, 1.5, 0.01, 0.1, 1.0],
            vec![10., 10., 5.0, 0.1, 1.0],
        ],
        0.3,
    );
    // regular least squares evidence
    let obs_values = vec![150.0, 200.0, 26.0];
    let nd_values = vec![13.0, 2.5];
    let mut obs: Vec<Box<dyn letsbayes::likelihood::PartialLikelihood>> = vec![];
    for v in obs_values {
        obs.push(Box::new(letsbayes::likelihood::Observation::new(v, 0.3)))
    }
    for v in nd_values {
        obs.push(Box::new(letsbayes::likelihood::NondetectObservation::new(
            v, 0.0,
        )))
    }
    let obs = letsbayes::likelihood::ObservationSet::new(obs);

    // make inference problem
    let problem = letsbayes::InferenceProblem::new_unnamed(prior, obs, influence, 2);

    // sample posterior and write to file
    let posterior = problem.sample(100000, 8);
    posterior
        .to_csv("./examples/influence/influence.csv", 8000, 1000)
        .expect("Posterior write failed.")
}
