extern crate letsbayes;
use emcee;
use polars::{error::constants::TRUE, prelude::*};
use statrs::distribution::{Normal, Uniform};

fn main() {
    println!("Influence function from input file example...");
    // build normal or uninformed priors
    let prior = letsbayes::priors::BasicPrior::new(vec![
        Box::new(letsbayes::priors::IndependentPrior {
            distribution: Uniform::new(0.0, 100.0).unwrap(),
        }),
        Box::new(letsbayes::priors::IndependentPrior {
            distribution: Uniform::new(0.0, 100.0).unwrap(),
        }),
    ]);
    let df_csv = CsvReadOptions::default()
        .with_infer_schema_length(None)
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("examples/influence_input/inf_inputs.csv".into()))
        .unwrap()
        .finish()
        .unwrap();
    println!("{}", df_csv);
    let tmp = df_csv
        .clone()
        .select_series(["influence_function_1", "influence_function_2"])
        .unwrap();

    let vec1: Vec<f64> = tmp[0].f64().unwrap().into_no_null_iter().collect();
    let vec2: Vec<f64> = tmp[1].f64().unwrap().into_no_null_iter().collect();

    println!("{:?}, {:?}", vec1, vec2);

    let influence = letsbayes::models::InfluenceFunction::new(vec![vec1, vec2], 0.0);
    // regular least squares evidence

    let obs_values: Vec<f64> = df_csv
        .clone()
        .lazy()
        .filter(col("is_nondetect?").eq(false))
        .collect()
        .unwrap()
        .select_series(["observed concentration"])
        .unwrap()[0]
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    let nd_values: Vec<f64> = df_csv
        .clone()
        .lazy()
        .filter(col("is_nondetect?").eq(true))
        .collect()
        .unwrap()
        .select_series(["detection limit"])
        .unwrap()[0]
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect();

    println!("{:?}, {:?}", obs_values, nd_values);
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
        .to_csv("./examples/influence_input/influence_input.csv", 8000, 1000)
        .expect("Posterior write failed.")
}
