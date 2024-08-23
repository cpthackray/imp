use emcee::Guess;
use statrs::distribution::{Continuous, Normal, Uniform};
use statrs::statistics::Distribution;

pub trait PartialPrior {
    fn logprobability(&self, proposed: &f32) -> f64;

    fn initial_guess(&self) -> f64;
}
pub trait Prior {
    fn logprobability(&self, proposal: &Guess) -> f64;

    fn initial_guess(&self) -> Guess;
}

#[derive(Debug, Clone, Copy)]
pub struct IndependentPrior<T: Distribution<f64> + Continuous<f64, f64>> {
    distribution: T,
}

impl<T> PartialPrior for IndependentPrior<T>
where
    T: Distribution<f64> + Continuous<f64, f64>,
{
    fn logprobability(&self, proposed: &f32) -> f64 {
        let p: f64 = *proposed as f64;
        self.distribution.ln_pdf(p)
    }

    fn initial_guess(&self) -> f64 {
        self.distribution.mean().expect("Distribution has no mean?")
    }
}

pub struct BasicPrior {
    partial_priors: Vec<Box<dyn PartialPrior>>,
}

impl BasicPrior {
    fn new(partial_priors: Vec<Box<dyn PartialPrior>>) -> Self {
        Self { partial_priors }
    }
}

impl Prior for BasicPrior {
    fn logprobability(&self, proposal: &Guess) -> f64 {
        self.partial_priors
            .iter()
            .zip(proposal.values.iter())
            .map(|(x, y)| x.logprobability(y))
            .sum()
    }

    fn initial_guess(&self) -> Guess {
        Guess::new(
            &self
                .partial_priors
                .iter()
                .map(|x| x.initial_guess() as f32)
                .collect::<Vec<f32>>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dist_tests() {
        let prior1 = IndependentPrior {
            distribution: Uniform::new(0.0, 1.0).unwrap(),
        };
        assert_eq!(prior1.logprobability(&0.5), 0.0);
        assert_eq!(prior1.logprobability(&1.0), 0.0);
        assert_eq!(prior1.logprobability(&-0.1), f64::NEG_INFINITY)
    }

    #[test]
    fn basic_prior() {
        let basic = BasicPrior::new(vec![
            Box::new(IndependentPrior {
                distribution: Uniform::new(0.0, 1.0).unwrap(),
            }),
            Box::new(IndependentPrior {
                distribution: Uniform::new(0.0, 1.0).unwrap(),
            }),
        ]);
        let proposal1 = Guess::new(&[0.0, 1.0]);
        let proposal2 = Guess::new(&[0.0, -0.1]);
        let proposal3 = Guess::new(&[1.1, 0.0]);
        assert_eq!(basic.logprobability(&proposal1), 0.0);
        assert_eq!(basic.logprobability(&proposal2), f64::NEG_INFINITY);
        assert_eq!(basic.logprobability(&proposal3), f64::NEG_INFINITY)
    }
}
