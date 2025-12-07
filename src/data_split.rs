use std::str::FromStr;

use serde::Serialize;


#[derive(Serialize, Debug)]
#[serde(transparent)]
struct Proportion {
    inner: f64,
}
impl Proportion {
    fn new(v: f64) -> Self {
        if v.is_normal() && 0.0 <= v && v <= 1.0 {
            Self { inner: v }
        } else {
            panic!("Unexpected input: {:?}", v);
        }
    }
    fn value(&self) -> f64 {
        self.inner
    }
}
impl FromStr for Proportion {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let inner = s.parse::<f64>().map_err(|e| e.to_string())?;
        Ok(Self::new(inner))
    }
}

#[derive(Serialize)]
pub struct TrainValTestSplit {
    train: Proportion,
    validation: Proportion,
}
impl TrainValTestSplit {
    pub fn train_proportion(&self) -> f64 {
        self.train.value()
    }
    pub fn validation_proportion(&self) -> f64 {
        self.validation.value()
    }
    pub fn test_proportion(&self) -> f64 {
        let result = 1.0 - self.train_proportion() - self.validation_proportion();
        assert!(result.is_normal() || result == 0.0);
        assert!(result <= 1.0);
        result
    }
}
impl FromStr for TrainValTestSplit {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut vec = s.split(":").map(str::parse).collect::<Result<Vec<f64>, _>>()
            .map_err(|e| e.to_string())?;
        if vec.len() != 3 {
            return Err(format!("Unexpected length: {}", vec.len()));
        }
        let mut v_iter = vec.drain(..);
        let [Some(train), Some(validation), Some(test), None] =
            [v_iter.next(), v_iter.next(), v_iter.next(), v_iter.next()] else {
            panic!();
        };
        let total = train + validation + test;
        assert!(total.is_normal());
        let train_f = train / total;
        assert!(train_f.is_normal() || train_f == 0.0);
        let train = Proportion::new(train_f);
        let validation_f = validation / total;
        assert!(validation_f.is_normal() || validation_f == 0.0);
        let validation = Proportion::new(validation_f);
        Ok(Self { train, validation })
    }
}
