use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use ndarray::Array2;
use ndarray_npy::NpzReader;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::Serialize;


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

pub struct TrainValTestSplit {
    train: Proportion,
    validation: Proportion,
}
impl TrainValTestSplit {
    fn train_proportion(&self) -> f64 {
        self.train.value()
    }
    fn validation_proportion(&self) -> f64 {
        self.validation.value()
    }
    fn test_proportion(&self) -> f64 {
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
        let train = Proportion::new(train / total);
        let validation = Proportion::new(validation / total);
        Ok(Self { train, validation })
    }
}

#[derive(Serialize)]
pub struct Config {
    pub output_dir_path: PathBuf,
    pub data_input_dir_path: PathBuf,
    pub split: TrainValTestSplit,
    pub seed: u64,
}

struct Scene {
    views: Vec<View>,
    path: PathBuf,
}
impl Scene {
    pub fn new(path: PathBuf) -> Self {
        let views = std::fs::read_dir(&path).unwrap()
            .filter(|e| {
                let e = e.as_ref().unwrap();
                let f_type = e.file_type().unwrap();
                let f_name = e.file_name().into_string().unwrap();
                assert!(f_type.is_dir() || f_type.is_file());
                f_type.is_file() && f_name.ends_with("_rgb.png")
            })
            .map(|e| View::new(e.unwrap().path()))
            .collect();
        Self { views, path }
    }
    pub fn views(&self) -> &Vec<View> {
        &self.views
    }
}
struct View {
    rgb_path: PathBuf,
    npz_path: PathBuf,
}
impl View {
    pub fn new(rgb_path: PathBuf) -> Self {
        let id = rgb_path.file_name().unwrap().to_str().unwrap().split("_").next().unwrap();
        let npz_path = rgb_path.parent().unwrap().join(format!("{}.npz", id));
        Self { rgb_path, npz_path }
    }
    pub fn npz_path(&self) -> &Path {
        self.npz_path.as_path()
    }
}

pub fn main(config: Config) {
    println!("{}", serde_json::to_string(&config).unwrap());
    let mut scenes = vec![];
    for entry in std::fs::read_dir(&config.data_input_dir_path).unwrap() {
        let entry = entry.unwrap();
        if !entry.file_name().into_string().unwrap().starts_with("scene") {
            continue;
        }
        let f_type = entry.file_type().unwrap();
        if f_type.is_file() {
            continue;
        }
        if !f_type.is_dir() {
            panic!("Not sure what to do with this file type: {:?}", f_type);
        }

        scenes.push(Scene::new(entry.path()));
    }
    scenes.sort_by_key(|e| e.path.clone());
    let mut rng = Pcg64Mcg::seed_from_u64(config.seed);
    scenes.shuffle(&mut rng);

    let scenes_count_f = scenes.len() as f64;
    assert!(scenes_count_f.is_normal());
    let train_validation_boundary = scenes_count_f * config.split.train_proportion();
    assert!(train_validation_boundary.is_normal());
    let train_validation_boundary = train_validation_boundary as usize;

    let validation_test_boundary = scenes_count_f * config.split.validation_proportion();
    assert!(validation_test_boundary.is_normal());
    let validation_test_boundary = validation_test_boundary as usize;

    let test_scenes = scenes.split_off(validation_test_boundary);
    let validation_scenes = scenes.split_off(train_validation_boundary);
    let train_scenes = scenes;

    generate_json(test_scenes);
    generate_json(validation_scenes);
    generate_json(train_scenes);
}

fn generate_json(scenes: Vec<Scene>) {
    let metadata = Arc::new(Mutex::new(vec![]));
    scenes.par_iter().panic_fuse().map(|s| derive_view_metadata(s, Arc::clone(&metadata))).for_each(drop);
}

struct ViewMetadata {
    temp: f32,
}
impl ViewMetadata {
    fn new(temp: f32) -> Self {
        Self { temp }
    }
}

fn derive_view_metadata(scene: &Scene, metadata: Arc<Mutex<Vec<ViewMetadata>>>) {
    for view in scene.views() {
        let mut npz = NpzReader::new(std::fs::File::open(view.npz_path()).unwrap()).unwrap();
        let arr: Array2<f32> = npz.by_name("instances_semantic").unwrap();
        {
            let mut lock = metadata.lock().unwrap();
            lock.push(ViewMetadata::new(arr[[0, 0]]));
            print!("\r{}", lock.len());
        }
    }
}
