use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, HashSet};

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use ndarray::Array2;
use ndarray_npy::NpzReader;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::Serialize;


const OBJECT_VIEW_BACKGROUND_VALUE: u8 = 0;

#[derive(Eq, PartialEq, Hash, Clone, Copy)]
enum InstancesObjectsValue {
    Background,
    Object(ObjectViewId),
}
impl From<u8> for InstancesObjectsValue {
    fn from(value: u8) -> Self {
        if value == OBJECT_VIEW_BACKGROUND_VALUE {
            Self::Background
        } else {
            Self::Object(ObjectViewId(value))
        }
    }
}
impl From<f32> for InstancesObjectsValue {
    fn from(value: f32) -> Self {
        assert!(value.is_normal() || value == 0.0);
        let r = value as u8;
        assert_eq!(r as f32, value);
        r.into()
    }
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
struct ObjectViewId(u8);

#[derive(Serialize, Debug)]
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
    order_v2_csv_path: PathBuf,
}
impl View {
    pub fn new(rgb_path: PathBuf) -> Self {
        let id = rgb_path.file_name().unwrap().to_str().unwrap().split("_").next().unwrap();
        let npz_path = rgb_path.parent().unwrap().join(format!("{}.npz", id));
        let order_v2_csv_path = rgb_path.parent().unwrap().join(format!("{}_order_v2.csv", id));
        Self { rgb_path, npz_path, order_v2_csv_path }
    }
    pub fn npz_path(&self) -> &Path {
        self.npz_path.as_path()
    }
    pub fn order_v2_csv_path(&self) -> &Path {
        self.order_v2_csv_path.as_path()
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
    let train_validation_boundary_u = train_validation_boundary as usize;

    let validation_test_boundary = scenes_count_f * config.split.validation_proportion() + train_validation_boundary;
    assert!(validation_test_boundary.is_normal());
    let validation_test_boundary_u = validation_test_boundary as usize;

    let test_scenes = scenes.split_off(validation_test_boundary_u);
    let validation_scenes = scenes.split_off(train_validation_boundary_u);
    let train_scenes = scenes;

    generate_json(train_scenes);
    generate_json(validation_scenes);
    generate_json(test_scenes);
}

fn generate_json(scenes: Vec<Scene>) {
    let views_metadata = Arc::new(Mutex::new(vec![]));
    let bar = ProgressBar::new(scenes.len().try_into().unwrap());
    bar.set_style(
        ProgressStyle::with_template("{spinner} {wide_bar} {pos}/{len} ({percent}) [{per_sec:1} {elapsed}/{eta}]")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "));
    bar.enable_steady_tick(std::time::Duration::from_millis(50));
    rayon::ThreadPoolBuilder::new().num_threads(48).build().unwrap().install(|| {
        scenes
            .par_iter().panic_fuse()
            .progress_with(bar)
            .map(|s| derive_view_metadata(s, Arc::clone(&views_metadata))).for_each(drop);
    });
}

#[derive(Debug)]
struct ViewMetadata {
    rgb_relpath: PathBuf,
    visible: HashMap<ObjectViewId, Bboxx1y1x2y2>,
    height: usize,
    width: usize,
    id: usize,
}
impl ViewMetadata {
    fn new(rgb_relpath: PathBuf, visible: HashMap<ObjectViewId, Bboxx1y1x2y2>, height: usize, width: usize,
           id: usize) ->
        Self
    {
        Self { rgb_relpath, visible, height, width, id }
    }
}

#[derive(Debug)]
struct Bboxx1y1x2y2 {
    x1: usize,
    x2: usize,
    y1: usize,
    y2: usize,
}
impl Bboxx1y1x2y2 {
    fn bulider() -> Bboxx1y1x2y2Builder {
        Bboxx1y1x2y2Builder::default()
    }
}
#[derive(Default, Debug)]
struct Bboxx1y1x2y2Builder {
    x1: Option<usize>,
    x2: Option<usize>,
    y1: Option<usize>,
    y2: Option<usize>,
}
impl Bboxx1y1x2y2Builder {
    pub fn set_x1(mut self, v: usize) -> Self { self.x1 = Some(v); self }
    pub fn set_x2(mut self, v: usize) -> Self { self.x2 = Some(v); self }
    pub fn set_y1(mut self, v: usize) -> Self { self.y1 = Some(v); self }
    pub fn set_y2(mut self, v: usize) -> Self { self.y2 = Some(v); self }
    pub fn build(self) -> Result<Bboxx1y1x2y2, String> {
        let x1 = self.x1.expect("x1 is not set.");
        let x2 = self.x2.expect("x2 is not set.");
        let y1 = self.y1.expect("y1 is not set.");
        let y2 = self.y2.expect("y2 is not set.");
        if x2 < x1 {
            Err(format!("Invalid: x2 < x1. x1: {}, x2: {}", x1, x2))
        } else if y2 < y1 {
            Err(format!("Invalid: y2 < y1. y1: {}, y2: {}", y1, y2))
        } else {
            Ok(Bboxx1y1x2y2 { x1, x2, y1, y2 })
        }
    }
}

fn derive_view_metadata(scene: &Scene, metadata: Arc<Mutex<Vec<ViewMetadata>>>) {
    for view in scene.views() {
        let mut npz = NpzReader::new(std::fs::File::open(view.npz_path()).unwrap()).unwrap();
        let arr_original: Array2<f32> = npz.by_name("instances_objects").unwrap();
        let mut visible = HashSet::new();
        let arr: Array2<InstancesObjectsValue> = arr_original.mapv(|e| {
            let r = e.into();
            if let InstancesObjectsValue::Object(obj) = r {
                visible.insert(obj);
            }
            r
        });
        let mut visible_and_bboxes = HashMap::new();
        for id in visible.into_iter() {
            visible_and_bboxes.insert(id, bounding_box(&arr, &InstancesObjectsValue::Object(id)));
        }
        assert!(check_count_in_csv(&view, visible_and_bboxes.len()));
        let parent_to_remove = view.rgb_path
            .parent().expect("Expected at least one parent.")
            .parent().unwrap_or(Path::new(""));
        {
            let mut lock = metadata.lock().unwrap();
            let len = lock.len();  // TODO: Should the ID be global?
            lock.push(ViewMetadata::new(
                view.rgb_path.strip_prefix(parent_to_remove).unwrap().to_path_buf(),
                visible_and_bboxes,
                arr.nrows(),
                arr.ncols(),
                len,
            ));
        }
    }
}

fn check_count_in_csv(view: &View, expected_count: usize) -> bool {
    let mut csv_reader = csv::Reader::from_reader(std::fs::File::open(view.order_v2_csv_path()).unwrap());
    let mut count_records: usize = 0;
    for rec in csv_reader.records() {
        count_records = count_records.checked_add(1).unwrap();
        if rec.unwrap().len() != expected_count {
            return false;
        }
    };
    return count_records == expected_count;
}

fn bounding_box<T: Eq>(arr: &Array2<T>, target: &T) -> Bboxx1y1x2y2 {
    Bboxx1y1x2y2::bulider()
        .set_x1(find_single_bbox_coord(&arr, &target, 1, true).expect("col min not found."))
        .set_x2(find_single_bbox_coord(&arr, &target, 1, false).expect("col max not found."))
        .set_y1(find_single_bbox_coord(&arr, &target, 0, true).expect("row min not found."))
        .set_y2(find_single_bbox_coord(&arr, &target, 0, false).expect("row max not found."))
        .build().expect("Failed to build bbox.")
}

fn find_single_bbox_coord<T: Eq>(arr: &Array2<T>, target: &T, axis: usize, increasing: bool) -> Result<usize, ()> {
    let mut slice_iter = arr.axis_iter(ndarray::Axis(axis)).enumerate();
    loop {
        let next = if increasing { slice_iter.next() } else { slice_iter.next_back() };
        let (idx, slice) = if let Some(i) = next {
            i
        } else {
            return Err(());
        };
        for v in slice.iter() {
            if v == target {
                return if increasing {
                    Ok(idx)
                } else {
                    Ok(idx.checked_add(1).unwrap())  // So that the coordinates are at pixel intersections, not centers.
                }
            }
        }
    }
}
