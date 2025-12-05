use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, HashSet};
use std::io::Write;

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use ndarray::Array2;
use ndarray_npy::NpzReader;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::Serialize;


const OBJECT_VIEW_BACKGROUND_VALUE: usize = 0;

#[derive(Eq, PartialEq, Hash, Clone, Copy)]
enum InstancesObjectsValue {
    Background,
    Object(InstanceId),
}
impl From<usize> for InstancesObjectsValue {
    fn from(value: usize) -> Self {
        if value == OBJECT_VIEW_BACKGROUND_VALUE {
            Self::Background
        } else {
            Self::Object(InstanceId::new(value))
        }
    }
}
impl From<f32> for InstancesObjectsValue {
    fn from(value: f32) -> Self {
        assert!(value.is_normal() || value == 0.0);
        let r = value as usize;
        assert_eq!(r as f32, value);
        r.into()
    }
}

impl From<InstanceId> for usize {
    fn from(value: InstanceId) -> Self {
        value.inner.checked_sub(1).unwrap()
    }
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug, Serialize, Ord, PartialOrd)]
#[serde(transparent)]
struct InstanceId { inner: usize }
impl InstanceId {
    fn new(v: usize) -> Self { Self { inner: v } }
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug, Ord, PartialOrd)]
struct SceneId { inner: usize }
impl SceneId {
    fn new(v: usize) -> Self { Self { inner: v } }
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug, Ord, PartialOrd)]
struct ViewId { inner: usize }
impl ViewId {
    fn new(v: usize) -> Self { Self { inner: v } }
}

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
    pub exec_version: String,
    pub output_dir_path: PathBuf,
    pub data_input_dir_path: PathBuf,
    pub split: TrainValTestSplit,
    pub seed: u64,
}

struct Scene {
    id: SceneId,
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
        let re = regex::Regex::new(r"^scene([0-9]+)$").unwrap();
        let cap = re.captures(path.file_name().unwrap().to_str().unwrap()).unwrap();
        assert_eq!(cap.len(), 2, "{:?}", cap);
        let id = SceneId::new(cap[1].parse().unwrap());
        Self { views, path, id }
    }
    pub fn views(&self) -> &Vec<View> {
        &self.views
    }
}
struct View {
    id: ViewId,
    rgb_path: PathBuf,
    npz_path: PathBuf,
    /// One entry per object, regardless of whether the object is visible.
    order_v2_csv_path: PathBuf,
}
impl View {
    pub fn new(rgb_path: PathBuf) -> Self {
        let id = rgb_path.file_name().unwrap().to_str().unwrap().split("_").next().unwrap();
        let npz_path = rgb_path.parent().unwrap().join(format!("{}.npz", id));
        let order_v2_csv_path = rgb_path.parent().unwrap().join(format!("{}_order_v2.csv", id));
        let re = regex::Regex::new(r"^([0-9]+)_rgb\.png$").unwrap();
        let cap = re.captures(rgb_path.file_name().unwrap().to_str().unwrap()).unwrap();
        assert_eq!(cap.len(), 2, "{:?}", cap);
        let id = ViewId::new(cap[1].parse().unwrap());
        Self { id, rgb_path, npz_path, order_v2_csv_path }
    }
    pub fn npz_path(&self) -> &Path {
        self.npz_path.as_path()
    }
    pub fn order_v2_csv_path(&self) -> &Path {
        self.order_v2_csv_path.as_path()
    }
}

pub fn main(config: Config) {
    assert!(!config.output_dir_path.exists());
    println!("{}", serde_json::to_string_pretty(&config).unwrap());
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

    std::fs::create_dir_all(&config.output_dir_path).unwrap();
    generate_json(train_scenes, config.output_dir_path.join("train.json"), &mut rng);
    generate_json(validation_scenes, config.output_dir_path.join("valid.json"), &mut rng);
    generate_json(test_scenes, config.output_dir_path.join("test.json"), &mut rng);
}

fn generate_json<T: rand::Rng>(scenes: Vec<Scene>, out_json_path: PathBuf, rng: &mut T) {
    assert!(!out_json_path.exists());
    let views_metadata = Arc::new(Mutex::new(vec![]));
    let bar = ProgressBar::new(scenes.len().try_into().unwrap());
    bar.set_style(
        ProgressStyle::with_template("{spinner} {wide_bar} {pos}/{len} ({percent}%) \
                                     [rate: {per_sec:2} | elapsed: {elapsed} | eta: {eta}]")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "));
    bar.enable_steady_tick(std::time::Duration::from_millis(50));
    rayon::ThreadPoolBuilder::new().build().unwrap().install(|| {
        scenes
            .par_iter().panic_fuse()
            .map(|s| derive_view_metadata(s, Arc::clone(&views_metadata)))
            .progress_with(bar)
            .for_each(drop);
    });
    let mut annotations_metadata_serde = vec![];
    let mut views_metadata_serde = vec![];
    let mut views_list = views_metadata.lock().unwrap();
    views_list.sort_by_key(|a| (a.scene_id, a.id));
    views_list.shuffle(rng);
    for (view_idx, view) in views_list.drain(..).enumerate() {
        if view_idx == usize::MAX { panic!(); }
        let mut obj_list: Vec<AnnotationMetadata> = view.visible.iter().map(|(_, v)| v.clone()).collect();
        obj_list.sort_by_key(|a| (a.scene_id, a.view_id, a.instance_id));
        obj_list.shuffle(rng);
        for ann in obj_list {
            annotations_metadata_serde.push(AnnotationMetadataSerde::from_ann(
                ann,
                annotations_metadata_serde.len(),
                view_idx,
            ));
        }
        views_metadata_serde.push(ViewMetadataSerde::from_view(view, view_idx.checked_add(1).unwrap()));
    }
    let json_file_content = serde_json::json!({
        "info": {},
        "licenses": [],
        "images": views_metadata_serde,
        "categories": [
            {"supercategory": "background", "id": 0, "name": "background"},
            {"supercategory": "foreground", "id": 1, "name": "foreground"}
        ],
        "annotations": annotations_metadata_serde
    });
    let mut output_file = std::fs::File::create_new(out_json_path).unwrap();
    output_file.write(serde_json::to_string_pretty(&json_file_content).unwrap().as_bytes()).unwrap();
}

#[derive(Clone, Debug)]
struct AnnotationMetadata {
    scene_id: SceneId,
    view_id: ViewId,
    instance_id: InstanceId,
    bbox: Bboxx1y1x2y2,
}
impl AnnotationMetadata {
    fn new(scene_id: SceneId, view_id: ViewId, instance_id: InstanceId, bbox: Bboxx1y1x2y2) -> Self {
        Self { scene_id, view_id, instance_id, bbox }
    }
}
#[derive(Debug, Serialize)]
struct AnnotationMetadataSerde {
    id: usize,
    image_id: usize,
    instance_id: InstanceId,
    bbox: Bboxx1y1x2y2,
}
impl AnnotationMetadataSerde {
    fn from_ann(value: AnnotationMetadata, id: usize, image_id: usize) -> Self {
        Self { id, image_id, instance_id: value.instance_id, bbox: value.bbox }
    }
}

#[derive(Debug)]
struct ViewMetadata {
    scene_id: SceneId,
    id: ViewId,
    rgb_relpath: PathBuf,
    visible: HashMap<InstanceId, AnnotationMetadata>,
    height: usize,
    width: usize,
}
impl ViewMetadata {
    fn new(rgb_relpath: PathBuf, visible: HashMap<InstanceId, AnnotationMetadata>, height: usize, width: usize,
           id: ViewId, scene_id: SceneId) -> Self {
        Self { rgb_relpath, visible, height, width, id, scene_id }
    }
}
#[derive(Debug, Serialize)]
struct ViewMetadataSerde {
    id: usize,
    file_name: PathBuf,
    height: usize,
    width: usize,
}
impl ViewMetadataSerde {
    fn from_view(value: ViewMetadata, id: usize) -> Self {
        Self { id, file_name: value.rgb_relpath, height: value.height, width: value.width }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(into = "Bboxx1y1x2y2Serde")]
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
#[derive(Debug, Serialize)]
struct Bboxx1y1x2y2Serde(f64, f64, f64, f64);
impl From<Bboxx1y1x2y2> for Bboxx1y1x2y2Serde {
    fn from(value: Bboxx1y1x2y2) -> Self {
        let x1 = value.x1 as f64;
        assert_eq!(x1 as usize, value.x1);
        let x2 = value.x2 as f64;
        assert_eq!(x2 as usize, value.x2);
        let y1 = value.y1 as f64;
        assert_eq!(y1 as usize, value.y1);
        let y2 = value.y2 as f64;
        assert_eq!(y2 as usize, value.y2);
        Self(x1, x2, y1, y2)
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

fn derive_view_metadata(scene: &Scene, view_metadata: Arc<Mutex<Vec<ViewMetadata>>>) {
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
        let mut visible_map = HashMap::new();
        for id in visible.iter() {
            let bbox = bounding_box(&arr, &InstancesObjectsValue::Object(*id));
            let ann = AnnotationMetadata::new(scene.id, view.id, *id, bbox);
            visible_map.insert(*id, ann);
        }
        check_count_in_csv(
            &view,
            visible.iter().map(|e| (*e).into()).max().unwrap_or(0)
        );
        let visible_map_clone = visible_map.clone();
        let parent_to_remove = view.rgb_path
            .parent().expect("Expected at least one parent.")
            .parent().unwrap_or(Path::new(""));
        {
            let mut lock = view_metadata.lock().unwrap();
            lock.push(ViewMetadata::new(
                view.rgb_path.strip_prefix(parent_to_remove).unwrap().to_path_buf(),
                visible_map_clone,
                arr.nrows(),
                arr.ncols(),
                view.id,
                scene.id,
            ));
        }
    }
}

fn check_count_in_csv(view: &View, expected_low_bound_on_max: usize) {
    let mut csv_reader = csv::Reader::from_reader(std::fs::File::open(view.order_v2_csv_path()).unwrap());
    let mut count_records: usize = 0;
    let mut count_cols = None;
    for rec in csv_reader.records() {
        count_records = count_records.checked_add(1).unwrap();
        let cols = rec.unwrap().len();
        if let None = count_cols {
            count_cols = Some(cols);
        } else {
            assert_eq!(count_cols.unwrap(), cols);
        }
    };
    assert!(expected_low_bound_on_max <= count_records, "{:?} > {:?}", expected_low_bound_on_max, count_records);
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
    let axis = ndarray::Axis(axis);
    let mut slice_iter = arr.axis_iter(axis);
    let mut idx: usize = if increasing {
        0
    } else {
        // Intentionally starting at len so that the coords are at pixel intersections, not centers.
        slice_iter.len()
    };
    loop {
        let slice = if increasing { slice_iter.next() } else { slice_iter.next_back() };
        for v in slice.ok_or(())?.iter() {
            if v == target {
                return Ok(idx);
            }
        }
        idx = if increasing {
            idx.checked_add(1)
        } else {
            idx.checked_sub(1)
        }.unwrap();
    }
}
