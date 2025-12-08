use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, HashSet};
use std::io::Write;

use ndarray::Array2;
use ndarray_npy::NpzReader;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use serde::Serialize;

use crate::bar::MultiProgressBar;
use crate::data_split::TrainValTestSplit;
use crate::instance::{Area, AnnotationMetadata, AnnotationMetadataSerde, Bboxx1y1x2y2, CategoryId,
                      InstancesObjectsValue};
use crate::scene::Scene;
use crate::view::{View, ViewMetadata, ViewMetadataSerde};

mod bar;
mod data_split;
mod instance;
mod scene;
mod view;


#[derive(Serialize)]
pub struct Config {
    pub exec_version: String,
    pub output_dir_path: PathBuf,
    pub data_input_dir_path: PathBuf,
    pub split: TrainValTestSplit,
    pub seed: u64,
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
    scenes.sort_by_key(|e| e.path().to_path_buf());
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

    let mut multi_bar = MultiProgressBar::new(
        train_scenes.len().try_into().unwrap(),
        validation_scenes.len().try_into().unwrap(),
        test_scenes.len().try_into().unwrap(),
    );

    std::fs::create_dir_all(&config.output_dir_path).unwrap();
    std::fs::File::create_new(config.output_dir_path.join("config.json")).unwrap()
        .write(serde_json::to_string_pretty(&config).unwrap().as_bytes()).unwrap();

    multi_bar.start_train_bar(train_scenes.len().try_into().unwrap());
    generate_json(train_scenes, config.output_dir_path.join("train.json"), &mut rng, multi_bar.inc_train_callback());
    multi_bar.finish_train_bar();
    multi_bar.start_val_bar(validation_scenes.len().try_into().unwrap());
    generate_json(validation_scenes, config.output_dir_path.join("valid.json"), &mut rng, multi_bar.inc_val_callback());
    multi_bar.finish_val_bar();
    multi_bar.start_test_bar(test_scenes.len().try_into().unwrap());
    generate_json(test_scenes, config.output_dir_path.join("test.json"), &mut rng, multi_bar.inc_test_callback());
    multi_bar.finish_test_bar();
    multi_bar.finish();
}

fn generate_json<T, U>(scenes: Vec<Scene>, out_json_path: PathBuf, rng: &mut T, bar_callback: U)
where
    T: rand::Rng,
    U: Fn() -> () + std::marker::Sync,
{
    assert!(!out_json_path.exists());
    let views_metadata = Arc::new(Mutex::new(vec![]));
    rayon::ThreadPoolBuilder::new().build().unwrap().install(|| {
        scenes
            .par_iter().panic_fuse()
            .map(|s| derive_view_metadata(s, Arc::clone(&views_metadata)))
            .for_each(|_| { bar_callback() });
    });
    let mut annotations_metadata_serde = vec![];
    let mut views_metadata_serde = vec![];
    let mut views_list = views_metadata.lock().unwrap();
    views_list.sort_by_key(|v| (v.scene_id(), v.id()));
    views_list.shuffle(rng);
    let mut image_id: usize = 1;  // Start at 1 to be consistent with past format.
    for view in views_list.drain(..) {
        let mut obj_list: Vec<AnnotationMetadata> = view.visible().iter().map(|(_, v)| v.clone()).collect();
        obj_list.sort_by_key(|a| (a.scene_id(), a.view_id(), a.instance_id()));
        obj_list.shuffle(rng);
        for ann in obj_list {
            annotations_metadata_serde.push(
                AnnotationMetadataSerde::from_ann(ann, annotations_metadata_serde.len(), image_id));
        }
        views_metadata_serde.push(ViewMetadataSerde::from_view(view, image_id.into()));
        image_id = image_id.checked_add(1).unwrap();
    }
    views_metadata_serde.shuffle(rng);
    annotations_metadata_serde.shuffle(rng);
    let json_file_content = serde_json::json!({
        "info": {},
        "licenses": [],
        "images": views_metadata_serde,
        "categories": [
            {"supercategory": "background", "id": CategoryId::BACKGROUND, "name": "background"},
            {"supercategory": "foreground", "id": CategoryId::FOREGROUND, "name": "foreground"}
        ],
        "annotations": annotations_metadata_serde
    });
    let mut output_file = std::fs::File::create_new(out_json_path).unwrap();
    output_file.write(serde_json::to_string_pretty(&json_file_content).unwrap().as_bytes()).unwrap();
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
        let areas = areas(&arr);
        for id in visible.iter() {
            let inst_obj_val = InstancesObjectsValue::Object(*id);
            let bbox = bounding_box(&arr, &inst_obj_val);
            let mut ann_builder = AnnotationMetadata::builder();
            ann_builder.scene_id.replace(scene.id());
            ann_builder.view_id.replace(view.id());
            ann_builder.instance_id.replace(*id);
            ann_builder.bbox.replace(bbox);
            ann_builder.category_id.replace(CategoryId::FOREGROUND);
            ann_builder.area.replace(areas[&inst_obj_val]);
            ann_builder.iscrowd.replace(false.into());
            visible_map.insert(*id, ann_builder.build());
        }
        check_count_in_csv(
            &view,
            visible.iter().map(|e| (*e).into()).max().unwrap_or(0)
        );
        let parent_to_remove = view.rgb_path()
            .parent().expect("Expected at least one parent.")
            .parent().unwrap_or(Path::new(""));
        let mut view_builder = ViewMetadata::builder();
        view_builder.rgb_relpath.replace(view.rgb_path().strip_prefix(parent_to_remove).unwrap().to_path_buf());
        view_builder.visible.replace(visible_map);
        view_builder.height.replace(arr.nrows());
        view_builder.width.replace(arr.ncols());
        view_builder.scene_id.replace(scene.id());
        view_builder.id.replace(view.id());
        view_metadata.lock().unwrap().push(view_builder.build());
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
    Bboxx1y1x2y2::builder()
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
        slice_iter.len()  // Intentionally starting at len so that the coords are at pixel intersections, not centers.
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

fn areas(arr: &Array2<InstancesObjectsValue>) -> HashMap<InstancesObjectsValue, Area> {
    let mut counts = HashMap::<_, u64>::new();
    assert!(TryInto::<u64>::try_into(arr.len()).unwrap() <= u64::MAX);
    for value in arr.iter() {
        *counts.entry(*value).or_insert(0) += 1;
    }
    let mut areas = HashMap::new();
    for (k, v) in counts.drain() {
        areas.insert(k, v.into());
    }
    areas
}

#[cfg(test)]
mod tests;
