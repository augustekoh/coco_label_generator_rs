use std::path::PathBuf;
use std::string::String;

use clap::{Arg, Command};

use coco_label_generator_rs::Config;


const TRAIN_VAL_TEST_SPLIT_ARGNAME: &str = "train_val_test_split";
const DATA_INPUT_DIR_PATH_ARGNAME: &str = "data_input_dir_path";
const OUTPUT_DIR_PATH_ARGNAME: &str = "output_dir_path";
const SEED_ARGNAME: &str = "seed";

fn main() {
    let mut matches = Command::new("coco_label_generator_rs")
        .arg(Arg::new(TRAIN_VAL_TEST_SPLIT_ARGNAME).value_parser(clap::value_parser!(String)))
        .arg(Arg::new(SEED_ARGNAME).value_parser(clap::value_parser!(u64)))
        .arg(Arg::new(DATA_INPUT_DIR_PATH_ARGNAME).value_parser(clap::value_parser!(PathBuf)))
        .arg(Arg::new(OUTPUT_DIR_PATH_ARGNAME).value_parser(clap::value_parser!(PathBuf)))
        .get_matches();

    let split: String = matches.remove_one(TRAIN_VAL_TEST_SPLIT_ARGNAME).unwrap();
    let config = Config {
        output_dir_path: matches.remove_one(OUTPUT_DIR_PATH_ARGNAME).unwrap(),
        data_input_dir_path: matches.remove_one(DATA_INPUT_DIR_PATH_ARGNAME).unwrap(),
        split: split.parse().unwrap(),
        seed: matches.remove_one(SEED_ARGNAME).unwrap(),
    };

    coco_label_generator_rs::main(config);
}
