use std::path::PathBuf;
use std::string::String;

use clap::{Arg, Command};

use coco_label_generator_rs::Config;

const CRATE_VERSION_GIT: &str = git_version::git_version!(args = ["--abbrev=0", "--always", "--dirty=-modified"]);
const TRAIN_VAL_TEST_SPLIT_ARGNAME: &str = "train_val_test_split";
const DATA_INPUT_DIR_PATH_ARGNAME: &str = "data_input_dir_path";
const OUTPUT_DIR_PATH_ARGNAME: &str = "output_dir_path";
const SEED_ARGNAME: &str = "seed";

pub fn get_full_crate_version() -> String {
    format!("{}-git-{}", clap::crate_version!(), CRATE_VERSION_GIT)
}

fn main() {
    let mut matches = Command::new("coco_label_generator_rs")
        .version(get_full_crate_version())
        .arg(Arg::new(TRAIN_VAL_TEST_SPLIT_ARGNAME)
             .value_parser(clap::value_parser!(String))
             .help("E.g., 7:1:2 for 70% training, 10% validation, and 20% testing. No requirement on total sum."))
        .arg(Arg::new(SEED_ARGNAME)
             .value_parser(clap::value_parser!(u64))
             .help("Seed for random shuffling (0 to 2^64 - 1). \
                    The shuffling algorithm used is intended to be portable."))
        .arg(Arg::new(DATA_INPUT_DIR_PATH_ARGNAME).value_parser(clap::value_parser!(PathBuf)))
        .arg(Arg::new(OUTPUT_DIR_PATH_ARGNAME).value_parser(clap::value_parser!(PathBuf)))
        .get_matches();

    let split: String = matches.remove_one(TRAIN_VAL_TEST_SPLIT_ARGNAME).unwrap();
    let config = Config {
        exec_version: get_full_crate_version(),
        output_dir_path: matches.remove_one(OUTPUT_DIR_PATH_ARGNAME).unwrap(),
        data_input_dir_path: matches.remove_one(DATA_INPUT_DIR_PATH_ARGNAME).unwrap(),
        split: split.parse().unwrap(),
        seed: matches.remove_one(SEED_ARGNAME).unwrap(),
    };

    coco_label_generator_rs::main(config);
}
