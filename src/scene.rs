use std::path::{Path, PathBuf};

use crate::view::View;


#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug, Ord, PartialOrd)]
pub struct SceneId { inner: usize }
impl SceneId {
    fn new(v: usize) -> Self { Self { inner: v } }
}
pub struct Scene {
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
    pub fn id(&self) -> SceneId { self.id }
    pub fn views(&self) -> &Vec<View> { &self.views }
    pub fn path(&self) -> &Path { self.path.as_path() }
}
