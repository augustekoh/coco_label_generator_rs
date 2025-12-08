use std::path::{Path, PathBuf};
use std::collections::HashMap;

use serde::Serialize;

use crate::instance::{AnnotationMetadata, InstanceId};
use crate::scene::SceneId;


#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug, Serialize, Ord, PartialOrd)]
#[serde(transparent)]
pub struct ViewId { inner: usize }
impl ViewId {
    fn new(v: usize) -> Self { Self { inner: v } }
}
impl<T: num_traits::cast::ToPrimitive + num_traits::sign::Unsigned + num_traits::int::PrimInt> From<T> for ViewId {
    fn from(value: T) -> Self {
        Self { inner: value.to_usize().unwrap() }
    }
}
pub struct View {
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
    pub fn id(&self) -> ViewId { self.id }
    pub fn rgb_path(&self) -> &Path {
        self.rgb_path.as_path()
    }
    pub fn npz_path(&self) -> &Path {
        self.npz_path.as_path()
    }
    pub fn order_v2_csv_path(&self) -> &Path {
        self.order_v2_csv_path.as_path()
    }
}

#[derive(Debug)]
pub struct ViewMetadata {
    scene_id: SceneId,
    id: ViewId,
    rgb_relpath: PathBuf,
    visible: HashMap<InstanceId, AnnotationMetadata>,
    height: usize,
    width: usize,
}
impl ViewMetadata {
    pub fn builder() -> ViewMetadataBuilder { ViewMetadataBuilder::default() }
    pub fn scene_id(&self) -> SceneId { self.scene_id }
    pub fn id(&self) -> ViewId { self.id }
    pub fn visible(&self) -> &HashMap<InstanceId, AnnotationMetadata> { &self.visible }
}
#[derive(Debug, Default)]
pub struct ViewMetadataBuilder {
    pub scene_id: Option<SceneId>,
    pub id: Option<ViewId>,
    pub rgb_relpath: Option<PathBuf>,
    pub visible: Option<HashMap<InstanceId, AnnotationMetadata>>,
    pub height: Option<usize>,
    pub width: Option<usize>,
}
impl ViewMetadataBuilder {
    pub fn build(self) -> ViewMetadata {
        ViewMetadata {
            scene_id: self.scene_id.expect("scene_id is not set."),
            id: self.id.expect("id is not set."),
            rgb_relpath: self.rgb_relpath.expect("rgb_relpath is not set."),
            visible: self.visible.expect("visible is not set."),
            height: self.height.expect("height is not set."),
            width: self.width.expect("width is not set."),
        }
    }
}
#[derive(Debug, Serialize)]
pub struct ViewMetadataSerde {
    id: ViewId,
    file_name: PathBuf,
    height: usize,
    width: usize,
}
impl ViewMetadataSerde {
    pub fn from_view(value: ViewMetadata, id: ViewId) -> Self {
        Self { id, file_name: value.rgb_relpath, height: value.height, width: value.width }
    }
}
