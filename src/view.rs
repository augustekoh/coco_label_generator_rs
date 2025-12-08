use std::path::PathBuf;
use std::collections::HashMap;

use getset::{Getters, CopyGetters, WithSetters};
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
#[derive(Debug, Getters, CopyGetters)]
pub struct View {
    #[getset(get_copy = "pub")]
    id: ViewId,
    #[getset(get = "pub")]
    rgb_path: PathBuf,
    #[getset(get = "pub")]
    npz_path: PathBuf,
    /// One entry per object, regardless of whether the object is visible.
    #[getset(get = "pub")]
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
}

#[derive(Debug, Getters, CopyGetters)]
pub struct ViewMetadata {
    #[getset(get_copy = "pub")]
    scene_id: SceneId,
    #[getset(get_copy = "pub")]
    id: ViewId,
    rgb_relpath: PathBuf,
    #[getset(get = "pub")]
    visible: HashMap<InstanceId, AnnotationMetadata>,
    height: usize,
    width: usize,
}
impl ViewMetadata {
    pub fn builder() -> ViewMetadataBuilder { ViewMetadataBuilder::default() }
}
#[derive(Debug, Default, WithSetters)]
#[getset(set_with = "pub")]
pub struct ViewMetadataBuilder {
    scene_id: Option<SceneId>,
    id: Option<ViewId>,
    rgb_relpath: Option<PathBuf>,
    visible: Option<HashMap<InstanceId, AnnotationMetadata>>,
    height: Option<usize>,
    width: Option<usize>,
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
