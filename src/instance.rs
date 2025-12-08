use serde::Serialize;

use crate::scene::SceneId;
use crate::view::ViewId;


#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum InstancesObjectsValue {
    Background,
    Object(InstanceId),
}
impl InstancesObjectsValue {
    const BACKGROUND_RAW_VALUE: usize = 0;
}
impl From<usize> for InstancesObjectsValue {
    fn from(value: usize) -> Self {
        if value == Self::BACKGROUND_RAW_VALUE {
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
pub struct InstanceId { inner: usize }
impl InstanceId {
    pub fn new(v: usize) -> Self { Self { inner: v } }
}

#[derive(Clone, Debug)]
pub struct AnnotationMetadata {
    scene_id: SceneId,
    view_id: ViewId,
    instance_id: InstanceId,
    bbox: Bboxx1y1x2y2,
    category_id: CategoryId,
    area: Area,
    iscrowd: IsCrowdBool,
}
impl AnnotationMetadata {
    pub fn builder() -> AnnotationMetadataBuilder { AnnotationMetadataBuilder::default() }
    pub fn scene_id(&self) -> SceneId { self.scene_id }
    pub fn view_id(&self) -> ViewId { self.view_id }
    pub fn instance_id(&self) -> InstanceId { self.instance_id }
}
#[derive(Default, Debug)]
pub struct AnnotationMetadataBuilder {
    pub scene_id: Option<SceneId>,
    pub view_id: Option<ViewId>,
    pub instance_id: Option<InstanceId>,
    pub bbox: Option<Bboxx1y1x2y2>,
    pub category_id: Option<CategoryId>,
    pub area: Option<Area>,
    pub iscrowd: Option<IsCrowdBool>,
}
impl AnnotationMetadataBuilder {
    pub fn build(self) -> AnnotationMetadata {
        AnnotationMetadata {
            scene_id: self.scene_id.expect("scene_id is not set."),
            view_id: self.view_id.expect("view_id is not set."),
            instance_id: self.instance_id.expect("instance_id is not set."),
            bbox: self.bbox.expect("bbox is not set."),
            category_id: self.category_id.expect("category_id is not set."),
            area: self.area.expect("area is not set."),
            iscrowd: self.iscrowd.expect("iscrowd is not set."),
        }
    }
}
#[derive(Debug, Serialize)]
pub struct AnnotationMetadataSerde {
    id: usize,
    image_id: usize,
    instance_id: InstanceId,
    bbox: Bboxx1y1x2y2,
    category_id: CategoryId,
    area: Area,
    iscrowd: IsCrowdBool,
}
impl AnnotationMetadataSerde {
    pub fn from_ann(value: AnnotationMetadata, id: usize, image_id: usize) -> Self {
        Self { id, image_id, instance_id: value.instance_id, bbox: value.bbox, category_id: value.category_id,
               area: value.area, iscrowd: value.iscrowd }
    }
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug, Ord, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct CategoryId { inner: usize }
impl CategoryId {
    pub const BACKGROUND: Self = Self { inner: 0 };
    pub const FOREGROUND: Self = Self { inner: 1 };
}

#[derive(PartialEq, Clone, Copy, Debug, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Area { inner: f64 }
impl Area {
    pub fn new(a: f64) -> Self {
        assert!(a.is_normal());
        Self { inner: a }
    }
}
impl<T: num_traits::cast::ToPrimitive + num_traits::sign::Unsigned> From<T> for Area {
    fn from(value: T) -> Self {
        Self::new(value.to_f64().unwrap())
    }
}

#[derive(Debug, Clone, Serialize, Eq, PartialEq)]
#[serde(into = "Bboxx1y1x2y2Serde")]
pub struct Bboxx1y1x2y2 {
    x1: usize,
    x2: usize,
    y1: usize,
    y2: usize,
}
impl Bboxx1y1x2y2 {
    pub fn builder() -> Bboxx1y1x2y2Builder { Bboxx1y1x2y2Builder::default() }
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
        Self(x1, y1, x2, y2)
    }
}
#[derive(Default, Debug)]
pub struct Bboxx1y1x2y2Builder {
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

#[derive(PartialEq, Clone, Copy, Debug, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct IsCrowdBool { inner: u8 }
impl From<bool> for IsCrowdBool {
    fn from(value: bool) -> Self {
        Self { inner: value.into() }
    }
}
