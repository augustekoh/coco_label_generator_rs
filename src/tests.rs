use super::*;

#[test]
fn small_areas() {
    let b_ = InstancesObjectsValue::Background;
    let f1 = InstancesObjectsValue::Object(InstanceId::new(1));
    let f2 = InstancesObjectsValue::Object(InstanceId::new(2));
    let f3 = InstancesObjectsValue::Object(InstanceId::new(3));
    let arr: Array2<InstancesObjectsValue> = ndarray::array![
        [b_, f1, f1, f1, f2, f1, f2],
        [b_, f1, f1, f1, f2, b_, b_],
        [f3, f1, f2, f3, f2, f1, f2],
        [b_, f3, f3, f3, f1, f1, f1],
    ];
    let areas = areas(&arr);
    assert_eq!(areas[&f1], Area::new(12.0));
    assert_eq!(areas[&f2], Area::new(6.0));
    assert_eq!(areas[&f3], Area::new(5.0));
    assert_eq!(areas[&b_], Area::new(5.0));

    assert_eq!(bounding_box(&arr, &f1),
               Bboxx1y1x2y2::bulider().set_x1(1).set_x2(7).set_y1(0).set_y2(4).build().unwrap());
    assert_eq!(bounding_box(&arr, &f2),
               Bboxx1y1x2y2::bulider().set_x1(2).set_x2(7).set_y1(0).set_y2(3).build().unwrap());
    assert_eq!(bounding_box(&arr, &f3),
               Bboxx1y1x2y2::bulider().set_x1(0).set_x2(4).set_y1(2).set_y2(4).build().unwrap());
    assert_eq!(bounding_box(&arr, &b_),
               Bboxx1y1x2y2::bulider().set_x1(0).set_x2(7).set_y1(0).set_y2(4).build().unwrap());
}

#[test]
fn nano_areas() {
    let b_ = InstancesObjectsValue::Background;
    let f1 = InstancesObjectsValue::Object(InstanceId::new(1));
    let f2 = InstancesObjectsValue::Object(InstanceId::new(2));
    let f3 = InstancesObjectsValue::Object(InstanceId::new(3));
    let f4 = InstancesObjectsValue::Object(InstanceId::new(4));
    let f5 = InstancesObjectsValue::Object(InstanceId::new(5));
    let arr: Array2<InstancesObjectsValue> = ndarray::array![
        [b_, b_, b_, b_, f2, b_, b_],
        [b_, f1, b_, b_, f2, b_, b_],
        [b_, b_, b_, f3, b_, f4, b_],
        [b_, b_, f3, b_, b_, f4, f5],
    ];
    let areas = areas(&arr);
    assert_eq!(areas[&f1], Area::new(1.0));
    assert_eq!(areas[&f2], Area::new(2.0));
    assert_eq!(areas[&f3], Area::new(2.0));
    assert_eq!(areas[&f4], Area::new(2.0));
    assert_eq!(areas[&f5], Area::new(1.0));
    assert_eq!(areas[&b_], Area::new(20.0));

    assert_eq!(bounding_box(&arr, &f1),
               Bboxx1y1x2y2::bulider().set_x1(1).set_x2(2).set_y1(1).set_y2(2).build().unwrap());
    assert_eq!(bounding_box(&arr, &f2),
               Bboxx1y1x2y2::bulider().set_x1(4).set_x2(5).set_y1(0).set_y2(2).build().unwrap());
    assert_eq!(bounding_box(&arr, &f3),
               Bboxx1y1x2y2::bulider().set_x1(2).set_x2(4).set_y1(2).set_y2(4).build().unwrap());
    assert_eq!(bounding_box(&arr, &f4),
               Bboxx1y1x2y2::bulider().set_x1(5).set_x2(6).set_y1(2).set_y2(4).build().unwrap());
    assert_eq!(bounding_box(&arr, &f5),
               Bboxx1y1x2y2::bulider().set_x1(6).set_x2(7).set_y1(3).set_y2(4).build().unwrap());
    assert_eq!(bounding_box(&arr, &b_),
               Bboxx1y1x2y2::bulider().set_x1(0).set_x2(7).set_y1(0).set_y2(4).build().unwrap());
}
