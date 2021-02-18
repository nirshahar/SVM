use ndarray::{Array1, ArrayView1};

use crate::DataPoint;

struct Model<'a, K> {
    kernel: K,
    support_vectors: &'a [DataPoint],
}

fn dual_grad(
    data_points: &[DataPoint],
    alpha: Array1<f32>,
    kernel: impl Fn(ArrayView1<'_, f32>, ArrayView1<'_, f32>) -> f32,
) -> Array1<f32> {
    
    todo!()
}
