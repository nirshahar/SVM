use ndarray::{array, Array1, ArrayView1, Shape};

use crate::{bool_to_float, DataPoint};

struct Model<'a, K> {
    kernel: K,
    support_vectors: &'a [DataPoint],
}

fn dual_grad_at_data_point(
    data_point: &DataPoint,
    data_points: &[DataPoint],
    alpha: ArrayView1<f32>,
    kernel: &impl Fn(ArrayView1<'_, f32>, ArrayView1<'_, f32>) -> f32,
) -> f32 {
    let sum: f32 = data_points
        .iter()
        .zip(alpha)
        .map(|(point, coef)| {
            coef * bool_to_float(data_point.tag)
                * bool_to_float(point.tag)
                * kernel(data_point.data.view(), point.data.view())
        })
        .sum();

    sum - 1.
}

fn dual_grad_with_kernel(
    data_points: &[DataPoint],
    alpha: ArrayView1<f32>,
    kernel: impl Fn(ArrayView1<'_, f32>, ArrayView1<'_, f32>) -> f32,
) -> Array1<f32> {
    data_points
        .iter()
        .map(|point| dual_grad_at_data_point(point, data_points, alpha, &kernel))
        .collect()
}

fn learn_model<K : Fn(ArrayView1<'_, f32>, ArrayView1<'_, f32>) -> f32>(data_points: &[DataPoint], kernel: K) -> Model<K> {
    todo!()
}

fn square_kernel(x1: ArrayView1<'_, f32>, x2: ArrayView1<'_, f32>, r: i32) -> f32 {
    (x1.dot(&x2) + 1.).powi(r)
}

fn rbf_kernel(x1: ArrayView1<'_, f32>, x2: ArrayView1<'_, f32>, sigma: f32) -> f32 {
    let dist_vec: Array1<f32> = x1.to_owned() - x2.to_owned();

    (-dist_vec.dot(&dist_vec) / (2. * sigma).powi(2)).exp()
}
