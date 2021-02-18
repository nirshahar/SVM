use std::iter;

use crate::{bool_to_float, DataPoint};
use ndarray::{s, Array1, ArrayView1};

const C: f32 = 1.;

pub fn grad(data_point: &DataPoint, param_estimation: ArrayView1<'_, f32>) -> Array1<f32> {
    let d = param_estimation.dim();

    let y = bool_to_float(data_point.tag);
    let data = &data_point.data;

    let weight_grad: Array1<_> = param_estimation
        .slice(s![0..d - 1])
        .iter()
        .copied()
        .chain(iter::once(0.))
        .collect();

    if 1. - y * param_estimation.dot(data) >= 0. {
        weight_grad - C * y * data
    } else {
        weight_grad
    }
}

pub fn loss(data_points: &[DataPoint], param_estimation: &Array1<f32>) -> f32 {
    let w = param_estimation.slice(s![0..param_estimation.dim() - 1]);

    0.5 * w.dot(&w)
        + data_points
            .iter()
            .map(|point| {
                let value = C * (1. - bool_to_float(point.tag) * param_estimation.dot(&point.data));
                value.max(0.)
            })
            .sum::<f32>()
}
