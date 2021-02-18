use std::iter;

use ndarray::{s, Array, Array1, ArrayView1, Ix1};
use rand::{seq::SliceRandom, Rng};
use rand_distr::{Distribution, UnitDisc};

mod plot;

const C: f32 = 1.;

pub struct DataPoint {
    data: Array1<f32>,
    tag: bool,
}

impl DataPoint {
    fn generate_test_data(n: usize) -> Vec<Self> {
        let mut data_points = Vec::new();
        let mut rng = rand::thread_rng();

        for _ in 0..n {
            let tag = rng.gen_bool(0.5);

            let data: [f32; 2] = UnitDisc.sample(&mut rng);
            let data = Array1::from([data[0] + bool_to_float(tag), data[1] + bool_to_float(tag) * 2., 1.].to_vec());

            let data_point = DataPoint { data, tag };
            data_points.push(data_point);
        }

        data_points
    }
}

fn bool_to_float(b: bool) -> f32 {
    2. * (b as i32 as f32) - 1.
}

/// An implementation for the stochastic gradient descent function.
fn sgd<P>(
    points: &[P],
    dim: Ix1,
    epochs: usize,
    learning_rate: f32,
    batch_size: usize,
    grad: impl Fn(&P, ArrayView1<'_, f32>) -> Array1<f32>,
) -> Array1<f32> {
    let mut rng = rand::thread_rng();
    let mut optimum = Array::zeros(dim);

    for _ in 0..epochs {
        let average_gradient = points
            .choose_multiple(&mut rng, batch_size)
            .map(|point| grad(point, optimum.view()))
            .fold(Array::zeros(dim), |a, b| a + b)
            / (batch_size as f32);

        optimum = optimum - learning_rate * average_gradient;
    }

    optimum
}

fn soft_svm_grad(data_point: &DataPoint, param_estimation: ArrayView1<'_, f32>) -> Array1<f32> {
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

fn soft_svm_loss(data_points: &[DataPoint], param_estimation: &Array1<f32>) -> f32 {
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

fn main() {
    const LEARNING_RATE: f32 = 0.01;
    const BATCH_SIZE: usize = 4;

    let data_points = DataPoint::generate_test_data(100);

    let model = sgd(
        &data_points,
        Ix1(3),
        10000,
        LEARNING_RATE,
        BATCH_SIZE,
        soft_svm_grad,
    );

    println!("model: {}", model);
    println!("loss: {}", soft_svm_loss(&data_points, &model));

    plot::plot_model(&data_points, model.view(), "WOOSH.svg");
}
