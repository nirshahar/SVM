use ndarray::{Array, Array1, ArrayView1, Ix1};
use rand::{seq::SliceRandom, Rng};
use rand_distr::{Distribution, UnitDisc};

mod kernel;
mod plot;
mod soft;

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
            let data = Array1::from(
                [
                    data[0] + bool_to_float(tag),
                    data[1] + bool_to_float(tag) * 2.,
                    1.,
                ]
                .to_vec(),
            );

            let data_point = DataPoint { data, tag };
            data_points.push(data_point);
        }

        data_points
    }
}

fn bool_to_float(b: bool) -> f32 {
    2. * (b as i32 as f32) - 1.
}

/// An implementation for the stochastic gradient descent function with projection.
/// The projection must be into a convex set.
fn proj_sgd<P>(
    points: &[P],
    dim: usize,
    epochs: usize,
    learning_rate: f32,
    batch_size: usize,
    grad: impl Fn(&P, ArrayView1<'_, f32>) -> Array1<f32>,
    proj: impl Fn(Array1<f32>) -> Array1<f32>,
) -> Array1<f32> {
    let dim = Ix1(dim);

    let mut rng = rand::thread_rng();
    let mut optimum = Array::zeros(dim);

    for _ in 0..epochs {
        let average_gradient: Array1<f32> = points
            .choose_multiple(&mut rng, batch_size)
            .map(|point| grad(point, optimum.view()))
            .fold(Array::zeros(dim), |a, b| a + b)
            / (batch_size as f32);

        optimum = proj(optimum - learning_rate * average_gradient);
    }

    optimum
}

/// An implementation for the stochastic gradient descent function.
fn sgd<P>(
    points: &[P],
    dim: usize,
    epochs: usize,
    learning_rate: f32,
    batch_size: usize,
    grad: impl Fn(&P, ArrayView1<'_, f32>) -> Array1<f32>,
) -> Array1<f32> {
    proj_sgd(
        points,
        dim,
        epochs,
        learning_rate,
        batch_size,
        grad,
        std::convert::identity,
    )
}

fn proj_gd<P>(
    points: &[P],
    dim: usize,
    epochs: usize,
    learning_rate: f32,
    grad: impl Fn(&[P], ArrayView1<'_, f32>) -> Array1<f32>,
    proj: impl Fn(Array1<f32>) -> Array1<f32>,
) -> Array1<f32> {
    let dim = Ix1(dim);

    let mut optimum = Array::zeros(dim);

    for _ in 0..epochs {
        let grad_at_optimum = grad(points, optimum.view());

        optimum = proj(optimum - learning_rate * grad_at_optimum);
    }

    optimum
}

fn main() {
    const LEARNING_RATE: f32 = 0.01;
    const BATCH_SIZE: usize = 4;

    let data_points = DataPoint::generate_test_data(100);

    let model = sgd(
        &data_points,
        3,
        10000,
        LEARNING_RATE,
        BATCH_SIZE,
        soft::grad,
    );

    println!("model: {}", model);
    println!("loss: {}", soft::loss(&data_points, &model));

    plot::plot_model(&data_points, model.view(), "WOOSH.svg");
}
