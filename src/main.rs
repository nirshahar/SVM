use std::iter;

use ndarray::{s, Array, Array1, ArrayView1, Ix1};
use rand::{seq::SliceRandom, Rng};
use rand_distr::{Distribution, UnitDisc};
use svg::{
    node::element::{Circle, Line},
    Document, Node,
};

const C: f32 = 1.;
const RENDERED_POINT_RADIUS: f32 = 0.02;
const VIEWBOX_INSET: f32 = 2. * RENDERED_POINT_RADIUS;

struct DataPoint {
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
            let data = Array1::from([data[0] + bool_to_float(tag), data[1], 1.].to_vec());

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

fn render_data_point(point: &DataPoint) -> impl Node {
    Circle::new()
        .set("cx", point.data[0])
        .set("cy", point.data[1])
        .set("r", RENDERED_POINT_RADIUS)
        .set("fill", if point.tag { "red" } else { "blue" })
}

fn get_point_on_model(model: ArrayView1<'_, f32>, (bound_x, bound_y): (f32, f32)) -> (f32, f32) {
    // w_1 * x + w_2 * y + b = 0
    if model[1] != 0. {
        // y = - (w_1 * x + b) / w_2
        (bound_x, -(model[0] * bound_x + model[2]) / model[1])
    } else {
        // x = - (w_2 * y + b) / w_1
        (-(model[1] * bound_y + model[2]) / model[0], bound_y)
    }
}

fn render_line(
    model: ArrayView1<'_, f32>,
    view_box_min: (f32, f32),
    view_box_max: (f32, f32),
) -> impl Node {
    let (x1, y1) = get_point_on_model(model, view_box_min);
    let (x2, y2) = get_point_on_model(model, view_box_max);

    Line::new()
        .set("x1", x1)
        .set("y1", y1)
        .set("x2", x2)
        .set("y2", y2)
        .set("stroke", "black")
        .set("stroke-width", RENDERED_POINT_RADIUS / 2.)
}

fn render_svg(data_points: &[DataPoint], model: ArrayView1<'_, f32>) -> Document {
    let min_x = data_points
        .iter()
        .fold(f32::INFINITY, |a, b| a.min(b.data[0]))
        - VIEWBOX_INSET;
    let min_y = data_points
        .iter()
        .fold(f32::INFINITY, |a, b| a.min(b.data[1]))
        - VIEWBOX_INSET;
    let max_x = data_points
        .iter()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b.data[0]))
        + VIEWBOX_INSET;
    let max_y = data_points
        .iter()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b.data[1]))
        + VIEWBOX_INSET;

    let document = data_points
        .iter()
        .fold(
            Document::new().set("viewBox", (min_x, min_y, max_x - min_x, max_y - min_y)),
            |doc, point| doc.add(render_data_point(point)),
        )
        .add(render_line(model, (min_x, min_y), (max_x, max_y)));

    document
}

fn main() {
    const LEARNING_RATE: f32 = 0.01;
    const BATCH_SIZE: usize = 10;

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

    svg::save("out.svg", &render_svg(&data_points, model.view())).unwrap();
}
