use ndarray::{s, Array, Array1, ArrayView1, Ix1};
use rand::{seq::SliceRandom, Rng};
use rand_distr::{Distribution, UnitDisc};
use svg::{
    node::element::{Circle, Line},
    Document, Node,
};

const C: f32 = 1.;
const RENDERED_POINT_RADIUS: f32 = 0.01;
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

            let data = UnitDisc.sample(&mut rng);
            let data = Array1::from(
                [data[0] - 2. * (1. + (tag as i32 as f32) * 2.), data[1], 1.].to_vec(),
            );

            let data_point = DataPoint { data, tag };
            data_points.push(data_point);
        }

        data_points
    }
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
    let y = data_point.tag as i32 as f32;
    let data = &data_point.data;

    if 1. - y * param_estimation.dot(data) >= 0. {
        &param_estimation - &(C * y * data)
    } else {
        param_estimation.to_owned()
    }
}

fn soft_svm_loss(data_points: &[DataPoint], param_estimation: &Array1<f32>) -> f32 {
    let w = param_estimation.slice(s![0..param_estimation.dim() - 1]);

    let a = 0.5 * w.dot(&w);
    let b = data_points
        .iter()
        .map(|point| {
            let value = C * (1. - (point.tag as i32 as f32) * param_estimation.dot(&point.data));
            value.max(0.)
        })
        .sum::<f32>();

    dbg!(a);
    dbg!(b);

    0.
}

fn render_data_point(point: &DataPoint) -> impl Node {
    Circle::new()
        .set("cx", point.data[0])
        .set("cy", point.data[1])
        .set("r", RENDERED_POINT_RADIUS)
        .set("fill", if point.tag { "red" } else { "blue" })
}

fn get_point_on_model(model: ArrayView1<'_, f32>, (bound_x, bound_y): (f32, f32)) -> (f32, f32) {
    if model[1] != 0. {
        (bound_x, -(model[0] * bound_x + model[2]) / model[1])
    } else {
        (bound_x, bound_y)
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
    const BATCH_SIZE: usize = 4;

    let data_points = DataPoint::generate_test_data(40);

    // for epochs in 0..100 {
    //     let model = sgd(
    //         &data_points,
    //         Ix1(3),
    //         epochs,
    //         LEARNING_RATE,
    //         BATCH_SIZE,
    //         soft_svm_grad,
    //     );

    //     println!(
    //         "model ran with {} epochs, with loss of {}",
    //         epochs,
    //         soft_svm_loss(&data_points, &model)
    //     );

    // }

    svg::save(
        "out.svg",
        &render_svg(
            &data_points,
            sgd(
                &data_points,
                Ix1(3),
                100,
                LEARNING_RATE,
                BATCH_SIZE,
                soft_svm_grad,
            )
            .view(),
        ),
    )
    .unwrap();
}
