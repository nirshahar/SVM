use ndarray::ArrayView1;
use svg::{
    node::element::{Circle, Line},
    Document, Node,
};

use crate::DataPoint;

const RENDERED_POINT_RADIUS: f32 = 0.02;
const VIEWBOX_INSET: f32 = 2. * RENDERED_POINT_RADIUS;

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

pub fn plot_model(data_points: &[DataPoint], model: ArrayView1<'_, f32>, filename: &str) {
    svg::save(filename, &render_svg(&data_points, model.view())).unwrap();
}
