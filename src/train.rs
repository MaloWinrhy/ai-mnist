use crate::model::Perceptron;
use ndarray::{Array1, Array2, Axis};
use crate::utils::sigmoid_derivative;

pub fn train(model: &mut Perceptron, data: &Array2<f32>, labels: &Array1<f32>, epochs: usize, lr: f32) {
    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (i, row) in data.axis_iter(ndarray::Axis(0)).enumerate() {
            let x = row.to_owned();
            let y_true = labels[i];
            let y_pred = model.forward(&x);

            // ...existing code...
            let error = y_pred - y_true;
            total_loss += error.powi(2);

            // ...existing code...
            let grad = error * sigmoid_derivative(y_pred);

            for j in 0..model.weights.len() {
                model.weights[j] -= lr * grad * x[j];
            }

            model.bias -= lr * grad;
        }

        println!("Epoch {} – Loss: {:.4}", epoch + 1, total_loss / data.shape()[0] as f32);
    }
}


pub fn evaluate(model: &Perceptron, data: &Array2<f32>, labels: &Array1<f32>) -> f32 {
    let mut correct = 0;

    for (i, row) in data.axis_iter(Axis(0)).enumerate() {
        let x = row.to_owned();
        let prediction = model.predict(&x);
        if prediction as f32 == labels[i] {
            correct += 1;
        }
    }

    correct as f32 / data.len_of(Axis(0)) as f32
}