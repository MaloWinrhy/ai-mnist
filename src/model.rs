use ndarray::{Array2, Array1, Axis};
use rand::Rng;
use crate::utils::sigmoid;

pub struct Perceptron {
    pub weights: Array1<f32>,
    pub bias: f32,
}

impl Perceptron {
    pub fn new(n_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            weights: Array1::from_iter((0..n_inputs).map(|_| rng.r#gen::<f32>())),
            bias: rng.r#gen::<f32>(),
        }
    }

    pub fn forward(&self, inputs: &Array1<f32>) -> f32 {
        let sum = self.weights.dot(inputs) + self.bias;
        sigmoid(sum)
    }

    // ...existing code...
    pub fn predict(&self, inputs: &Array1<f32>) -> u8 {
        if self.forward(inputs) >= 0.5 {
            1
        } else {
            0
        }
    }
}