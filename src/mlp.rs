use ndarray::{Array1, Array2};
use rand::Rng;
use crate::utils::{sigmoid, sigmoid_derivative, relu, relu_derivative, softmax, cross_entropy};

pub struct MLP {
    input_size: usize,
    hidden1_size: usize,
    hidden2_size: usize,
    output_size: usize,
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
    w3: Array2<f32>,
    b3: Array1<f32>,
}

impl MLP {
    pub fn new(input_size: usize, hidden1_size: usize, hidden2_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let w1 = Array2::from_shape_fn((hidden1_size, input_size), |_| rng.random::<f32>());
        let b1 = Array1::zeros(hidden1_size);
        let w2 = Array2::from_shape_fn((hidden2_size, hidden1_size), |_| rng.random::<f32>());
        let b2 = Array1::zeros(hidden2_size);
        let w3 = Array2::from_shape_fn((output_size, hidden2_size), |_| rng.random::<f32>());
        let b3 = Array1::zeros(output_size);

        Self {
            input_size,
            hidden1_size,
            hidden2_size,
            output_size,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
        }
    }

    pub fn forward(&self, x: &Array1<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>) {
        let z1 = self.w1.dot(x) + &self.b1;
        let a1 = z1.mapv(relu);
        let z2 = self.w2.dot(&a1) + &self.b2;
        let a2 = z2.mapv(relu);
        let z3 = self.w3.dot(&a2) + &self.b3;
        let output = softmax(&z3);
        (z1, a1, z2, a2, output)
    }

    pub fn train(&mut self, data: &Array2<f32>, labels: &Array2<f32>, epochs: usize, lr: f32) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (i, row) in data.axis_iter(ndarray::Axis(0)).enumerate() {
                let x = row.to_owned();
                let y_true = labels.row(i).to_owned();

                // ...existing code...
                let z1 = self.w1.dot(&x) + &self.b1;
                let a1 = z1.mapv(relu);
                let z2 = self.w2.dot(&a1) + &self.b2;
                let a2 = z2.mapv(relu);
                let z3 = self.w3.dot(&a2) + &self.b3;
                let y_pred = softmax(&z3);

                // ...existing code...
                let loss = cross_entropy(&y_pred, &y_true);
                total_loss += loss;

                // ...existing code...
                let d_output = &y_pred - &y_true;

                // ...existing code...
                let mut d_w3 = Array2::<f32>::zeros((self.output_size, self.hidden2_size));
                for j in 0..self.output_size {
                    for k in 0..self.hidden2_size {
                        d_w3[[j, k]] = d_output[j] * a2[k];
                    }
                }
                let d_b3 = d_output.clone();

                // ...existing code...
                let mut d_hidden2 = Array1::<f32>::zeros(self.hidden2_size);
                for j in 0..self.hidden2_size {
                    let mut sum = 0.0;
                    for k in 0..self.output_size {
                        sum += d_output[k] * self.w3[[k, j]];
                    }
                    d_hidden2[j] = sum * relu_derivative(z2[j]);
                }

                let mut d_w2 = Array2::<f32>::zeros((self.hidden2_size, self.hidden1_size));
                for j in 0..self.hidden2_size {
                    for k in 0..self.hidden1_size {
                        d_w2[[j, k]] = d_hidden2[j] * a1[k];
                    }
                }
                let d_b2 = d_hidden2.clone();

                // ...existing code...
                let mut d_hidden1 = Array1::<f32>::zeros(self.hidden1_size);
                for j in 0..self.hidden1_size {
                    let mut sum = 0.0;
                    for k in 0..self.hidden2_size {
                        sum += d_hidden2[k] * self.w2[[k, j]];
                    }
                    d_hidden1[j] = sum * relu_derivative(z1[j]);
                }

                let mut d_w1 = Array2::<f32>::zeros((self.hidden1_size, self.input_size));
                for j in 0..self.hidden1_size {
                    for k in 0..self.input_size {
                        d_w1[[j, k]] = d_hidden1[j] * x[k];
                    }
                }
                let d_b1 = d_hidden1.clone();

                // ...existing code...
                self.w3 = &self.w3 - &(d_w3.mapv(|v| v * lr));
                self.b3 = &self.b3 - &(d_b3.mapv(|v| v * lr));
                self.w2 = &self.w2 - &(d_w2.mapv(|v| v * lr));
                self.b2 = &self.b2 - &(d_b2.mapv(|v| v * lr));
                self.w1 = &self.w1 - &(d_w1.mapv(|v| v * lr));
                self.b1 = &self.b1 - &(d_b1.mapv(|v| v * lr));
            }

            if epoch % 100 == 0 || epoch == epochs - 1 {
                println!("[TRAIN] [Epoch {}][Loss={:.4}]", epoch + 1, total_loss / data.len_of(ndarray::Axis(0)) as f32);
            }
        }
    }

    pub fn predict(&self, x: &Array1<f32>) -> u8 {
        let (_, _, _, _, output) = self.forward(x);
        output.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u8)
            .unwrap_or(0)
    }
}
