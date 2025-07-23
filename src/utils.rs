pub fn confusion_matrix(y_true: &[u8], y_pred: &[u8], num_classes: usize) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0; num_classes]; num_classes];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        matrix[t as usize][p as usize] += 1;
    }
    matrix
}

pub fn per_class_accuracy(conf_matrix: &[Vec<usize>]) -> Vec<f32> {
    conf_matrix.iter()
        .map(|row| {
            let total: usize = row.iter().sum();
            if total == 0 { 0.0 } else { row[row.iter().position(|&x| x == *row.iter().max().unwrap()).unwrap()] as f32 / total as f32 }
        })
        .collect()
}
pub fn softmax(x: &ndarray::Array1<f32>) -> ndarray::Array1<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    ndarray::Array1::from(exp.into_iter().map(|v| v / sum).collect::<Vec<_>>())
}

pub fn cross_entropy(pred: &ndarray::Array1<f32>, target: &ndarray::Array1<f32>) -> f32 {
    -target.iter().zip(pred.iter()).map(|(t, p)| t * p.ln().max(-100.0)).sum::<f32>()
}
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}
