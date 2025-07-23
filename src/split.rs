use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::thread_rng;

// ...existing code...
pub fn split_dataset(data: &Array2<f32>, labels: &Array1<f32>, ratio: f32) -> (Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>) {
    let n = data.shape()[0];
    let n_train = (n as f32 * ratio).round() as usize;
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut thread_rng());

    let train_idx = &indices[..n_train];
    let test_idx = &indices[n_train..];

    let train_data = Array2::from_shape_fn((train_idx.len(), data.shape()[1]), |(i, j)| data[[train_idx[i], j]]);
    let train_labels = Array1::from_shape_fn(train_idx.len(), |i| labels[train_idx[i]]);
    let test_data = Array2::from_shape_fn((test_idx.len(), data.shape()[1]), |(i, j)| data[[test_idx[i], j]]);
    let test_labels = Array1::from_shape_fn(test_idx.len(), |i| labels[test_idx[i]]);

    (train_data, train_labels, test_data, test_labels)
}
