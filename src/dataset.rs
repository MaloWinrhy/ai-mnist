use ndarray::{
    Array1, Array2,
};
use rand::Rng;

pub fn generate_dummy_data() -> (Array1<f32>, Array2<f32>) {
    let mut rng = rand::thread_rng();
    let mut data = Array2::<f32>::zeros((100, 2));
    let mut labels = Array1::<f32>::zeros(100);

    for i in 0..100 {
        let x = rng.r#gen::<f32>();
        let y = rng.r#gen::<f32>();
        data[[i, 0]] = x;
        data[[i, 1]] = y;
        labels[i] = if x + y > 1.0 { 1.0 } else { 0.0 };
    }

    (labels, data)
}
