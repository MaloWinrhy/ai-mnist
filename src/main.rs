
use ndarray::Axis;
mod mlp;
mod dataset;
mod utils;

use mlp::MLP;
mod split;
use split::split_dataset;
mod iris_loader;
use iris_loader::load_iris_csv;

fn main() {
    use std::fs::File;
    use std::io::Write;
    use utils::{confusion_matrix, per_class_accuracy};

    let (features, labels) = load_iris_csv("iris.csv");
    let (train_data, train_labels_raw, test_data, test_labels_raw) = split_dataset(&features, &labels.mapv(|v| v as f32), 0.8);

    fn to_one_hot(labels: &ndarray::Array1<f32>, num_classes: usize) -> ndarray::Array2<f32> {
        let mut one_hot = ndarray::Array2::<f32>::zeros((labels.len(), num_classes));
        for (i, &v) in labels.iter().enumerate() {
            one_hot[[i, v as usize]] = 1.0;
        }
        one_hot
    }

    let mut best_acc = 0.0;
    let mut best_params = (0, 0.0, 0);
    let mut best_preds = Vec::new();
    let mut best_true = Vec::new();

    for &hidden_size in &[6, 8, 12, 16] {
        for &lr in &[0.05, 0.1, 0.2] {
            for &epochs in &[500, 1000, 2000] {
                let mut model = MLP::new(4, hidden_size, hidden_size, 3);
                let train_labels = to_one_hot(&train_labels_raw, 3);
                let test_labels = to_one_hot(&test_labels_raw, 3);
                model.train(&train_data, &train_labels, epochs, lr);

    // Test predictions and accuracy
                let mut preds = Vec::new();
                let mut true_labels = Vec::new();
                let mut correct_test = 0;
                for (i, row) in test_data.axis_iter(Axis(0)).enumerate() {
                    let x = row.to_owned().into_dimensionality::<ndarray::Ix1>().unwrap();
                    let prediction = model.predict(&x);
                    let true_label = test_labels_raw[i] as u8;
                    preds.push(prediction);
                    true_labels.push(true_label);
                    if prediction == true_label {
                        correct_test += 1;
                    }
                }
                let accuracy_test = correct_test as f32 / test_data.len_of(Axis(0)) as f32;
                println!("[INFO] [hidden_size={}][lr={}][epochs={}] => [test_acc={:.2}%]", hidden_size, lr, epochs, accuracy_test * 100.0);
                if accuracy_test > best_acc {
                    best_acc = accuracy_test;
                    best_params = (hidden_size, lr, epochs);
                    best_preds = preds.clone();
                    best_true = true_labels.clone();
                }
            }
        }
    }

    println!("[RESULT] [Best params][hidden_size={}][lr={}][epochs={}]", best_params.0, best_params.1, best_params.2);
    println!("[RESULT] [Best test accuracy={:.2}%]", best_acc * 100.0);

    // Confusion matrix and per-class accuracy
    let conf = confusion_matrix(&best_true, &best_preds, 3);
    println!("[INFO] [Confusion matrix:]");
    for row in &conf {
        println!("[DATA] {:?}", row);
    }
    let per_class = per_class_accuracy(&conf);
    println!("[INFO] [Accuracy per class:]");
    for (i, acc) in per_class.iter().enumerate() {
        println!("[RESULT] [Class {}][Accuracy={:.2}%]", i, acc * 100.0);
    }

    // Export predictions to CSV for visualization
    let mut file = File::create("iris_predictions.csv").unwrap();
    writeln!(file, "y_true,y_pred").unwrap();
    for (t, p) in best_true.iter().zip(best_preds.iter()) {
        writeln!(file, "{}", format!("{},{}", t, p)).unwrap();
    }
}
