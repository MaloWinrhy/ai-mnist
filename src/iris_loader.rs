use ndarray::{Array1, Array2};
use serde::Deserialize;
use csv::ReaderBuilder;

#[derive(Debug, Deserialize)]
pub struct IrisRecord {
    #[serde(rename = "sepal.length")]
    pub sepal_length: f32,
    #[serde(rename = "sepal.width")]
    pub sepal_width: f32,
    #[serde(rename = "petal.length")]
    pub petal_length: f32,
    #[serde(rename = "petal.width")]
    pub petal_width: f32,
    #[serde(rename = "variety")]
    pub variety: String,
}

// ...existing code...
pub fn encode_class(class: &str) -> u8 {
    match class.to_lowercase().as_str() {
        "setosa" => 0,
        "versicolor" => 1,
        "virginica" => 2,
        _ => 0,
    }
}

// ...existing code...
pub fn load_iris_csv(path: &str) -> (Array2<f32>, Array1<u8>) {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path).unwrap();
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for result in rdr.deserialize() {
        let record: IrisRecord = result.unwrap();
        features.push(vec![record.sepal_length, record.sepal_width, record.petal_length, record.petal_width]);
        labels.push(encode_class(&record.variety));
    }
    let n = features.len();
    let m = features[0].len();
    let flat: Vec<f32> = features.into_iter().flatten().collect();
    let mut features_arr = Array2::from_shape_vec((n, m), flat).unwrap();
    // ...existing code...
    for col in 0..m {
        let col_data = features_arr.column(col);
        let min = col_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = col_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for row in 0..n {
            features_arr[[row, col]] = (features_arr[[row, col]] - min) / (max - min);
        }
    }
    let labels_arr = Array1::from(labels);
    (features_arr, labels_arr)
}
