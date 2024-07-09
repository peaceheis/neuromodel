pub mod model;

use std::fs;
use std::time::Instant;
use ndarray::{Array, arr1};
use crate::model::Network;

fn main() {
    const DURATION: i32 = 10000;
    const STIM_TIME: i32 = 250;
    const BIN_SIZE: i32 = DURATION / 10;
    assert_eq!(DURATION % BIN_SIZE, 0);
    const STEPS: i32 = (DURATION as f64 / model::DELTA_T) as i32;
    
    let now = Instant::now();

    let data = fs::read_to_string("config.json").expect("Unable to read file");
    let prefs: serde_json::Value = serde_json::from_str(&*data).expect("JSON was not well-formatted");
    let prefix: String = prefs["prefix"].to_string();

    let mut network: Network = model::Network::new(STIM_TIME, model::NetworkType::Additive, [0, 1, 2, 3, 4, 5]);

    let vals = Array::linspace(0f64, DURATION as f64, STEPS as usize);
    
    for val in vals.iter() {
        network.update();
        println!("{}", val);
    }
    
    println!("Simulated {} ms in {} seconds", DURATION, now.elapsed().as_secs());




}
