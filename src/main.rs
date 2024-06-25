pub mod model;

use std::fs;

fn main() {
    const duration: i32 = 500;
    const stim_time: i32 = 250;
    const BIN_SIZE: i32 = duration / 10;
    assert_eq!(duration % BIN_SIZE, 0);
    const STEPS: i32 = (duration as f64 / model::DELTA_T) as i32;

    let data = fs::read_to_string("config.json").expect("Unable to read file");
    let prefs: serde_json::Value = serde_json::from_str(&*data).expect("JSON was not well-formatted");
    let prefix: String = prefs["prefix"].to_string();
  
    let network = model::Network::new(stim_time, model::NetworkType::Additive, &[0, 1, 2, 3, 4, 5]);
        
    

}
