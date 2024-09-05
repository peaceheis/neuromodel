use std::{fs, io};
use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use ndarray::Array;

use crate::model::{Config, Network, SerializedNetwork};

pub mod model;

const DURATION: i32 = 1500;
const STIM_TIME: i32 = 250;
const BIN_SIZE: i32 = DURATION / 10;


fn main() -> std::io::Result<()>{
    assert_eq!(DURATION % BIN_SIZE, 0);
    const STEPS: i32 = (DURATION as f64 / model::DELTA_T) as i32;
    
    
    let now = Instant::now();

    let data = fs::read_to_string("config.json").expect("Unable to read file");
    let prefs: serde_json::Value = serde_json::from_str(&*data).expect("JSON was not well-formatted");
    let prefix = std::env::current_dir().unwrap().display().to_string() + "\\output";
    println!("PREFIX {}", prefix);

    let mut network: Network = model::Network::new(STIM_TIME, model::NetworkType::Additive, [0, 1, 2, 3, 4, 5]);

    let vals = Array::linspace(0f64, DURATION as f64, STEPS as usize);
    
    for val in vals.iter() {
        network.update();
        println!("{}", val);
    }
    
    for (i, neuron) in network.neurons.iter().enumerate() {
        println!("{}, {}", i, neuron.spike_counts.len())
    }
    println!("Simulated {} ms in {} seconds", DURATION, now.elapsed().as_secs());
    
    let parent: String = format!("{}\\{}", prefix, chrono::offset::Local::now().format("%F %H-%M"));
    println!("{}", parent);
    fs::create_dir(&parent)?;
    
    let mut buffer = File::create(format!("{}\\result.json", prefix))?;
    buffer.write_all(serde_json::to_string_pretty(&SerializedNetwork::from_network(network, parent.clone()))?.as_ref())?;
    println!("Saved result JSON");
    
    buffer = File::create(format!("{}\\config.json", parent))?;
    buffer.write_all(serde_json::to_string_pretty(&Config::create())?.as_ref())?;
    println!("Saved config");
    
    // 
    // println!("Creating graphs");
    // let output = Command::new("python3")
    //     .arg("graph.py")
    //     .output()
    //     .expect("Failed to execute command");
    // 
    // println!("status: {}", output.status);
    // io::stdout().write_all(&output.stdout).unwrap();
    
    Ok(())
}
