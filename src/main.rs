use std::fs;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use ndarray::Array;

use crate::model::{Config, Network, SerializedNetwork, DELTA_T};

pub mod model;

const DURATION: usize = 1500;
const STIM_TIME: i32 = 500;
const BIN_SIZE: usize = DURATION / 10;

fn main() -> std::io::Result<()> {
    assert_eq!(DURATION % BIN_SIZE, 0);
    
    const STEPS: i32 = (DURATION as f64 / model::DELTA_T) as i32;
    const SHOULD_SKIP_VALS: bool = true;

    let now = Instant::now();
    let mut t: f64 = 0.0;
    let data = fs::read_to_string("config.json").expect("Unable to read file");
    let prefs: serde_json::Value =
        serde_json::from_str(&*data).expect("JSON was not well-formatted");
    let prefix: String = prefs["prefix"].to_string().replace('"', "");
    let parent: String = format!("{}{}", prefix, chrono::offset::Local::now().format("%F %H-%M-%S"));
    println!("again");
    fs::create_dir_all(&parent)?;
    println!("created dir");
    println!("PREFIX {}", prefix);

    let mut network: Network = model::Network::new(
        STIM_TIME,
        model::NetworkType::Mech,
        [0, 1, 2, 3, 4, 5],
        DURATION,
        SHOULD_SKIP_VALS,
    );

    let vals = Array::linspace(0f64, DURATION as f64, STEPS as usize);

    for val in vals.iter() {
        network.update(t);
        println!("{}", val);
        t += DELTA_T;
    }

    for (i, neuron) in network.neurons.iter().enumerate() {
        println!("{}, {}, {}", i, neuron.g_stim_vals.len(), neuron.count);
    }
    println!(
        "Simulated {} ms in {} seconds",
        DURATION,
        now.elapsed().as_secs()
    );

    let mut buffer = File::create(format!("{}/result.json", prefix))?;
    buffer.write_all(
        serde_json::to_string_pretty(&SerializedNetwork::from_network(network, parent.clone()))?
            .as_ref(),
    )?;
    println!("Saved result JSON");

    buffer = File::create(format!("{}/config.json", parent))?;
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
    
    let val: usize = 0;

    Ok(())
}
