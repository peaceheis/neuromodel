use rand::{Rng, SeedableRng};
use rand::distributions::Uniform;
use rand::rngs::SmallRng;
use std::collections::HashSet;
use std::f64::consts::E;
use std::iter::Sum;
use std::cmp::Ordering;

const DELTA_T: f64 = 0.1;

fn random_choice(rng: &mut SmallRng, given_prob: f64) -> bool {
    let rand_val = rng.sample(Uniform::new(0.0, 1.0));
    rand_val < given_prob
}

struct Neuron {
    t_stim_on: f64,
    t_stim_off: f64,
    exc_times: Vec<f64>,
    slow_exc_times: Vec<f64>,
    inh_times: Vec<f64>,
    slow_inh_times: Vec<f64>,
    connected_neurons: Vec<Neuron>,
    neuron_type: String,
    t: f64,
    v: f64,
    dv_dt: f64,
    lambda_odor: f64,
    lambda_mech: f64,
    spike_times: Vec<f64>,
    voltages: Vec<f64>,
    g_inh_vals: Vec<f64>,
    g_slow_vals: Vec<f64>,
    g_sk_vals: Vec<f64>,
    spike_counts: Vec<usize>,
    slow_exc_vals: Vec<f64>,
    g_exc_vals: Vec<f64>,
    refractory_counter: f64,
    n_id: usize,
    total_inhibition: usize,
    total_excitation: usize,
}

impl Neuron {
    const V_L: f64 = 0.0;
    const V_EXC: f64 = 14.0 / 3.0;
    const V_STIM: f64 = 14.0 / 3.0;
    const V_INH: f64 = -2.0 / 3.0;
    const V_SK: f64 = -2.0 / 3.0;
    const V_THRES: f64 = 1.0;
    const TAU_V: f64 = 20.0; // leakage timecsale
    const TAU_EXC: f64 = 2.0;
    const TAU_INH: f64 = 2.0;
    const TAU_SLOW: f64 = 750.0;
    const TAU_EXC_SLOW: f64 = 400.0;
    const TAU_STIM: f64 = 2.0;
    const TAU_DECAY: f64 = 384.0;
    const TAU_SK: f64 = 250.0;
    const TAU_HALF_RISE_SK: f64 = 25.0;
    const TAU_HALF_RISE_EXC: f64 = 200.0;
    const STIMULUS_TAU_DECAY: f64 = 2.0;
    const TAU_REFRACTORY: f64 = 2.0;
    const LAMBDA_BG: f64 = 0.15;
    const LAMBDA_ODOR_MAX: f64 = 3.6;
    const LAMBDA_MECH_MAX: f64 = 1.8;
    const STIM_DURATION: f64 = 500.0;
    const SK_MU: f64 = 0.5;
    const SK_STDEV: f64 = 0.2;
    const LN_ODOR_TAU_RISE: f64 = 0.0;
    const LN_MECH_TAU_RISE: f64 = 300.0;
    const LN_S_PN: f64 = 0.006;
    const LN_S_PN_SLOW: f64 = 0.0;
    const LN_S_INH: f64 = 0.015 * 1.15;
    const LN_S_SLOW: f64 = 0.04;
    const LN_L_STIM: f64 = 0.0026;
    const PN_ODOR_TAU_RISE: f64 = 35.0;
    const PN_MECH_TAU_RISE: f64 = 0.0;
    const PN_S_PN: f64 = 0.006;
    const PN_S_PN_SLOW: f64 = 0.02;
    const PN_S_INH: f64 = 0.0169;
    const PN_S_SLOW: f64 = 0.0338;
    const PN_S_STIM: f64 = 0.004;

    fn new(t_stim_on: f64, lambda_odor: f64, lambda_mech: f64, neuron_type: &str, neuron_id: usize) -> Self {
        let s_sk = if neuron_type == "PN" {
            SmallRng::from_entropy().sample(Uniform::new(Neuron::SK_MU, Neuron::SK_STDEV))
        } else {
            0.0
        };

        Neuron {
            t_stim_on,
            t_stim_off: t_stim_on + Neuron::STIM_DURATION,
            exc_times: Vec::new(),
            slow_exc_times: Vec::new(),
            inh_times: Vec::new(),
            slow_inh_times: Vec::new(),
            connected_neurons: Vec::new(),
            neuron_type: neuron_type.to_string(),
            t: 0.0,
            v: 0.0,
            dv_dt: 0.0,
            lambda_odor,
            lambda_mech,
            spike_times: Vec::new(),
            voltages: Vec::new(),
            g_inh_vals: Vec::new(),
            g_slow_vals: Vec::new(),
            g_sk_vals: Vec::new(),
            spike_counts: Vec::new(),
            slow_exc_vals: Vec::new(),
            g_exc_vals: Vec::new(),
            refractory_counter: 0.0,
            n_id: neuron_id,
            total_inhibition: 0,
            total_excitation: 0,
        }
    }

    fn heaviside(num: f64) -> i32 {
        if num >= 0.0 {
            1
        } else {
            0
        }
    }

    fn alpha(&self, tau: f64, spike_time: f64) -> f64 {
        let heaviside_term = (Neuron::heaviside(self.t - spike_time) as f64) / tau;
        let exp_decay_term = (self.t - spike_time) / tau;
        heaviside_term * E.powf(exp_decay_term)
    }

    fn beta(&self, stim_time: f64) -> f64 {
        if self.t <= stim_time + 2.0 * Neuron::TAU_HALF_RISE_SK {
            let heaviside_term = (Neuron::heaviside(self.t - stim_time) as f64) / Neuron::TAU_SK;
            let sigmoid_term_num = E.powf(5.0 * ((self.t - stim_time) - Neuron::TAU_HALF_RISE_SK) / Neuron::TAU_HALF_RISE_SK);
            let sigmoid_term_den =