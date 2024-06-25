use std::f64::consts::E;
use std::cmp::PartialEq;

use rand::{Rng, SeedableRng};
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::rngs::SmallRng;
use vec_to_array::vec_to_array;
use crate::model::NeuronType::PN;

pub(crate) const DELTA_T: f64 = 0.1;

fn random_choice(rng: &mut SmallRng, given_prob: f64) -> bool {
    let rand_val = rng.sample(Uniform::new(0.0, 1.0).unwrap());
    rand_val < given_prob
}

#[derive(PartialEq)]
pub enum NeuronType {
    PN,
    LN
}

pub struct Neuron {

    rng: SmallRng,
    stim_times: Vec<f64>,
    lambda_vals: Vec<f64>,
    g_stim_vals: Vec<f64>,
    mech_vals: Vec<f64>,
    t_stim_on: f64,
    t_stim_off: f64,
    exc_times: Vec<f64>,
    slow_exc_times: Vec<f64>,
    inh_times: Vec<f64>,
    slow_inh_times: Vec<f64>,
    connected_neurons: Vec<Neuron>,
    neuron_type: NeuronType,
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
    odor_tau_rise: f64,
    mech_tau_rise: f64,
    s_pn: f64,
    s_pn_slow: f64,
    s_inh: f64,
    s_slow: f64,
    s_sk: f64,
    s_stim: f64,
    excitation_time: f64,
    slow_excitation_time: f64,
    inhibition_time: f64,
    slow_inhibtion_time: f64,
    excitation_level: f64,
    slow_excitation_level: f64,
    inhibition_level: f64,
    slow_inhibition_level: f64,
    slow_inhibition_time: f64,
    g_exc: f64,
    g_inh: f64,
    g_slow: f64,
    g_stim: f64,
    g_exc_slow: f64,
    g_sk: f64,
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
    const HALF_LAMBDA_ODOR_MAX: f64 = 1.8;
    const LAMBDA_MECH_MAX: f64 = 1.8;
    const HALF_LAMBDA_MECH_MAX: f64 = 0.9;
    const STIM_DURATION: f64 = 500.0;
    const SK_MU: f64 = 0.5;
    const SK_STDEV: f64 = 0.2;
    const LN_ODOR_TAU_RISE: f64 = 0.0;
    const LN_MECH_TAU_RISE: f64 = 300.0;
    const LN_S_PN: f64 = 0.006;
    const LN_S_PN_SLOW: f64 = 0.0;
    const LN_S_INH: f64 = 0.015 * 1.15;
    const LN_S_SLOW: f64 = 0.04;
    const LN_S_STIM: f64 = 0.0026;
    const PN_ODOR_TAU_RISE: f64 = 35.0;
    const PN_MECH_TAU_RISE: f64 = 0.0;
    const PN_S_PN: f64 = 0.006;
    const PN_S_PN_SLOW: f64 = 0.02;
    const PN_S_INH: f64 = 0.0169;
    const PN_S_SLOW: f64 = 0.0338;
    const PN_S_STIM: f64 = 0.004;

    fn new(t_stim_on: f64, lambda_odor: f64, lambda_mech: f64, neuron_type: NeuronType, neuron_id: usize) -> Self {
        let odor_tau_rise = if neuron_type == NeuronType::PN { Neuron::PN_ODOR_TAU_RISE } else { Neuron::LN_ODOR_TAU_RISE };
        let mech_tau_rise = if neuron_type == NeuronType::PN { Neuron::PN_MECH_TAU_RISE } else { Neuron::LN_MECH_TAU_RISE };
        let s_pn = if neuron_type == NeuronType::PN { Neuron::PN_S_PN } else { Neuron::LN_S_PN };
        let s_pn_slow = if neuron_type == NeuronType::PN { Neuron::PN_S_PN_SLOW } else { Neuron::LN_S_PN_SLOW };
        let s_inh = if neuron_type == NeuronType::PN { Neuron::PN_S_INH } else { Neuron::LN_S_INH };
        let s_slow = if neuron_type == NeuronType::PN { Neuron::PN_S_SLOW } else { Neuron::LN_S_SLOW };
        let mut rng: SmallRng = SmallRng::from_thread_rng();
        
        let s_sk =
            if neuron_type == PN {
                rng.sample(Uniform::new(Neuron::SK_MU, Neuron::SK_STDEV).unwrap())
            } else { 0.0 };

        let s_stim= if neuron_type == NeuronType::PN { Neuron::PN_S_STIM } else { Neuron::LN_S_STIM };

        Neuron {
            rng,
            stim_times: Vec::new(),
            lambda_vals: Vec::new(),
            g_stim_vals: Vec::new(),
            mech_vals: Vec::new(),
            t_stim_on,
            t_stim_off: t_stim_on + Neuron::STIM_DURATION,
            exc_times: Vec::new(),
            slow_exc_times: Vec::new(),
            inh_times: Vec::new(),
            slow_inh_times: Vec::new(),
            connected_neurons: Vec::new(),
            neuron_type,
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
            odor_tau_rise,
            mech_tau_rise,
            s_pn,
            s_pn_slow,
            s_inh,
            s_slow,
            s_sk,
            s_stim,
            excitation_time: 0.0,
            slow_excitation_time: 0.0,
            inhibition_time: 0.0,
            slow_inhibtion_time: 0.0,
            excitation_level: 0.0,
            slow_excitation_level: 0.0,
            inhibition_level: 0.0,
            slow_inhibition_level: 0.0,
            slow_inhibition_time: 0.0,
            g_exc: 0.0,
            g_inh: 0.0,
            g_slow: 0.0,
            g_stim: 0.0,
            g_exc_slow: 0.0,
            g_sk: 0.0,
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
        return if self.t <= stim_time + 2.0 * Neuron::TAU_HALF_RISE_SK {
            let heaviside_term = (Neuron::heaviside(self.t - stim_time) as f64) / Neuron::TAU_SK;
            let sigmoid_term_num = E.powf(5.0 * ((self.t - stim_time) - Neuron::TAU_HALF_RISE_SK) / Neuron::TAU_HALF_RISE_SK);
            let sigmoid_term_den = sigmoid_term_num + 1.0;
            (heaviside_term * sigmoid_term_num) / sigmoid_term_den
        } else {
            let exp_decay = -(self.t - (stim_time + (2.0 * Neuron::TAU_HALF_RISE_SK))) / Neuron::TAU_SK;
            (1.0 / Neuron::TAU_SK) * E.powf(exp_decay)
        }
    }

    fn beta_slow_exc(&self, stim_time: f64) -> f64 {
        if self.t <= (stim_time + 2.0 * Neuron::TAU_HALF_RISE_EXC) {
            let heaviside_term = Neuron::heaviside(self.t - stim_time) as f64 / Neuron::TAU_EXC_SLOW;
            let sigmoid_term_num = E.powf((5.0 * ((self.t - stim_time) - Neuron::TAU_HALF_RISE_EXC)) / Neuron::TAU_HALF_RISE_EXC);
            let sigmoid_term_den = 1.0 + sigmoid_term_num;
            heaviside_term * sigmoid_term_num / sigmoid_term_den
        } else {
            let exp_decay = -(self.t - (stim_time + (2.0 * Neuron::TAU_HALF_RISE_EXC))) / Neuron::TAU_EXC_SLOW;
            (1.0 / Neuron::TAU_EXC_SLOW) * E.powf(exp_decay)
        }
    }

    fn g_gen(&self, s_val: f64, tau_val: f64, time: f64) -> f64 {
        s_val * self.alpha(tau_val, time)
    }

    fn g_sk_func(&self) -> f64 {
        self.spike_times.iter().map(|x| self.beta(*x) * self.s_sk).sum()
    }

    fn slow_exc_func(self) -> f64 {
        self.spike_times.iter().map(|x| self.s_pn_slow * self.beta_slow_exc(*x)).sum()
    }

    fn odor_dyn(&self) -> f64 {
        if self.neuron_type == NeuronType::PN {
            if self.t <= self.t_stim_on + 2.0 * self.odor_tau_rise {
                let heaviside_term = Neuron::heaviside(self.t - self.t_stim_on);
                let sigmoid_term_num = E.powf((5.0 * ((self.t - self.t_stim_on) - self.odor_tau_rise)) / self.odor_tau_rise);
                let sigmoid_term_den = 1.0 + sigmoid_term_num;
                heaviside_term as f64 * (sigmoid_term_num / sigmoid_term_den)
            } else if self.t_stim_on + 2.0 * self.odor_tau_rise < self.t && self.t <= self.t_stim_off {
                1.0
            } else {
                return E.powf(-(self.t - self.t_stim_off) / Neuron::TAU_DECAY);
            }
        } else {
            return if self.t <= self.t_stim_off {
                Neuron::heaviside(self.t - self.t_stim_on) as f64
            } else {
                E.powf(-(self.t - self.t_stim_off) / Neuron::TAU_DECAY)
            }
        }
    }

    fn mech_dyn(&self) -> f64 {
        if self.neuron_type == NeuronType::PN {
            if self.t <= self.t_stim_off {
                Neuron::heaviside(self.t - self.t_stim_on) as f64
            } else {
                E.powf(-(self.t - self.t_stim_off) / Neuron::TAU_DECAY)
            }
        } else {
            if self.t <= self.t_stim_on + 2.0 * self.mech_tau_rise {
                let heaviside_term = Neuron::heaviside(self.t - self.t_stim_on);
                let sigmoid_term_num = E.powf((5.0 * ((self.t - self.t_stim_on) - self.mech_tau_rise)) / self.mech_tau_rise);
                let sigmoid_term_den = 1.0 + sigmoid_term_num;
                heaviside_term as f64 * sigmoid_term_num / sigmoid_term_den
            } else if self.t_stim_on + 2.0 * self.mech_tau_rise < self.t && self.t <= self.t_stim_off {
                1.0
            } else {
                E.powf(-(self.t - self.t_stim_off) / Neuron::TAU_DECAY)
            }
        }
    }

    fn lambda_tot(&mut self) -> f64 {
        let odor = self.odor_dyn();
        let mech = self.mech_dyn();
        self.mech_vals.push(mech);
        Neuron::LAMBDA_BG + (self.lambda_odor * odor) + (self.lambda_mech * mech)
    }

    fn inhibit(&mut self) {
        let inhibition_strength = if self.neuron_type == NeuronType::PN { Neuron::PN_S_INH } else { Neuron::LN_S_SLOW };
        let slow_inhibition_strength = if self.neuron_type == NeuronType::PN { Neuron::PN_S_SLOW } else { Neuron::LN_S_SLOW };
        self.excitation_level = self.excitation_level * E.powf(-(self.t - self.excitation_time) / Neuron::TAU_INH) + inhibition_strength;
        self.slow_excitation_level = self.slow_excitation_level + E.powf(-(self.t - self.slow_excitation_time) / Neuron::TAU_SLOW) + slow_inhibition_strength;
        self.inhibition_time = Neuron::TAU_INH;
        self.slow_inhibition_time = Neuron::TAU_SLOW;
    }

    fn excite(&mut self) {
        let excitation_strength = if self.neuron_type == NeuronType::PN { Neuron::PN_S_PN } else { Neuron::LN_S_PN };
        let slow_excitation_strength = if self.neuron_type == NeuronType::PN { Neuron::PN_S_PN_SLOW } else { Neuron::LN_S_PN_SLOW };
        self.excitation_level = self.excitation_level * E.powf(-(self.t - self.excitation_time) / Neuron::TAU_EXC) + excitation_strength;
        self.slow_excitation_level = self.slow_excitation_level + E.powf(-(self.t - self.slow_excitation_time) / Neuron::TAU_EXC_SLOW) + slow_excitation_strength;
        self.excitation_time = Neuron::TAU_EXC;
        self.slow_excitation_time = Neuron::TAU_EXC_SLOW;
    }
    
    fn partition_spike_times(&self, duration: i8, bin_size: i8) -> Vec<Vec<f64>> {
        let mut partition: Vec<Vec<f64>>;
        partition = (1..duration/bin_size).map(|x| Vec::new()).collect();
  
            // fill bins

            let mut index = 0;
            let mut bin_max = bin_size;
            for val in self.spike_times.iter() {
                if *val > bin_max as f64 {
                    index += 1;
                    bin_max += bin_size;
                }
                partition.get_mut(index).expect("Bruh fucked up fix it").push(*val);
            }
        partition
    }
    
    fn spike(&mut self) {
        self.refractory_counter = Neuron::TAU_REFRACTORY;
        self.spike_times.push(self.t);
        self.v = Neuron::V_L;

        if self.neuron_type == NeuronType::LN {
            for neuron in &mut self.connected_neurons {
                neuron.inhibit();
            }
        } else {
            for neuron in &mut self.connected_neurons {
                neuron.excite();
            }
        }
    }

    fn update(&mut self) {
        if self.refractory_counter > 0f64 {
            self.refractory_counter -= DELTA_T
        } else {
            self.v = self.v + self.dv_dt * DELTA_T;
            // poisson model
            let lamb_ = self.lambda_tot();
            let rate = lamb_ * DELTA_T;
            self.lambda_vals.push(rate);

            if self.v >= Neuron::V_THRES {
                self.spike();
            }

            // poisson modeling
            if random_choice(&mut self.rng, rate) {
                self.stim_times.push(self.t)
            }

            self.g_exc = self.g_gen(self.excitation_level, Neuron::TAU_EXC, self.excitation_time);
            self.g_exc_slow = self.g_gen(self.slow_excitation_level, Neuron::TAU_EXC_SLOW, self.excitation_time);
            self.g_inh = self.g_gen(self.inhibition_level, Neuron::TAU_INH, self.inhibition_time);
            self.g_slow = self.g_gen(self.slow_inhibition_level, Neuron::TAU_SLOW, self.slow_inhibition_time);
            self.g_stim = self.g_gen(self.s_stim, Neuron::TAU_STIM, self.slow_excitation_time);

            if self.neuron_type == NeuronType::PN {
                self.g_sk = self.g_sk_func();
                self.g_exc_slow = self.g_gen(self.s_pn_slow, Neuron::TAU_EXC_SLOW, self.slow_excitation_time);
                self.dv_dt = (-1f64 * (self.v - Neuron::V_L) / Neuron::TAU_V) -
                    (self.g_sk * (self.v - Neuron::V_SK)) / Neuron::TAU_SK -
                    (self.g_exc * (self.v - Neuron::V_EXC)) / Neuron::TAU_EXC -
                    (self.g_inh * (self.v - Neuron::V_INH)) / Neuron::TAU_INH -
                    (self.g_slow * (self.v - Neuron::V_INH)) / Neuron::TAU_SLOW -
                    (self.g_exc_slow * (self.v - Neuron::V_EXC)) / Neuron::TAU_EXC_SLOW;
                // (self.g_stim * (self.v - Neuron::V_STIM)) - \
            } else {
                //self.type == "LN"
                self.dv_dt = (-1f64 * (self.v - Neuron::V_L) / Neuron::TAU_V) -
                    (self.g_exc * (self.v - Neuron::V_EXC)) / Neuron::TAU_EXC -
                    (self.g_inh * (self.v - Neuron::V_INH)) / Neuron::TAU_INH -
                    (self.g_slow * (self.v - Neuron::V_INH)) / Neuron::TAU_SLOW -
                    (self.g_exc_slow * (self.v - Neuron::V_EXC)) / Neuron::TAU_EXC_SLOW;
                // (self.g_stim * (self.v - Neuron::V_STIM)) - \
            }

            self.t += DELTA_T;

            self.voltages.push(self.v);
            self.g_inh_vals.push(self.g_inh * 0.4);
            self.g_slow_vals.push(self.g_slow * 5f64);
            self.g_sk_vals.push(self.g_sk * 5f64);
            self.spike_counts.push(self.spike_times.len());
            self.slow_exc_vals.push(self.g_exc_slow);
            self.g_exc_vals.push(self.g_exc);
        }
    }

    fn generate_firing_rates(self, duration: i8, bin_size: i8) -> Vec<f64> {
        let mut rates: Vec<f64> = Vec::new();
        let partition = self.partition_spike_times(duration, bin_size);
        for bin_ in partition {
            rates.push((bin_.len() / (bin_size as usize)) as f64)
        }
        rates
    }
}

struct Glomerulus<'a> {
    stim_time: i32,
    lambda_odor: f64,
    lambda_mech: f64,
    pns: [Neuron; 10],
    lns: [Neuron; 6],
    neurons: [&'a mut Neuron; 16],
    g_id: i8
}

impl<'a> Glomerulus<'a> {
    const PN_PN_PROBABILITY: f64 = 0.02;
    const PN_LN_PROBABILITY: f64 = 0.05;
    const LN_PN_PROBABILITY: f64 = 0.18; 
    const LN_LN_PROBABILITY: f64 = 0.08; // .25


    fn new(stim_time: i32, lambda_odor: f64, lambda_mech: f64, g_id: i8) -> Self {
        let pn_vec: Vec<Neuron> = (1..11).map(|x| Neuron::new(stim_time as f64, lambda_odor, lambda_mech, NeuronType::PN, x)).collect::<Vec<Neuron>>();
        let mut pns = vec_to_array!(pn_vec, Neuron, 10);
        let ln_vec: Vec<Neuron> = (11..17).map(|x| Neuron::new(stim_time as f64, lambda_odor, lambda_mech, NeuronType::LN, x)).collect::<Vec<Neuron>>();
        let mut lns = vec_to_array!(ln_vec, Neuron, 6);
        let neurons_vec: Vec<&mut Neuron> = pns.iter_mut().chain(lns.iter_mut()).collect::<Vec<&mut Neuron>>();
        let neurons = vec_to_array!(neurons_vec, &mut Neuron, 16);
        

        let glomerulus = Glomerulus {
            stim_time,
            lambda_odor,
            lambda_mech,
            pns,
            lns,
            neurons,
            g_id,
        };

        glomerulus
    }



    fn update(&mut self) {
        for neuron in &mut self.neurons {
            neuron.update();
        }
    }
}


pub(crate) enum NetworkType {
    Odor,
    Mech,
    Additive,
    Normalized
}

pub(crate) struct Network<'a> {
    stim_time: i32,
    network_type: NetworkType,
    affected_glomeruli: Box<[i8]>,
    glomeruli: Vec<Glomerulus<'a>>,
}

impl<'a> Network<'a> {
    pub(crate) fn new(stim_time: i32, network_type: NetworkType, affected_glomeruli: &[i8]) {
        let mut glomeruli: Vec<Glomerulus> = Vec::new();
        let mut count: i8 = -1;
        match network_type {
            NetworkType::Odor => {
                for i in 1..6 {
                    count += 1;
                    glomeruli.push(
                        Glomerulus::new(stim_time, Neuron::LAMBDA_ODOR_MAX * affected_glomeruli.iter().filter(|&x| *x == i).count() as f64, 0.0, count)
                    )
                }
            },
            NetworkType::Mech => {
                for i in 1..6 {
                    count += 1;
                    glomeruli.push(
                        Glomerulus::new(stim_time, 0.0, Neuron::LAMBDA_MECH_MAX, count)
                    )
                }
            },
            NetworkType::Additive => {
                for i in 1..6 {
                    count += 1;
                    glomeruli.push(
                        Glomerulus::new(stim_time, Neuron::LAMBDA_ODOR_MAX * affected_glomeruli.iter().filter(|&x| *x == i).count() as f64, Neuron::LAMBDA_MECH_MAX, count)
                    )
                }
            },
            NetworkType::Normalized => {
                for i in 1..6 {
                    count += 1;
                    glomeruli.push(
                        Glomerulus::new(stim_time, Neuron::HALF_LAMBDA_ODOR_MAX * affected_glomeruli.iter().filter(|&x| *x == i).count() as f64, Neuron::HALF_LAMBDA_MECH_MAX, count)
                    )
                }
            }
        }
    }


    fn update(&mut self) {
        for glomerulus in &mut self.glomeruli {
            glomerulus.update()
        }
    }
}

