use std::cell::{RefCell, RefMut};
use std::cmp::PartialEq;
use std::f64::consts::E;

use rand::distributions::{DistIter, Standard};
use rand::Rng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Normal};
use serde::Serialize;
use vec_to_array::vec_to_array;

use crate::{DURATION, STIM_TIME};
use crate::model::NeuronType::PN;

pub(crate) const DELTA_T: f64 = 0.1;

#[derive(PartialEq, Debug)]
pub enum NeuronType {
    PN,
    LN,
}

pub struct Neuron {
    stim_times: Vec<f64>,
    pub lambda_vals: Vec<f64>,
    pub g_stim_vals: Vec<f64>,
    mech_vals: Vec<f64>,
    t_stim_on: f64,
    t_stim_off: f64,
    neuron_type: NeuronType,
    t: f64,
    v: f64,
    dv_dt: f64,
    lambda_odor: f64,
    lambda_mech: f64,
    pub spike_times: Vec<f64>,
    pub voltages: Vec<f64>,
    pub g_inh_vals: Vec<f64>,
    pub g_slow_vals: Vec<f64>,
    pub g_sk_vals: Vec<f64>,
    pub spike_counts: Vec<usize>,
    pub g_slow_exc_vals: Vec<f64>,
    pub g_exc_vals: Vec<f64>,
    pub dv_dt_vals: Vec<f64>,
    refractory_counter: f64,
    odor_tau_rise: f64,
    mech_tau_rise: f64,
    s_exc: f64,
    s_exc_slow: f64,
    s_inh: f64,
    s_slow: f64,
    s_sk: f64,
    s_stim: f64,
    pub excitation_times: Vec<f64>,
    pub inhibition_times: Vec<f64>,
    g_exc: f64,
    g_inh: f64,
    g_slow: f64,
    g_stim: f64,
    g_exc_slow: f64,
    g_sk: f64,
    pub(crate) count: usize,
}

impl Neuron {
    const V_L: f64 = 0.0;
    const V_EXC: f64 = 14.0 / 3.0;
    const V_STIM: f64 = 14.0 / 3.0;
    const V_INH: f64 = -2.0 / 3.0;
    const V_SK: f64 = -2.0 / 3.0;
    const V_THRES: f64 = 1.0;
    const TAU_V: f64 = 20.0; // leakage timescale
    const TAU_EXC: f64 = 2.0;
    const TAU_INH: f64 = 2.0;
    const TAU_SLOW: f64 = 750.0;
    const TAU_EXC_SLOW: f64 = 850.0;
    const TAU_STIM: f64 = 2.0;
    const TAU_DECAY: f64 = 384.0; 
    const TAU_SK: f64 = 250.0;
    const TAU_HALF_RISE_SK: f64 = 25.0;
    const TAU_HALF_RISE_EXC: f64 = 70.0;
    const STIMULUS_TAU_DECAY: f64 = 2.0;
    const TAU_REFRACTORY: f64 = 2.0;
    const BG_SPIKES_PER_MS: f64 = 3.6;
    const LAMBDA_ODOR_MAX: f64 = 3.6;
    const HALF_LAMBDA_ODOR_MAX: f64 = Neuron::LAMBDA_ODOR_MAX * 0.5;
    const LAMBDA_MECH_MAX: f64 = 1.8;
    const HALF_LAMBDA_MECH_MAX: f64 = Neuron::LAMBDA_MECH_MAX * 0.5;
    const STIM_DURATION: f64 = 500.0;
    const SK_MU: f64 = 0.5;
    const SK_STDEV: f64 = 0.2;
    const LN_ODOR_TAU_RISE: f64 = 0.0;
    const LN_MECH_TAU_RISE: f64 = 150.0;
    const LN_S_PN: f64 = 0.006;
    const LN_S_PN_SLOW: f64 = 0.0;
    const LN_S_INH: f64 = 0.015;
    const LN_S_SLOW: f64 = 0.04;
    const LN_S_STIM: f64 = 0.0031;
    const PN_ODOR_TAU_RISE: f64 = 35.0;
    const PN_MECH_TAU_RISE: f64 = 0.0;
    const PN_S_PN: f64 = 0.01;
    const PN_S_PN_SLOW: f64 = 0.02;
    const PN_S_INH: f64 = 0.0169;
    const PN_S_SLOW: f64 = 0.0338;
    const PN_S_STIM: f64 = 0.004;

    fn new(
        t_stim_on: f64,
        lambda_odor: f64,
        lambda_mech: f64,
        neuron_type: NeuronType,
        rng: &mut SmallRng,
        duration: usize,
    ) -> Self {
        let odor_tau_rise = if neuron_type == PN {
            Neuron::PN_ODOR_TAU_RISE
        } else {
            Neuron::LN_ODOR_TAU_RISE
        };
        let mech_tau_rise = if neuron_type == PN {
            Neuron::PN_MECH_TAU_RISE
        } else {
            Neuron::LN_MECH_TAU_RISE
        };
        let s_pn = if neuron_type == PN {
            Neuron::PN_S_PN
        } else {
            Neuron::LN_S_PN
        };
        let s_pn_slow = if neuron_type == PN {
            Neuron::PN_S_PN_SLOW
        } else {
            Neuron::LN_S_PN_SLOW
        };
        let s_inh = if neuron_type == PN {
            Neuron::PN_S_INH
        } else {
            Neuron::LN_S_INH
        };
        let s_slow = if neuron_type == PN {
            Neuron::PN_S_SLOW
        } else {
            Neuron::LN_S_SLOW
        };

        let s_sk = if neuron_type == PN {
            rng.sample(Normal::new(Neuron::SK_MU, Neuron::SK_STDEV).unwrap())
        } else {
            0.0
        };

        let s_stim = if neuron_type == PN {
            Neuron::PN_S_STIM
        } else {
            Neuron::LN_S_STIM
        };
        let num_iterations = (duration as f64 * DELTA_T) as usize;
        Neuron {
            stim_times: Vec::new(),
            lambda_vals: Vec::with_capacity(num_iterations),
            g_stim_vals: Vec::with_capacity(num_iterations),
            mech_vals: Vec::with_capacity(num_iterations),
            t_stim_on,
            t_stim_off: t_stim_on + Neuron::STIM_DURATION,
            neuron_type,
            t: 0.0,
            v: 0.0,
            dv_dt: 0.0,
            lambda_odor,
            lambda_mech,
            spike_times: Vec::new(),
            voltages: Vec::with_capacity(num_iterations),
            g_inh_vals: Vec::with_capacity(num_iterations),
            g_slow_vals: Vec::with_capacity(num_iterations),
            g_sk_vals: Vec::with_capacity(num_iterations),
            spike_counts: Vec::with_capacity(num_iterations),
            g_slow_exc_vals: Vec::with_capacity(num_iterations),
            g_exc_vals: Vec::with_capacity(num_iterations),
            dv_dt_vals: Vec::with_capacity(num_iterations),
            refractory_counter: 0.0,
            odor_tau_rise,
            mech_tau_rise,
            s_exc: s_pn,
            s_exc_slow: s_pn_slow,
            s_inh,
            s_slow,
            s_sk,
            s_stim,
            excitation_times: Vec::new(),
            inhibition_times: Vec::new(),
            g_exc: 0.0,
            g_inh: 0.0,
            g_slow: 0.0,
            g_stim: 0.0,
            g_exc_slow: 0.0,
            g_sk: 0.0,
            count: 0,
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
        heaviside_term * E.powf(-exp_decay_term)
    }

    fn beta(&self, stim_time: f64) -> f64 {
        if self.t <= stim_time + 2.0 * Neuron::TAU_HALF_RISE_SK {
            let heaviside_term = (Neuron::heaviside(self.t - stim_time) as f64) / Neuron::TAU_SK;
            let sigmoid_term_num = E.powf(
                5.0 * ((self.t - stim_time) - Neuron::TAU_HALF_RISE_SK) / Neuron::TAU_HALF_RISE_SK,
            );
            let sigmoid_term_den = sigmoid_term_num + 1.0;
            (heaviside_term * sigmoid_term_num) / sigmoid_term_den
        } else {
            let exp_decay =
                (self.t - (stim_time + (2.0 * Neuron::TAU_HALF_RISE_SK))) / Neuron::TAU_SK;
            (1.0 / Neuron::TAU_SK) * E.powf(-exp_decay)
        }
    }

    fn beta_slow_exc(&self, stim_time: f64) -> f64 {
        if self.t <= (stim_time + 2.0 * Neuron::TAU_HALF_RISE_EXC) {
            let heaviside_term =
                Neuron::heaviside(self.t - stim_time) as f64 / Neuron::TAU_EXC_SLOW;
            let sigmoid_term_num = E.powf(
                (5.0 * ((self.t - stim_time) - Neuron::TAU_HALF_RISE_EXC))
                    / Neuron::TAU_HALF_RISE_EXC,
            );
            let sigmoid_term_den = 1.0 + sigmoid_term_num;
            heaviside_term * sigmoid_term_num / sigmoid_term_den
        } else {
            let exp_decay =
                (self.t - (stim_time + (2.0 * Neuron::TAU_HALF_RISE_EXC))) / Neuron::TAU_EXC_SLOW;
            (1.0 / Neuron::TAU_EXC_SLOW) * E.powf(-exp_decay)
        }
    }

    fn g_gen(&self, s_val: f64, tau_val: f64, s_set: &Vec<f64>) -> f64 { 
        s_set.iter().map(|s| s_val * self.alpha(tau_val, *s)).sum()
    }

    fn g_sk_func(&self) -> f64 {
        self.spike_times
            .iter()
            .map(|x| self.beta(*x) * self.s_sk)
            .sum()
    }

    fn g_slow_exc(&self) -> f64 {
        self.excitation_times
            .iter()
            .map(|x| self.s_exc_slow * self.beta_slow_exc(*x))
            .sum()
    }

    fn odor_dyn(&self) -> f64 {
        if self.neuron_type == PN {
            if self.t <= self.t_stim_on + 2.0 * self.odor_tau_rise {
                let heaviside_term = Neuron::heaviside(self.t - self.t_stim_on);
                let sigmoid_term_num = E.powf(
                    (5.0 * ((self.t - self.t_stim_on) - self.odor_tau_rise)) / self.odor_tau_rise,
                );
                let sigmoid_term_den = 1.0 + sigmoid_term_num;
                heaviside_term as f64 * (sigmoid_term_num / sigmoid_term_den)
            } else if self.t_stim_on + 2.0 * self.odor_tau_rise < self.t
                && self.t <= self.t_stim_off
            {
                1.0
            } else {
                return E.powf(-(self.t - self.t_stim_off) / Neuron::TAU_DECAY);
            } 
        } else {
            return if self.t <= self.t_stim_off {
                Neuron::heaviside(self.t - self.t_stim_on) as f64
            } else {
                E.powf(-(self.t - self.t_stim_off) / Neuron::TAU_DECAY)
            };
        }
    }

    fn mech_dyn(&self) -> f64 {
        if self.neuron_type == PN {
            if self.t <= self.t_stim_off {
                Neuron::heaviside(self.t - self.t_stim_on) as f64
            } else {
                E.powf(-(self.t - self.t_stim_off) / Neuron::TAU_DECAY)
            }
        } else {
            if self.t <= self.t_stim_on + 2.0 * self.mech_tau_rise {
                let heaviside_term = Neuron::heaviside(self.t - self.t_stim_on);
                let sigmoid_term_num = E.powf(
                    (5.0 * ((self.t - self.t_stim_on) - self.mech_tau_rise)) / self.mech_tau_rise,
                );
                let sigmoid_term_den = 1.0 + sigmoid_term_num;
                heaviside_term as f64 * sigmoid_term_num / sigmoid_term_den
            } else if self.t_stim_on + 2.0 * self.mech_tau_rise < self.t
                && self.t <= self.t_stim_off
            {
                1.0
            } else {
                E.powf(-(self.t - self.t_stim_off) / Neuron::TAU_DECAY)
            }
        }
    }

    fn incoming_spikes_per_ms(&mut self) -> f64 {
        let odor = self.odor_dyn();
        let mech = self.mech_dyn();
        self.mech_vals.push(mech);
        Neuron::BG_SPIKES_PER_MS + (self.lambda_odor * odor) + (self.lambda_mech * mech)
    }

    fn partition_spike_times(&self, duration: i8, bin_size: i8) -> Vec<Vec<f64>> {
        let mut partition: Vec<Vec<f64>>;
        partition = (1..duration / bin_size).map(|_| Vec::new()).collect();

        // fill bins

        let mut index = 0;
        let mut bin_max = bin_size;
        for val in self.spike_times.iter() {
            if *val > bin_max as f64 {
                index += 1;
                bin_max += bin_size;
            }
            partition
                .get_mut(index)
                .expect("Bruh fucked up fix it")
                .push(*val);
        }
        partition
    }

    fn update(&mut self, rng: &RefCell<SmallRng>, t: f64) -> bool {
        if self.refractory_counter > 0f64 {
            self.refractory_counter -= DELTA_T;
        } else {
            self.v = self.v + self.dv_dt * DELTA_T;

            if self.v >= Neuron::V_THRES {
                self.spike_times.push(self.t);
                self.v = Neuron::V_L;
                self.refractory_counter = Neuron::TAU_REFRACTORY;
                self.t += DELTA_T;
                if self.count % 10 == 0 {
                    self.record_values();
                }
                self.count += 1;
                return true;
            }

            //poisson model
            let lamb_ = self.incoming_spikes_per_ms() * 0.1;
            self.lambda_vals.push(lamb_);
            let int_val: usize = lamb_.trunc() as usize;
            self.stim_times.extend((0..int_val).map(|_| t));
            if rng.borrow_mut().gen_bool(lamb_ - int_val as f64) {
                self.stim_times.push(t)
            }
            self.g_exc = self.g_gen(self.s_exc, Neuron::TAU_EXC, &self.excitation_times);
            self.g_exc_slow = self.g_slow_exc();
            self.g_inh = self.g_gen(self.s_inh, Neuron::TAU_INH, &self.inhibition_times);
            self.g_slow = self.g_gen(self.s_slow, Neuron::TAU_SLOW, &self.inhibition_times);
            self.g_stim = self.g_gen(self.s_stim, Neuron::TAU_STIM, &self.stim_times);
            if self.neuron_type == PN {
                self.g_sk = self.g_sk_func();
                self.dv_dt = (-1f64 * (self.v - Neuron::V_L) / Neuron::TAU_V)
                    - (self.g_sk * (self.v - Neuron::V_SK))
                    - (self.g_exc * (self.v - Neuron::V_EXC))
                    - (self.g_inh * (self.v - Neuron::V_INH))
                    - (self.g_slow * (self.v - Neuron::V_INH))
                    - (self.g_exc_slow * (self.v - Neuron::V_EXC))
                    - (self.g_stim * (self.v - Neuron::V_STIM))
            } else {
                //self.type == "LN"
                self.dv_dt = (-1f64 * (self.v - Neuron::V_L) / Neuron::TAU_V)
                    - (self.g_exc * (self.v - Neuron::V_EXC)) 
                    - (self.g_inh * (self.v - Neuron::V_INH))
                    - (self.g_slow * (self.v - Neuron::V_INH)) 
                    - (self.g_exc_slow * (self.v - Neuron::V_EXC))
                    - (self.g_stim * (self.v - Neuron::V_STIM))
            };
        }
        if t as i32 % (Neuron::TAU_STIM * 3.0) as i32 == 0 {
            self.stim_times = self
                .stim_times
                .iter()
                .filter(|x| **x - t < Neuron::TAU_STIM * 3.0)
                .map(|x| *x)
                .collect();
        }
        if t as i32 % (Neuron::TAU_EXC_SLOW * 3.0) as i32 == 0 {
            self.excitation_times = self
                .excitation_times
                .iter()
                .filter(|x| **x - t < Neuron::TAU_EXC_SLOW * 3.0)
                .map(|x| *x)
                .collect();
        }
        if t as i32 % (Neuron::TAU_SLOW * 3.0) as i32 == 0 {
            self.inhibition_times = self
                .inhibition_times
                .iter()
                .filter(|x| **x - t < Neuron::TAU_SLOW * 3.0)
                .map(|x| *x)
                .collect();
        }
        if t as i32 % (Neuron::TAU_SK * 3.0) as i32 == 0 {
            self.spike_times = self
                .spike_times
                .iter()
                .filter(|x| **x - t < Neuron::TAU_SK * 3.0)
                .map(|x| *x)
                .collect();
        }
        
        if self.count % 10 == 0 {
            self.record_values()
        }
        self.count += 1;
        self.t += DELTA_T;
        false
    }

    fn record_values(&mut self) {
        self.voltages.push(self.v);
        self.spike_counts.push(self.spike_times.len());
        self.g_exc_vals.push(self.g_exc);
        self.g_slow_vals.push(self.g_slow);
        self.g_inh_vals.push(self.g_inh);
        self.g_slow_exc_vals.push(self.g_exc_slow);
        self.g_stim_vals.push(self.g_stim);
        self.g_sk_vals.push(self.g_sk);
        self.dv_dt_vals.push(self.dv_dt);
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

#[derive(Debug, Serialize)]
struct SerializedNeuron {
    neuron_type: String,
    voltages: Vec<f64>,
    excitation_vals: Vec<f64>,
    slow_excitation_vals: Vec<f64>,
    inhibition_vals: Vec<f64>,
    slow_inhibition_vals: Vec<f64>,
    g_sk_vals: Vec<f64>,
    spike_times: Vec<f64>,
    stim_vals: Vec<f64>,
    dv_dt_vals: Vec<f64>,
}

impl SerializedNeuron {
    fn from_neuron(neuron: &Neuron) -> Self {
        SerializedNeuron {
            neuron_type: if neuron.neuron_type == NeuronType::PN {
                String::from("pn")
            } else {
                String::from("ln")
            },
            voltages: neuron.voltages.clone(),
            excitation_vals: neuron.g_exc_vals.clone(),
            slow_excitation_vals: neuron.g_slow_exc_vals.clone(),
            inhibition_vals: neuron.g_inh_vals.clone(),
            slow_inhibition_vals: neuron.g_slow_vals.clone(),
            g_sk_vals: neuron.g_sk_vals.clone(),
            spike_times: neuron.spike_times.clone(),
            stim_vals: neuron.g_stim_vals.clone(),
            dv_dt_vals: neuron.dv_dt_vals.clone(),
        }
    }
}

pub(crate) enum NetworkType {
    Odor,
    Mech,
    Additive,
    Normalized,
}

pub(crate) struct Network {
    pub neurons: [Neuron; 96],
    pub connectivity_matrix: [Vec<usize>; 96],
    rng: RefCell<SmallRng>,
    t: f64,
}



impl Network {
    const NUM_PNS: usize = 4;
    const NUM_LNS: usize = 12;
    const PN_PN_PROBABILITY: f64 = 0.30; // MODIFIED: originally 0.75
    const PN_LN_PROBABILITY: f64 = 0.55;
    const LN_PN_PROBABILITY: f64 = 0.35;
    const LN_LN_PROBABILITY: f64 = 0.25; // .25
    pub(crate) fn new(
        stim_time: i32,
        network_type: NetworkType,
        affected_glomeruli: [i8; 6],
        duration: usize,
    ) -> Self {
        let rng_struct: RefCell<SmallRng> = RefCell::new(SmallRng::from_thread_rng());

        let mut neurons_vec: Vec<Neuron> = Vec::new();
        const DUMMY_VEC: Vec<usize> = Vec::new();
        let mut connectivity_matrix: [Vec<usize>; 96] = [DUMMY_VEC; 96];

        for i in 0..6 {
            let mut rng: RefMut<SmallRng> = rng_struct.borrow_mut();

            // assign stimulus amounts based on network type and "glomerulus"
            let (odor_val, mech_val) = match network_type {
                NetworkType::Odor => (
                    Neuron::LAMBDA_ODOR_MAX
                        * affected_glomeruli.iter().filter(|&x| *x == i).count() as f64,
                    0.0,
                ),
                NetworkType::Mech => (0.0, Neuron::LAMBDA_MECH_MAX),
                NetworkType::Additive => (
                    Neuron::LAMBDA_ODOR_MAX
                        * affected_glomeruli.iter().filter(|&x| *x == i).count() as f64,
                    Neuron::LAMBDA_MECH_MAX,
                ),
                NetworkType::Normalized => (
                    Neuron::HALF_LAMBDA_ODOR_MAX
                        * affected_glomeruli.iter().filter(|&x| *x == i).count() as f64,
                    Neuron::HALF_LAMBDA_MECH_MAX,
                ),
            };

            for j in 0..16 {
                if (0 <= j) && (j < Network::NUM_PNS) {
                    neurons_vec.push(Neuron::new(
                        stim_time as f64,
                        odor_val,
                        mech_val,
                        NeuronType::PN,
                        &mut rng,
                        duration,
                    ));
                    let mut random_vals: DistIter<Standard, &mut SmallRng, f64> =
                        Standard.sample_iter(&mut rng);
                    // intraglomerular PN connections
                    let intermediate_val = random_vals
                        .by_ref()
                        .take(16)
                        .enumerate()
                        .filter(|&(k, x)| {
                            (k != (j) as usize)
                                && (if k < Network::NUM_PNS {
                                    // neurons 0 to 9 are PNs...
                                    x < Network::PN_PN_PROBABILITY
                                } else {
                                    // ... and neurons 10 to 15 are LNs.
                                    x < Network::PN_LN_PROBABILITY
                                })
                        })
                        .collect::<Vec<(usize, f64)>>();
                    
                    connectivity_matrix[(i as usize * 16 + j) as usize] = intermediate_val
                        .into_iter()
                        .chain(random_vals.take(96).enumerate().filter(|&(k, x)| {
                            (k != (i as usize * 16 + j) as usize)
                                && ((k % 16 < Network::NUM_PNS) && (x < Network::LN_PN_PROBABILITY))
                        }))
                        .map(|(k, _)| k)
                        .collect::<Vec<usize>>();
                    
                    
                } else {
                    neurons_vec.push(Neuron::new(
                        stim_time as f64,
                        odor_val,
                        mech_val,
                        NeuronType::LN,
                        &mut rng,
                        duration,
                    ));
                    let mut random_vals: DistIter<Standard, &mut SmallRng, f64> =
                        Standard.sample_iter(&mut rng);
                    // intraglomerular LN connections
                    let intermediate_val = random_vals
                        .by_ref()
                        .take(16)
                        .enumerate()
                        .filter(|&(k, x)| {
                            (k != (j) as usize)
                                && (if k < Network::NUM_PNS {
                                    x < Network::LN_PN_PROBABILITY
                                } else {
                                    x < Network::LN_LN_PROBABILITY
                                })
                        })
                        .collect::<Vec<(usize, f64)>>();

                    connectivity_matrix[(i as usize * 16 + j) as usize] = intermediate_val
                        .into_iter()
                        .chain(random_vals.take(96).enumerate().filter(|&(k, x)| {
                            (k != (i as usize * 16 + j) as usize)
                                && ((k % 16 < Network::NUM_PNS) && (x < Network::LN_PN_PROBABILITY))
                        }))
                        .map(|(k, _)| k)
                        .collect::<Vec<usize>>();
                }
            }
        }

        let neurons: [Neuron; 96] = vec_to_array!(neurons_vec, Neuron, 96);
        println!("HERE DOOFUS{:?}", connectivity_matrix);
        Network {
            neurons,
            connectivity_matrix,
            rng: rng_struct,
            t: 0.0,
        }
    }

    pub(crate) fn update(&mut self, t: f64) {
        let mut spiked_neurons: Vec<usize> = Vec::new();
        for (i, neuron) in &mut self.neurons.iter_mut().enumerate() {
            if neuron.update(&self.rng, t) {
                spiked_neurons.push(i);
            }
        }

        for index in spiked_neurons {
            for &target_index in &self.connectivity_matrix[index] {
                let neuron_type = &self.neurons[index].neuron_type;
                match neuron_type {
                    NeuronType::PN => {
                        self.neurons[target_index]
                            .excitation_times
                            .push(self.t + DELTA_T);
                    }
                    NeuronType::LN => {
                        self.neurons[target_index]
                            .inhibition_times
                            .push(self.t + DELTA_T);
                    }
                }
            }
        }

        self.t += DELTA_T;
    }
}

#[derive(Debug, Serialize)]
pub struct SerializedNetwork {
    dir: String,
    neurons: Vec<SerializedNeuron>,
    stim_time: i32,
    duration: usize,
    delta_t: f64,
}

impl SerializedNetwork {
    pub fn from_network(network: Network, dir: String) -> Self {
        let serialized_neurons: Vec<SerializedNeuron> = network
            .neurons
            .iter()
            .map(|neuron| SerializedNeuron::from_neuron(neuron))
            .collect();
        SerializedNetwork {
            dir,
            neurons: serialized_neurons,
            stim_time: STIM_TIME,
            duration: DURATION,
            delta_t: DELTA_T,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Config {
    V_EXC: f64,
    V_INH: f64,
    TAU_INH: f64,
    TAU_SLOW: f64,
    TAU_EXC: f64,
    TAU_EXC_SLOW: f64,
    TAU_STIM: f64,
    TAU_SK: f64,
    STIM_TIME: i32,
    STIM_DURATION: f64,
    SK_MU: f64,
    SK_STDEV: f64,
    LN_ODOR_TAU_RISE: f64,
    LN_MECH_TAU_RISE: f64,
    LN_S_PN: f64,
    LN_S_PN_SLOW: f64,
    LN_S_INH: f64,
    LN_S_SLOW: f64,
    LN_S_STIM: f64,
    PN_ODOR_TAU_RISE: f64,
    PN_MECH_TAU_RISE: f64,
    HALF_RISE_SLOW_EXC: f64,
    PN_S_PN: f64,
    PN_S_PN_SLOW: f64,
    PN_S_INH: f64,
    PN_S_SLOW: f64,
    PN_S_STIM: f64,
    PN_PN_PROB: f64,
    PN_LN_PROB: f64,
    LN_PN_PROB: f64,
    LN_LN_PROB: f64,
}

impl Config {
    pub fn create() -> Self {
        Config {
            V_EXC: Neuron::V_EXC,
            V_INH: Neuron::V_INH,
            TAU_INH: Neuron::TAU_INH,
            TAU_SLOW: Neuron::TAU_SLOW,
            TAU_EXC: Neuron::TAU_EXC,
            TAU_EXC_SLOW: Neuron::TAU_EXC_SLOW,
            TAU_STIM: Neuron::TAU_STIM,
            TAU_SK: Neuron::TAU_SK,
            STIM_TIME,
            STIM_DURATION: Neuron::STIM_DURATION,
            SK_MU: Neuron::SK_MU,
            SK_STDEV: Neuron::SK_STDEV,
            LN_ODOR_TAU_RISE: Neuron::LN_ODOR_TAU_RISE,
            LN_MECH_TAU_RISE: Neuron::LN_MECH_TAU_RISE,
            HALF_RISE_SLOW_EXC: Neuron::TAU_HALF_RISE_EXC,
            LN_S_PN: Neuron::LN_S_PN,
            LN_S_PN_SLOW: Neuron::LN_S_PN_SLOW,
            LN_S_INH: Neuron::LN_S_INH,
            LN_S_SLOW: Neuron::LN_S_SLOW,
            LN_S_STIM: Neuron::LN_S_STIM,
            PN_ODOR_TAU_RISE: Neuron::PN_ODOR_TAU_RISE,
            PN_MECH_TAU_RISE: Neuron::LN_MECH_TAU_RISE,

            PN_S_PN: Neuron::PN_S_PN,
            PN_S_PN_SLOW: Neuron::PN_S_PN_SLOW,
            PN_S_INH: Neuron::PN_S_INH,
            PN_S_SLOW: Neuron::PN_S_SLOW,
            PN_S_STIM: Neuron::PN_S_STIM,
            PN_PN_PROB: Network::PN_PN_PROBABILITY,
            PN_LN_PROB: Network::PN_LN_PROBABILITY,
            LN_PN_PROB: Network::LN_PN_PROBABILITY,
            LN_LN_PROB: Network::LN_LN_PROBABILITY,
        }
    }
}
