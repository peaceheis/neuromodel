from copy import copy
import json

import matplotlib.pyplot as plt
import numpy as np

SEED = 1432847328947938279
NP_RANDOM_GENERATOR = np.random.default_rng(SEED)
DELTA_T = 0.1


def random_choice(given_prob: float):
    rand_val = NP_RANDOM_GENERATOR.uniform(0, 1)
    if rand_val < given_prob:
        return True
    else:
        return False


class Neuron:
    V_L = 0.0
    V_EXC = V_STIM = 14 / 3
    V_INH = V_SK = -2 / 3
    V_THRES = 1
    # tau values in milliseconds
    TAU_V = 20  # leakage timecsale
    TAU_EXC = 2
    TAU_INH = 2
    TAU_SLOW = 750
    TAU_EXC_SLOW = 780
    TAU_STIM = 2
    TAU_DECAY = 384
    TAU_SK = 250
    TAU_HALF_RISE_SK = 25
    STIMULUS_TAU_DECAY = 2
    TAU_REFRACTORY = 2

    # input source rates
    LAMBDA_BG = 1  # spikes / ms
    LAMBDA_ODOR_MAX = 3.6  # spikes / ms
    LAMBDA_MECH_MAX = 1.8  # spikes / ms

    STIM_DURATION = 500  # ms

    SK_MU = 0.5
    SK_STDEV = 0.2

    def __init__(self, t_stim_on: float, lambda_odor: float, lambda_mech: float, neuron_type: str, neuron_id: int = 0):
        self.stim_times = []
        self.lambda_vals = []
        self.g_stim_vals = []
        self.mech_vals = []
        self.dv_vals = []
        self.t_stim_on = t_stim_on
        self.t_stim_off = t_stim_on + Neuron.STIM_DURATION
        self.exc_times = []
        self.inh_times = []
        self.g_sk_vals = []
        self.connected_neurons = []
        self.neuron_type = neuron_type
        self.t = 0
        self.v = 0
        self.dv_dt = 0
        self.lambda_odor = lambda_odor
        self.lambda_mech = lambda_mech
        self.spike_times = []
        self.spike_counts = []
        self.voltages = []
        self.g_sk_vals = []
        self.g_inh_vals = []
        self.g_slow_vals = []
        self.refractory_counter = 0.0
        self.n_id = neuron_id
        self.total_inhibition: int = 0
        self.total_excitation: int = 0
        # 4.2.1 The neuron model
        # reversal potentials (nondimensional)

        if self.neuron_type == "LN":
            self.odor_tau_rise = 0
            self.mech_tau_rise = 300
            self.s_pn = 0.006 * .60
            self.s_pn_slow = self.s_pn * 0
            self.s_inh = 0.015
            self.s_slow = 0.04
            self.s_stim = 0.0026

        else:  # self.type == "PN"
            self.odor_tau_rise = 35
            self.mech_tau_rise = 0
            self.s_pn = 0.01
            self.s_pn_slow = 0.01
            self.s_inh = 0.0169 * .85
            self.s_slow = 0.0338
            self.s_stim = 0.004
            self.s_sk = NP_RANDOM_GENERATOR.normal(Neuron.SK_MU, Neuron.SK_STDEV)
            if self.s_sk < 0:
                self.s_sk = 0

        # conductance values
        self.g_sk = 0.0
        self.g_stim = 0.0
        self.g_slow = 0.0
        self.g_inh = 0.0
        self.g_exc = 0.0
        self.g_exc_slow = 0.0

    def __repr__(self):
        return f"Neuron {self.n_id}"

    @staticmethod
    def Heaviside(num: float | int) -> int:  # gives the h value for alpha_exc function
        if num >= 0:
            return 1
        if num < 0:
            return 0

    def alpha(self, tau: float, spike_time: float | int) -> float:
        heaviside_term = (self.Heaviside(self.t - spike_time)) / tau
        exp_decay_term = np.exp((self.t - spike_time) / tau)
        return heaviside_term * exp_decay_term

    def beta(self, stim_time: float) -> float:
        if self.t <= (stim_time + 2 * Neuron.TAU_HALF_RISE_SK):
            heaviside_term = self.Heaviside(self.t - stim_time) / self.TAU_SK
            sigmoid_term_num = np.exp((5 * ((self.t - stim_time) - Neuron.TAU_HALF_RISE_SK)) / Neuron.TAU_HALF_RISE_SK)
            sigmoid_term_den = 1 + sigmoid_term_num
            return heaviside_term * sigmoid_term_num / sigmoid_term_den
        else:
            exp_decay = -(self.t - (stim_time + (2 * Neuron.TAU_HALF_RISE_SK))) / self.TAU_SK
            return (1 / self.TAU_SK) * np.exp(exp_decay)

    def g_gen(self, s_val, tau_val: float, s_set: list) -> float:
        return np.sum([s_val * self.alpha(tau_val, s) for s in s_set])

    def g_sk_func(self):
        return np.sum([self.s_sk * self.beta(s) for s in self.spike_times])

    def odor_dyn(self) -> float:
        if self.neuron_type == "PN":
            if self.t <= self.t_stim_on + 2 * self.odor_tau_rise:
                heaviside_term = self.Heaviside(self.t - self.t_stim_on)
                sigmoid_term_num = np.exp((5 * ((self.t - self.t_stim_on) - self.odor_tau_rise)) / self.odor_tau_rise)
                sigmoid_term_den = 1 + sigmoid_term_num
                return heaviside_term * (sigmoid_term_num / sigmoid_term_den)
            elif self.t_stim_on + 2 * self.odor_tau_rise < self.t <= self.t_stim_off:
                return 1
            else:
                return np.exp(-(self.t - self.t_stim_off) / Neuron.TAU_DECAY)
        else:
            if self.t <= self.t_stim_off:
                return self.Heaviside(self.t - self.t_stim_on)
            else:
                return np.exp(-(self.t - self.t_stim_off) / Neuron.TAU_DECAY)

    def mech_dyn(self) -> float:
        if self.neuron_type == "PN":
            if self.t <= self.t_stim_off:
                return self.Heaviside(self.t - self.t_stim_on)
            else:
                return np.exp(-(self.t - self.t_stim_off) / Neuron.TAU_DECAY)
        else:
            if self.t <= self.t_stim_on + 2 * self.mech_tau_rise:
                heaviside_term = self.Heaviside(self.t - self.t_stim_on)
                sigmoid_term_num = np.exp((5 * ((self.t - self.t_stim_on) - self.mech_tau_rise)) / self.mech_tau_rise)
                sigmoid_term_den = 1 + sigmoid_term_num
                return heaviside_term * sigmoid_term_num / sigmoid_term_den
            elif self.t_stim_on + 2 * self.mech_tau_rise < self.t <= self.t_stim_off:
                return 1
            else:
                return np.exp(-(self.t - self.t_stim_off) / Neuron.TAU_DECAY)

    def lambda_tot(self) -> float:
        odor = self.odor_dyn()
        mech = self.mech_dyn()
        self.mech_vals.append(mech)
        return Neuron.LAMBDA_BG + (self.lambda_odor * odor) + (self.lambda_mech * mech)

    def spike(self):
        self.refractory_counter = Neuron.TAU_REFRACTORY
        self.spike_times.append(self.t)
        self.v = Neuron.V_L

        if self.neuron_type == "LN":
            for neuron in self.connected_neurons:
                neuron.inh_times.append(self.t)
        else:
            for neuron in self.connected_neurons:
                neuron.exc_times.append(self.t)

    def filter_exc_times(self):
        for val in self.exc_times:
            if self.t - val > Neuron.TAU_EXC * 3:
                self.exc_times.remove(val)
            else:
                return

    def filter_inh_times(self):
        for val in self.inh_times:
            if self.t - val > Neuron.TAU_INH * 3:
                self.inh_times.remove(val)
            else:
                return

    def filter_stim_times(self):
        for val in self.stim_times:
            if self.t - val > Neuron.TAU_STIM * 3:
                self.stim_times.remove(val)
            else:
                return

    def generate_firing_rates(self, duration, bin_size):
        if duration % bin_size != 0:
            raise Exception("bin size should divide duration")

        num_bins = int(duration / bin_size)

        if num_bins <= 1:
            return [len(self.spike_times) / duration]

        # partition spike_counts into bins
        maxes = np.linspace(bin_size, duration, num=bin_size)

        bins = []

        index = 0
        for max_ in maxes:
            bin_ = []
            while index < len(self.spike_times):
                if spike := self.spike_times[index] < max_:
                    bin_.append(spike)
                index += 1

            bins.append(bin_)

        rates = []
        for bin_ in bins:
            rates.append(len(bin_) / bin_size * (1 / 1000))  # (adjusted for fires / sec)

        return rates

    def render(self, vals: np.linspace):
        """creates a pyplot visualization of the voltage values"""
        plt.figure(figsize=(6.4, 4.8))
        plt.title(
            f"Spike Count - {self.neuron_type} {self.n_id} - Lambda Odor: {self.lambda_odor}, Lambda Mech: {self.lambda_mech}")
        plt.plot(vals, self.spike_counts, color="red" if self.neuron_type == "LN" else "blue")
        plt.figure()
        plt.title(
            f"Voltage - {self.neuron_type} {self.n_id} - Lambda Odor: {self.lambda_odor}, Lambda Mech: {self.lambda_mech}")
        plt.plot(vals, self.voltages, color="red" if self.neuron_type == "LN" else "blue")
        plt.plot(vals, self.g_sk_vals, color='purple')
        plt.plot(vals, self.g_slow_vals, color='orange')
        plt.plot(self.g_inh_vals, color='green')
        plt.show()
        pass

    def update(self):
        if self.refractory_counter > 0:
            self.refractory_counter -= DELTA_T

        else:
            self.v = self.v + self.dv_dt * DELTA_T
            # poisson model
            lamb_ = self.lambda_tot()
            rate = lamb_ * DELTA_T
            self.lambda_vals.append(rate)

            if self.v >= Neuron.V_THRES:
                self.spike()

            # poisson modeling
            if random_choice(rate):
                self.stim_times.append(self.t)

            # update everything
            self.g_exc = self.g_gen(self.s_pn, Neuron.TAU_EXC, self.exc_times)
            self.g_inh = self.g_gen(self.s_inh, Neuron.TAU_INH, self.inh_times)
            self.g_slow = self.g_gen(self.s_slow, Neuron.TAU_SLOW, self.inh_times)
            self.g_stim = self.g_gen(self.s_stim, Neuron.TAU_STIM, self.stim_times)

            self.filter_exc_times()
            self.filter_inh_times()
            self.filter_stim_times()

            if self.neuron_type == "PN":
                self.g_sk = self.g_sk_func()
                self.dv_dt = (-1 * (self.v - self.V_L) / self.TAU_V) - \
                             (self.g_sk * (self.v - self.V_SK)) - \
                             (self.g_stim * (self.v - self.V_STIM)) - \
                             (self.g_exc * (self.v - self.V_EXC)) - \
                             (self.g_inh * (self.v - self.V_INH)) - \
                             (self.g_slow * (self.v - self.V_INH)) - \
                             (self.g_exc * (self.v - self.V_EXC))

            else:  # self.type == "LN"
                self.dv_dt = (-1 * (self.v - self.V_L) / self.TAU_V) - \
                             (self.g_stim * (self.v - self.V_STIM)) - \
                             (self.g_exc * (self.v - self.V_EXC)) - \
                             (self.g_inh * (self.v - self.V_INH)) - \
                             (self.g_slow * (self.v - self.V_INH)) - \
                             (self.g_exc * (self.v - self.V_EXC))

        self.t += DELTA_T
        self.voltages.append(self.v)
        self.g_inh_vals.append(self.g_inh)
        self.g_slow_vals.append(self.g_slow)
        self.g_sk_vals.append(self.g_sk)
        self.spike_counts.append(len(self.spike_times))


class Glomerulus:
    PN_PN_PROBABILITY = 0.50
    PN_LN_PROBABILITY = 0.50
    LN_PN_PROBABILITY = 0.40
    LN_LN_PROBABILITY = 0.25
    count = 0

    def __init__(self, stim_time: float, lambda_odor: float, lambda_mech: float, g_id: int):
        self.stim_times = stim_time
        self.lambda_odor_factor = lambda_odor
        self.lambda_mech_factor = lambda_mech
        self.pns: list[Neuron] = [Neuron(stim_time, lambda_odor, lambda_mech, "PN", neuron_id=i) for i in range(1, 11)]
        self.lns: list[Neuron] = [Neuron(stim_time, lambda_odor, lambda_mech, "LN", neuron_id=j) for j in range(11, 17)]
        self.neurons = copy(self.pns)
        self.neurons.extend(self.lns)
        self.g_id = g_id

        print(f"{self.pns}")
        print(f"NEURON COUNTS: {len(self.pns)}, {len(self.lns)}")

        # build synapse network
        for pn in self.pns:
            for target in self.pns:
                if target is pn:
                    continue
                else:
                    if random_choice(Glomerulus.PN_PN_PROBABILITY):
                        print(f"Glomerulus {self.g_id} - PN {pn.n_id} synapsing onto PN {target.n_id}")
                        pn.connected_neurons.append(target)
                        target.total_excitation += 1

            for target in self.lns:
                if random_choice(Glomerulus.PN_LN_PROBABILITY):
                    print(f"Glomerulus {self.g_id} - PN {pn.n_id} synapsing onto LN {target.n_id}")
                    pn.connected_neurons.append(target)
                    target.total_excitation += 1

        for ln in self.lns:
            for target in self.pns:
                if random_choice(Glomerulus.LN_PN_PROBABILITY):
                    print(f"Glomerulus {self.g_id} - LN {ln.n_id} synapsing onto PN {target.n_id}")
                    ln.connected_neurons.append(target)
                    target.total_inhibition += 1

            for target in self.lns:
                if ln is target:
                    continue
                else:
                    if random_choice(Glomerulus.LN_LN_PROBABILITY):
                        print(f"Glomerulus {self.g_id} - LN {ln.n_id} synapsing onto LN {target.n_id}")
                        ln.connected_neurons.append(target)
                        target.total_inhibition += 1

    def synapse_onto_other_glomerulus(self, glomerulus: "Glomerulus"):
        for ln in self.lns:
            for pn in glomerulus.pns:
                if random_choice(Glomerulus.LN_PN_PROBABILITY):
                    ln.connected_neurons.append(pn)
                    pn.total_inhibition += 1

    def get_neurons(self):
        return self.neurons

    def update(self):
        for neuron in self.neurons:
            neuron.update()

    def get_normalized_average_firing_rates(self, duration, bin_size):
        """Returns normalized average firing rates for given time intervals, with PNs being returned first."""
        pn_firing_rates = [pn.generate_firing_rates(duration, bin_size) for pn in self.pns]
        ln_firing_rates = [ln.generate_firing_rates(duration, bin_size) for ln in self.lns]

        avg_pn_firing_rates = []
        avg_ln_firing_rates = []

        for i in range(len(pn_firing_rates)):
            avg_pn_firing_rates.append(np.average([rates[i] for rates in pn_firing_rates]))
            avg_ln_firing_rates.append(np.average([rates[i] for rates in ln_firing_rates]))

        pn_bg = avg_pn_firing_rates[0]
        normalized_avg_pn_firing_rates = [rate / pn_bg for rate in avg_pn_firing_rates]

        ln_bg = avg_pn_firing_rates[0]
        normalized_avg_ln_firing_rates = [rate / ln_bg for rate in avg_ln_firing_rates]

        return normalized_avg_pn_firing_rates, normalized_avg_ln_firing_rates


class Network:
    def __init__(self, stim_time: float, network_type: str, affected_glomeruli: list[int]):
        self.stim_time = stim_time
        self.network_type = network_type
        self.affected_glomeruli = affected_glomeruli
        self.glomeruli = []

        match self.network_type:
            case "Odor":
                for i in range(6):
                    self.glomeruli.append(Glomerulus(self.stim_time,
                                                     Neuron.LAMBDA_ODOR_MAX * affected_glomeruli.count(i),
                                                     0, i))
            case "Mechanosensory":
                for i in range(6):
                    self.glomeruli.append(Glomerulus(self.stim_time, 0, Neuron.LAMBDA_MECH_MAX, i))
            case "Normalized":
                for i in range(6):
                    self.glomeruli.append(Glomerulus(self.stim_time,
                                                     0.5 * Neuron.LAMBDA_ODOR_MAX * affected_glomeruli.count(i),
                                                     0.5 * Neuron.LAMBDA_MECH_MAX, i))
            case "Additive":
                for i in range(6):
                    self.glomeruli.append(Glomerulus(self.stim_time,
                                                     Neuron.LAMBDA_ODOR_MAX * affected_glomeruli.count(i),
                                                     Neuron.LAMBDA_MECH_MAX, i))

        for glomerulus in self.glomeruli:
            for target in self.glomeruli:
                if glomerulus is target:
                    pass
                else:
                    glomerulus.synapse_onto_other_glomerulus(target)

    def update(self):
        for glomerulus in self.glomeruli:
            glomerulus.update()
