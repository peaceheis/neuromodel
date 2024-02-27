import math
import os
from copy import copy
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

DELTA_T = .1  # ms
SEED = 1982739482797832
NP_RANDOM_GENERATOR = np.random.default_rng(SEED)


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

    STIM_DURATION = 1000  # ms

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
            self.s_pn = 0.006
            self.s_inh = 0.015
            self.s_slow = 0.04
            self.s_stim = 0.0031
            self.g_stim = 0.0031

        else:  # self.type == "PN"
            self.odor_tau_rise = 35
            self.mech_tau_rise = 0
            self.s_pn = 0.01
            self.s_inh = 0.0169
            self.s_slow = 0.0338
            self.s_stim = 0.004
            self.g_stim = 0.004
            self.s_sk = NP_RANDOM_GENERATOR.normal(Neuron.SK_MU, Neuron.SK_STDEV)
            if self.s_sk < 0:
                self.s_sk = 0

        # conductance values
        self.g_sk = 0.0
        self.g_stim = 0.0
        self.g_slow = 0.0
        self.g_inh = 0.0
        self.g_exc = 0.0

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
            exp_decay = -(self.t - (stim_time + 2 * Neuron.TAU_HALF_RISE_SK)) / self.TAU_SK
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
        plt.plot(vals, self.g_sk_vals, color = 'purple')
        plt.plot(vals, self.g_slow_vals, color = 'orange')
        plt.plot(self.g_inh_vals, color = 'green')
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
                             (self.g_slow * (self.v - self.V_INH))

            else:  # self.type == "LN"
                self.dv_dt = (-1 * (self.v - self.V_L) / self.TAU_V) - \
                             (self.g_stim * (self.v - self.V_STIM)) - \
                             (self.g_exc * (self.v - self.V_EXC)) - \
                             (self.g_inh * (self.v - self.V_INH)) - \
                             (self.g_slow * (self.v - self.V_INH))

        self.t += DELTA_T
        self.voltages.append(self.v)
        self.g_inh_vals.append(self.g_inh)
        self.g_slow_vals.append(self.g_slow)
        self.g_sk_vals.append(self.g_sk)
        self.spike_counts.append(len(self.spike_times))


class Glomerulus:
    PN_PN_PROBABILITY = 0.75
    PN_LN_PROBABILITY = 0.75
    LN_PN_PROBABILITY = 0.38
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
                    target.total_excitation += 1

            for target in self.lns:
                if ln is target:
                    continue
                else:
                    if random_choice(Glomerulus.LN_LN_PROBABILITY):
                        print(f"Glomerulus {self.g_id} - LN {ln.n_id} synapsing onto LN {target.n_id}")
                        ln.connected_neurons.append(target)
                        target.total_excitation += 1

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


duration = 1000
stim_time = 500
bin_size = 50
steps = int(duration / DELTA_T)

network = Network(stim_time, "Additive", [0, 1, 2, 3, 4, 5])

vals = np.linspace(0, duration, steps)  # linspace for iteration of the network

for val in vals:
    network.update()
    print(val)

# for neuron in network.glomeruli[0].get_neurons():
#     neuron.render(vals)

should_serialize = True

if should_serialize:
    rn = datetime.now()
    prefix = f"/Users/scakolatse/coding-projects/neuromodel/output/{rn.year}{rn.month}{rn.day}-{rn.hour}.{rn.minute}.{rn.second}-{rn.microsecond}/"
    os.mkdir(prefix)

totaldata= []
for i, glomerulus in enumerate(network.glomeruli):
    data = []
    for neuron in glomerulus.get_neurons():
        data.append(neuron.spike_times)
        totaldata.append(neuron.spike_times)

    fig, axs = plt.subplots(2, 2, figsize=(14, 5))
    fig.suptitle(f"Glomerulus {i}")

    colors = ["blue" if neuron.neuron_type == "LN" else "red" for neuron in
              glomerulus.get_neurons()]

    axs[0, 0].eventplot(data, colors=colors)
    axs[0, 0].set_title(f"Eventplot")

    axs[0, 1].eventplot([[n.n_id for n in neuron.connected_neurons] for neuron in glomerulus.get_neurons()], colors=colors)
    axs[0, 1].set_title(f"Connectivity")

    plt.figure()
    plt.eventplot(totaldata, colors='blue')
    plt.show()

    axs[1, 0].bar([f"{i}" in range (1, 17)], [neuron.total_excitation for neuron in glomerulus.get_neurons()], width = 0.25, color=colors)
    axs[1, 0].set_xlabel("Excitation Amounts")

    axs[1, 1].bar([f"{i}" in range(1, 17)], [neuron.total_inhibition for neuron in glomerulus.get_neurons()], width = 0.25, color=colors)
    axs[1, 0].set_xlabel("Inhibition Amounts")

    if should_serialize:
        plt.savefig(prefix + f"{i}.png")
    else:
        plt.show()

    for neuron in glomerulus.get_neurons():
        print(f"Glomerulus {i} - Neuron Inhibition {neuron.total_inhibition}")
        print(f"Glomerulus {i} - Neuron Excitation {neuron.total_excitation}")

    # plt.figure(16 + i)
    # plt.title(f"Glomerulus {i} PN Rates")
    # plt.bar(rate_bins, pn_rates)
    #
    # plt.show()
    #
    #
    # plt.figure(32 + i)
    # plt.title(f"Glomerulus {i} Ln Rates")
    # plt.bar(rate_bins, ln_rates)
    # print("ok here")
    # plt.show()
    # print("yay!")

# for i in range(6):
# glomerulus = Glomerulus(1000, Neuron.LAMBDA_ODOR_MAX, Neuron.LAMBDA_MECH_MAX)
# neuron = Neuron(250, 0, Neuron.LAMBDA_MECH_MAX, "PN")
#
# steps = np.linspace(0, duration, num=int(duration / DELTA_T))
#
# for step in steps:
#     print(step)
#     glomerulus.update()
#
# neurons = glomerulus.pns
# neurons.extend(glomerulus.lns)
# plt.figure(figsize=(6.4, 4.8))
# plt.title(f"Spike Count - {neuron.neuron_type} - Lambda Odor: {neuron.lambda_odor}, Lambda Mech: {neuron.lambda_mech}")
# plt.plot(steps, neuron.spike_counts, color="red" if neuron.neuron_type == "LN" else "blue")
# plt.show()
# plt.figure()
# plt.title(f"Voltage - {neuron.neuron_type} - Lambda Odor: {neuron.lambda_odor}, Lambda Mech: {neuron.lambda_mech}")
# plt.plot(steps, neuron.voltages, color="red" if neuron.neuron_type == "LN" else "blue")
# plt.show()

'''
neuron2.connected_neurons.append(neuron)
neuron.connected_neurons.append(neuron2)

for i, neuron in enumerate(neurons):
    plt.figure(figsize=(8, 6))
    plt.plot(steps, neuron.spike_counts, color=([f'C{i}' for i in range(10)] + [f'C{i}' for i in range(6)])[i], linewidth=1)
    plt.show()
    data.append(neuron.spike_times)
plt.eventplot(data, colors=[f'C{i}' for i in range(10)] + [f'C{i}' for i in range(6)], lineoffsets=2, linewidths=1)
print(data)
plt.show()

for step in steps:
    neuron.update()
    neuron2.update()
    if int(neuron.t) == 1000:
        bg_count = len(neuron.spike_times)
        bg_count_2 = len(neuron2.spike_times)
    if int(neuron.t) == 1500:
        last_500 = len(neuron.spike_times)
        last_500_2 = len(neuron2.spike_times)

bg_rate = bg_count / 1000
bg_rate_2 = bg_count_2 / 1000
end_count = len(neuron.spike_times) - last_500
end_count_2 = len(neuron2.spike_times) - last_500_2
end_rate = end_count / 500
end_rate_2 = end_count_2 / 500

print(f"FIRING COUNTS - NEURON 1 - 1s bg {bg_count}, last 500ms {end_count}")
print(f"FIRING COUNTS - NEURON 2 - 1s bg {bg_count_2}, last 500ms {end_count_2}")
print(f"FIRING RATES - NEURON 1 - 1s bg {bg_rate}, last 500ms {end_rate}")
print(f"FIRING RATES - NEURON 2 - 1s bg {bg_rate_2}, last 500ms {end_rate_2}")
print(f"NORMALIZED RATE - NEURON 1 - {end_rate / bg_rate}")
print(f"NORMALIZED RATE - NEURON 2 - {end_rate_2 / bg_rate_2}")
'''
