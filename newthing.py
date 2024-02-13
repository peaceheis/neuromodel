
from enum import Enum, EnumType

import numpy as np
import matplotlib.pyplot as plt

DELTA_T = .1  # ms
NP_RANDOM_GENERATOR = np.random.default_rng(985047891247389)


def random_choice(given_prob: float):
    rand_val = NP_RANDOM_GENERATOR.uniform(0, 1)
    if rand_val < given_prob:
        return True
    else:
        return False


class Neuron:
    V_L = 0.0
    V_EXC = 14 / 3
    V_STIM = 14 / 3
    V_SK = - 2/3
    V_INH = -2 / 3
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

    STIM_DURATION = 200  # ms

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
        self.refractory_counter = 0.0
        self.neuron_id = neuron_id
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

    def render(self, vals: np.linspace):
        """creates a pyplot visualization of the voltage values"""
        plt.figure(figsize=(6.4, 4.8))
        plt.title(f"Spike Count - {self.neuron_type} {self.neuron_id} - Lambda Odor: {self.lambda_odor}, Lambda Mech: {self.lambda_mech}")
        plt.plot(vals, self.spike_counts, color="red" if self.neuron_type == "LN" else "blue")
        plt.figure()
        plt.title(f"Voltage - {self.neuron_type} {self.neuron_id} - Lambda Odor: {self.lambda_odor}, Lambda Mech: {self.lambda_mech}")
        plt.plot(vals, self.voltages, color="red" if self.neuron_type == "LN" else "blue")
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
        self.spike_counts.append(len(self.spike_times))


class Glomerulus:
    PN_PN_PROBABILITY = 0.75
    PN_LN_PROBABILITY = 0.75
    LN_PN_PROBABILITY = 0.38
    LN_LN_PROBABILITY = 0.28
    count = 0

    def __init__(self, stim_time: float, lambda_odor: float, lambda_mech: float):
        self.stim_times = stim_time
        self.lambda_odor_factor = lambda_odor
        self.lambda_mech_factor = lambda_mech
        self.pns: list[Neuron] = [Neuron(stim_time, lambda_odor, lambda_mech, "PN", neuron_id=i) for i in range(1, 11)]
        self.lns: list[Neuron] = [Neuron(stim_time, lambda_odor, lambda_mech, "LN", neuron_id=j) for j in range(11, 17)]

        print(f"{len(self.pns)}, {len(self.lns)}")

        # build synapse network
        for pn in self.pns:
            for target in self.pns:
                if target is pn:
                    continue
                else:
                    if random_choice(Glomerulus.PN_PN_PROBABILITY):
                        pn.connected_neurons.append(target)

            for target in self.lns:
                if random_choice(Glomerulus.PN_LN_PROBABILITY):
                    pn.connected_neurons.append(target)

        for ln in self.lns:
            for target in self.pns:
                if random_choice(Glomerulus.LN_PN_PROBABILITY):
                    ln.connected_neurons.append(target)
            for target in self.lns:
                if ln is target:
                    continue
                else:
                    if random_choice(Glomerulus.LN_LN_PROBABILITY):
                        ln.connected_neurons.append(target)

    def synapse_onto_other_glomerulus(self, glomerulus: "Glomerulus"):
        for ln in self.lns:
            for pn in glomerulus.pns:
                if random_choice(Glomerulus.LN_PN_PROBABILITY):
                    ln.connected_neurons.append(pn)

    def get_neurons(self):
        neurons = self.pns
        neurons.extend(self.lns)
        return neurons

    def update(self):
        for pn in self.pns:
            pn.update()
        for ln in self.lns:
            ln.update()


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
                                                     0))
            case "Mechanosensory":
                for _ in range(6):
                    self.glomeruli.append(Glomerulus(self.stim_time, 0, Neuron.LAMBDA_MECH_MAX))
            case "Normalized":
                for i in range(6):
                    self.glomeruli.append(Glomerulus(self.stim_time,
                                                     0.5 * Neuron.LAMBDA_ODOR_MAX * affected_glomeruli.count(i),
                                                     0.5 * Neuron.LAMBDA_MECH_MAX))
            case "Additive":
                for i in range(6):
                    self.glomeruli.append(Glomerulus(self.stim_time,
                                                     Neuron.LAMBDA_ODOR_MAX * affected_glomeruli.count(i),
                                                     Neuron.LAMBDA_MECH_MAX))

        for glomerulus in self.glomeruli:
            for target in self.glomeruli:
                if glomerulus is target:
                    pass
                else:
                    glomerulus.synapse_onto_other_glomerulus(target)

    def update(self):
        for glomerulus in self.glomeruli:
            glomerulus.update()

duration = 250
stim_time = 50
steps = int(duration / DELTA_T)

network = Network(stim_time, "Additive", [0, 1, 2, 3, 4, 5])

vals = np.linspace(0, duration, steps) #linspace for iteration of the network

for val in vals:
    network.update()
    print(val)

# for neuron in network.glomeruli[0].get_neurons():
#     neuron.render(vals)

for i, glomerulus in enumerate(network.glomeruli):
    data = []
    for neuron in glomerulus.get_neurons():
        data.append(neuron.spike_times)

    plt.figure(figsize=(10, 7.5))
    plt.title(f"Glomerulus {i} Eventplot")
    plt.eventplot(data, colors="red")
    plt.show()





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