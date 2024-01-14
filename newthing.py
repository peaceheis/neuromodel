from enum import Enum

import numpy as np

V_PN = 0
V_LN = 0

S_SK = np.random.normal(.5,
                        .2)  # strength of the SK current, randomly determined by sampling from a normal distribution

DELTA_T = .01


class Neuron:
    NeuronType = Enum("NeuronType", ["LN", "PN"])

    V_L = 0
    V_EXC = 14 / 3
    V_STIM = 14 / 3
    V_SK = -2 / 3
    V_INH = -2 / 3
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

    # input source rates
    LAMBDA_BG = 3.6  # spikes / ms
    LAMBDA_ODOR_MAX = 3.6  # spikes / ms
    LAMBDA_MECH_MAX = 1.8  # spikes / ms

    STIM_DURATION = 1000  # ms

    NP_RANDOM_GENERATOR = np.random.default_rng()
    SK_MU = 0.5
    SK_STDEV = 0.2

    def __init__(self, spike_times: list, neuron_type: NeuronType):
        self.spike_times = spike_times
        self.type = neuron_type
        self.t = 0
        self.v = 0
        self.dv = 0
        self.lambda_mech = 0.0
        self.lambda_odor = 0.0
        # 4.2.1 The neuron model
        # reversal potentials (nondimensional)

        if self.type == "LN":
            self.odor_tau_rise = 0
            self.mech_tau_rise = 300
            self.s_pn = 0.006
            self.s_inh = 0.015
            self.s_slow = 0.04
            self.s_stim = .0031
            self.g_stim = 0.0031

        elif self.type == "PN":
            self.odor_tau_rise = 35
            self.mech_tau_rise = 0
            self.s_pn = 0.01
            self.s_inh = 0.0169
            self.s_slow = 0.0338
            self.s_stim = 0.004
            self.g_stim = 0.004
            self.s_sk = Neuron.NP_RANDOM_GENERATOR.normal(Neuron.SK_MU, Neuron.SK_STDEV)
            if self.sk < 0:
                self.sk = 0

        # conductance values
        self.g_sk = self.g_sk(S_SK)
        self.g_stim: float = 0.0
        self.g_slow: float = 0.0
        self.g_inh: float = 0.0
        self.g_exc: float = 0.0

    @staticmethod
    def Heaviside(num: float | int) -> int:  # gives the h value for alpha_exc function
        if num >= 0:
            return 1
        if num < 0:
            return 0

    def alpha(self, tau: float, spike_time: float | int) -> float:
        heaviside_term = (self.Heaviside(self.t - self.t - spike_time)) / tau
        exp_decay_term = np.exp((self.t - spike_time) / tau)
        return heaviside_term * exp_decay_term

    def beta(self, s) -> float:
        if self.t <= (s + 2 * Neuron.TAU_HALF_RISE_SK):
            heaviside_term = self.Heaviside(self.t - s) / self.TAU_SK
            sigmoid_term_num = np.exp((5 * ((self.t - s) - Neuron.TAU_HALF_RISE_SK)) / Neuron.TAU_HALF_RISE_SK)
            sigmoid_term_den = 1 + sigmoid_term_num
            return heaviside_term * sigmoid_term_num / sigmoid_term_den
        else:
            exp_decay = -(self.t - (s + 2 * Neuron.TAU_HALF_RISE_SK)) / self.TAU_SK
            return (1 / self.TAU_SK) * np.exp(exp_decay)

    def g_gen(self, s_val, tau_val) -> float:
        return np.sum([s_val * self.alpha(tau_val, s) for s in self.spike_times])

    def g_sk(self) -> float:
        return np.sum([self.s_sk * self.beta(s) for s in self.spike_times])

    def update_dv(self):
        if self.type == "PN":
            self.dv = (-1 * (self.v - self.V_L) / self.TAU_V) - (self.g_sk * (self.v - self.V_SK)) - (
                    self.g_stim * (self.v - self.V_STIM)) - (
                              self.g_exc * (self.v - self.V_EXC)) - (
                              self.g_inh * (self.v - self.V_INH)) - (
                              self.g_slow * (self.v - self.V_INH))
        else:  # self.type == "LN"
            self.dv = (-1 * (self.v - self.V_L) / self.TAU_V) - (self.g_sk * (self.v - self.V_SK)) - (
                    self.g_stim * (self.v - self.V_STIM)) - (
                              self.g_exc * (self.v - self.V_EXC)) - (
                              self.g_inh * (self.v - self.V_INH)) - (
                              self.g_slow * (self.v - self.V_INH))

    def odor_j_dyn(self, j: int, t: float, t_on: float | int) -> float:
        t_off = t_on + Neuron.STIM_DURATION
        if self.type == "PN":
            if t <= t_on + 2 * self.odor_tau_rise:
                heaviside_term = self.Heaviside(t - t_on)
                sigmoid_term_num = np.exp((5 * ((t - t_on) - self.odor_tau_rise)) / self.odor_tau_rise)
                sigmoid_term_den = 1 + sigmoid_term_num
                return heaviside_term * sigmoid_term_num / sigmoid_term_den
            elif t_on + 2 * self.odor_tau_rise < t <= t_off:
                return 1
            else:
                return np.exp(-(t - t_off) / Neuron.TAU_DECAY)
        else:
            if t <= t_off:
                return self.Heaviside(t - t_on)
            else:
                return np.exp(-(t - t_off) / Neuron.TAU_DECAY)

    def mech_j_dyn(self, j: int, t: float, t_on: float | int) -> float:
        t_off = t_on + Neuron.STIM_DURATION

        if self.type == "PN":
            if t <= t_off:
                return self.Heaviside(t - t_on)
            else:
                return np.exp(-(t - t_off) / Neuron.TAU_DECAY)
        else:
            if t <= t_on + 2 * self.mech_tau_rise:
                heaviside_term = self.Heaviside(t - t_on)
                sigmoid_term_num = np.exp((5 * ((t - t_on) - self.odor_tau_rise)) / self.odor_tau_rise)
                sigmoid_term_den = 1 + sigmoid_term_num
                return heaviside_term * sigmoid_term_num / sigmoid_term_den
            elif t_on + 2 * self.mech_tau_rise < t <= t_off:
                return 1
            else:
                return np.exp(-(t - t_off) / Neuron.TAU_DECAY)

    def lambda_j_tot(self, j: int, t: float, t_on: float) -> float:
        return (Neuron.LAMBDA_BG + Neuron.LAMBDA_ODOR_MAX * self.odor_j_dyn(j, t, t_on) + Neuron.LAMBDA_MECH_MAX
                * self.mech_j_dyn(j, t, t_on))




class Glomerulus:
    PN_PN_PROBABILITY = 0.75
    PN_LN_PROBABILITY = 0.75
    LN_PN_PROBABILITY = 0.38
    LN_LN_PROBABILITY = 0.28
    def __init__(self):
        self.neurons: list[Neuron] = []

    def propagate_lambda_odor(self, lambda_odor: float):
        for neuron in self.neurons:
            neuron.lambda_odor = lambda_odor

    def propagate_lambda_mech(self, lambda_mech: float):
        for neuron in self.neurons:
            neuron.lambda_mech = lambda_mech


class Network:
    def __init__(self):
        self.glomeruli: list[Glomerulus] = []

    def input_odor(self, affected_glomeruli: set[int], affected_glomeruli_2=None):
        if affected_glomeruli_2 is not None:
            self.input_two_odors(affected_glomeruli, affected_glomeruli_2, 0)
        else:
            for i, glomerulus in enumerate(self.glomeruli):
                glomerulus.propagate_lambda_odor(Neuron.LAMBDA_ODOR_MAX if i in affected_glomeruli else 0)

    def input_mech(self):
        for glomerulus in self.glomeruli:
            glomerulus.propagate_lambda_mech(Neuron.LAMBDA_MECH_MAX)
            glomerulus.propagate_lambda_odor(0)

    def additive_input(self, affected_glomeruli: set[int], affected_glomeruli_2=None):
        if affected_glomeruli_2 is not None:
            self.input_two_odors(affected_glomeruli, affected_glomeruli_2, Neuron.LAMBDA_MECH_MAX)
        else:
            for i, glomerulus in enumerate(self.glomeruli):
                glomerulus.propagate_lambda_odor(Neuron.LAMBDA_ODOR_MAX if i in affected_glomeruli else 0)
                glomerulus.propagate_lambda_mech(Neuron.LAMBDA_MECH_MAX)

    def normalized_input(self, affected_glomeruli: set[int], affected_glomeruli_2=None):
        if affected_glomeruli_2 is not None:
            self.input_two_odors(affected_glomeruli, affected_glomeruli_2, 0.5 * Neuron.LAMBDA_MECH_MAX)
        else:
            for i, glomerulus in enumerate(self.glomeruli):
                glomerulus.propagate_lambda_odor(0.5 * Neuron.LAMBDA_ODOR_MAX if i in affected_glomeruli else 0)
                glomerulus.propagate_lambda_mech(0.5 * Neuron.LAMBDA_MECH_MAX)

    def input_two_odors(self, affected_glomeruli1: set[int], affected_glomeruli_2: set[int], lambda_mech):
        for i, glomerulus in enumerate(self.glomeruli):
            affected1 = i in affected_glomeruli1
            affected2 = i in affected_glomeruli_2
            if affected1 and affected2:
                glomerulus.propagate_lambda_odor(2 * Neuron.LAMBDA_ODOR_MAX)
            elif affected1:
                glomerulus.propagate_lambda_odor(Neuron.LAMBDA_ODOR_MAX)
            elif affected2:
                glomerulus.propagate_lambda_odor(Neuron.LAMBDA_ODOR_MAX)
            else:
                glomerulus.propagate_lambda_odor(0)
            glomerulus.propagate_lambda_mech(lambda_mech)


def encounter_threshold():
    pass


def record_spike():
    pass
