# Hodgkin-Huxley Equation Modeling
import math

import numpy as np
import matplotlib.pyplot as plt

T_TOTAL = 2  # (sec)
DELTA_T = .005 * .001 # (sec)

C_M = 1  # General Membrane Conductance (F/cm^2)

# Equilibrium Potential Constants (mV)
E_NA = 55  # Sodium Channel Equilibrium
E_K = -90  # Potassium Channel Equilibrium
E_L = -65  # Leak Channel Equilibrium
V_REST = -70  # Rest Voltage

# Voltage Independent Conductance Constants (Ohm^-1cm^-2)
G_NA_MAX = 120  # Sodium
G_K_MAX = 36  # Potassium
G_L_MAX = 0.3  # Leak

# Initial Parameters (all placeholder for now)
NUM_STEPS = int(T_TOTAL / DELTA_T)


class Neuron:
    def __init__(self) -> None:
        self.i_external: float = 0  # external current from dendrites

        # membrane voltage
        self.v_m: float = V_REST

        # sodium activation variable
        self.m: float = self.alpha_m() / (self.alpha_m() + self.beta_m())

        # sodium inactivation variable
        self.h: float = self.alpha_h() / (self.alpha_h() + self.beta_h())

        # k activation variable
        self.n: float = self.alpha_n() / (self.alpha_n() + self.beta_n())


        self.is_spiking: bool = False
        self.spike_count: int = 0

        self.voltages: list = []
        self.m_vals: list = []
        self.h_vals: list = []
        self.n_vals: list = []

    def update(self, delta_t: float):
        """updates neuronal model in the following order:
            - caches invalidates the current m, h, and n values, putting them into the appropriate prev variable
            - caches and invalidates the current V value, putting it into the v_prev variable
            - updates current m, h, and n values
            - updates current V value"""
        self.m_vals.append(self.m)
        self.h_vals.append(self.h)
        self.n_vals.append(self.n)

        self.voltages.append(self.v_m)

        self.m += self.dm_dt(self.m) * delta_t
        self.h += self.h + self.dh_dt(self.h) * delta_t
        self.n += self.n + self.dn_dt(self.n) * delta_t

        self.v_m = self.v_m + self.dv_dt() * delta_t


    def dv_dt(self) -> float:
        """voltage rate of change"""
        return (self.i_external - self.i_na() - self.i_k() - self.i_leak()) / C_M

    # ion channel currents

    def i_na(self) -> float:
        """sodium channel current"""
        return self.g_na() * (self.v_m - E_NA)

    def i_k(self) -> float:
        """potassium channel current"""
        return self.g_k() * (self.v_m - E_K)

    def i_leak(self) -> float:
        """leak channel current"""
        return G_L_MAX * (self.v_m - E_L)

    # ion channel conductances

    def g_na(self) -> float:
        """sodium channel conductance"""
        return G_NA_MAX * (self.m ** 3) * self.h

    def g_k(self) -> float:
        """potassium channel conductance"""
        return G_K_MAX * (self.n ** 4)

    # gate variable rate functions

    def dm_dt(self, m: float) -> float:
        """sodium gate activation rate function"""
        return self.alpha_m() * self.v_m * (1 - m) - self.beta_m() * self.v_m * m

    def dh_dt(self, h: float) -> float:
        """sodium gate inactivation rate function"""
        return self.alpha_h() * self.v_m * (1 - h) - self.beta_h() * self.v_m * h

    def dn_dt(self, n: float) -> float:
        """potassium gate activation rate function"""
        return self.alpha_n() * self.v_m * (1 - n) - self.beta_h() * self.v_m * n


    # alpha (closed to open state) transition rate functions

    def alpha_m(self) -> float:
        """sodium activation gate opening transition rate function"""
        return .1*((25-self.v_m) / (np.exp((25-self.v_m)/10)-1))

    def alpha_h(self) -> float:
        """sodium inactivation gate opening transition rate function"""
        return .07*np.exp(-self.v_m/20)

    def alpha_n(self) -> float:
        """potassium activation gate opening transition rate function"""
        return 0.01 * ((10-self.v_m) / (np.exp((10-self.v_m)/10)-1))

    # beta (open to closed state) transition rate functions

    def beta_m(self) -> float:
        """sodium activation gate closing transition rate function"""
        return 4*np.exp(-self.v_m/18)

    def beta_h(self) -> float:
        """sodium inactivation gate closing transition rate function"""
        return 1/(np.exp((30-self.v_m)/10)+1)

    def beta_n(self) -> float:
        """potassium activation gate closing transition rate function"""
        return .125*np.exp(-self.v_m/80)


neuron = Neuron()
x = np.linspace(0, T_TOTAL, NUM_STEPS)

for _ in range(NUM_STEPS):
    neuron.update(DELTA_T)

plt.plot(x, neuron.voltages)
plt.title("V_m")
plt.show()

plt.plot(x, neuron.m_vals)
plt.title("m")
plt.show()

plt.plot(x, neuron.h_vals)
plt.title("h")
plt.show()

plt.plot(x, neuron.n_vals)
plt.title("n")
plt.show()

print(neuron.voltages)