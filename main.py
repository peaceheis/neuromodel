# Hodgkin-Huxley Equation Modeling
import math

import numpy as np
import matplotlib.pyplot as plt

T_TOTAL = 10  # (sec)
DELTA_T = .001  # (sec)

C_M = 1 * (10 ** -6)  # General Membrane Conductance (F/cm^2)

TEMP = 18  # temperature for the psi expression?

# Equilibrium Potential Constants (mV)
E_NA = 55  # Sodium Channel Equilibrium
E_K = -90  # Potassium Channel Equilibrium
E_L = -65  # Leak Channel Equilibrium
V_REST = -70  # Rest Voltage

# Voltage Independent Conductance Constants (Ohm^-1cm^-2)
G_NA_MAX = 120  # Sodium
G_K_MAX = 36  # Potassium
G_L_MAX = .3  # Leak

# Initial Parameters (all placeholder for now)
V_m = V_REST
NUM_STEPS = int(T_TOTAL / DELTA_T)


class Neuron:
    def __init__(self, temp: float) -> None:
        self.phi = 3 ** ((temp - 6.3) / 10)

        self.i_external: float = 0  # external current from dendrites

        # sodium activation variable
        self.m_prev: float = 0
        self.m_current: float = 0.2

        # sodium inactivation variable
        self.h_prev: float = 0
        self.h_current: float = 0.5

        # k activation variable
        self.n_prev: float = 0
        self.n_current: float = 1

        # membrane voltage
        self.v_prev: float = V_REST
        self.v_current: float = V_REST
        self.membrane_potential: float = self.v_current - V_REST

        self.is_spiking: bool = False
        self.spike_count: int = 0

        self.voltages: list = []

    def update(self, delta_t: float):
        self.voltages.append(self.v_current)

        self.v_prev = self.v_current
        self.m_prev = self.m_current
        self.h_prev = self.h_current
        self.n_prev = self.n_current

        self.v_current = self.v_prev + self.dv_dt()*delta_t
        self.m_current = self.m_prev + self.dm_dt(self.m_prev)*delta_t
        self.h_current = self.h_prev + self.dh_dt(self.h_prev)*delta_t
        self.n_current = self.n_prev + self.dn_dt(self.n_prev)*delta_t


    def dv_dt(self) -> float:
        """voltage rate of change"""
        return (self.i_external - self.i_na() - self.i_k() - self.i_leak()) / C_M

    # ion channel currents

    def i_na(self) -> float:
        """sodium channel current"""
        return self.g_na() * (self.v_current - E_NA)

    def i_k(self) -> float:
        """potassium channel current"""
        return self.g_k() * (self.v_current - E_K)

    def i_leak(self) -> float:
        """leak channel current"""
        return G_L_MAX * (self.v_current - E_L)

    # ion channel conductances

    def g_na(self) -> float:
        """sodium channel conductance"""
        return G_NA_MAX * self.m_current ** 3 * self.h_current

    def g_k(self) -> float:
        """potassium channel conductance"""
        return G_K_MAX * self.n_current ** 4

    # gate variable rate functions

    def dm_dt(self, m: float) -> float:
        """sodium gate activation rate function"""
        return 1 / (self.tau_m() * (self.m_infinity() - m))

    def dh_dt(self, h: float) -> float:
        """sodium gate inactivation rate function"""
        return 1 / (self.tau_h() * (self.h_infinity() - h))

    def dn_dt(self, n: float) -> float:
        """potassium gate activation rate function"""
        return 1 / (self.tau_n() * (self.n_infinity() - n))

    # gate variable steady state functions

    def m_infinity(self) -> float:
        """sodium activation gate steady state function"""
        return self.alpha_m() / (self.alpha_m() + self.beta_m())

    def h_infinity(self) -> float:
        """sodium inactivation gate steady state function"""
        return self.alpha_h() / (self.alpha_h() + self.beta_h())

    def n_infinity(self) -> float:
        """potassium activation gate steady state function"""
        return self.alpha_n() / (self.alpha_n() + self.beta_n())

    # time constant functions

    def tau_m(self) -> float:
        """sodium activation time constant function"""
        return 1 / (self.alpha_m() + self.beta_m())

    def tau_h(self) -> float:
        """sodium inactivation time constant function"""
        return 1 / (self.alpha_h() + self.beta_h())

    def tau_n(self) -> float:
        """potassium activation time constant function"""
        return 1 / (self.alpha_n() + self.beta_n())

    # alpha (closed to open state) transition rate functions

    def alpha_m(self) -> float:
        """sodium activation gate opening transition rate function"""
        return self.phi * (2.5 - 0.1 * self.membrane_potential) / (math.exp(2.5 - 0.1 * self.membrane_potential) + 1)

    def alpha_h(self) -> float:
        """sodium inactivation gate opening transition rate function"""
        return 0.07 * self.phi * math.exp(-self.membrane_potential / 20)

    def alpha_n(self) -> float:
        """potassium activation gate opening transition rate function"""
        return self.phi * (0.1 - 0.01 * self.membrane_potential) / (math.exp(1 - 0.1 * self.membrane_potential) - 1)

    # beta (open to closed state) transition rate functions

    def beta_m(self) -> float:
        """sodium activation gate closing transition rate function"""
        return 4 * self.phi * math.exp(-self.membrane_potential / 20)

    def beta_h(self) -> float:
        """sodium inactivation gate closing transition rate function"""
        return self.phi / (math.exp(3.0 - 0.1 * self.membrane_potential) + 1)

    def beta_n(self) -> float:
        """potassium activation gate closing transition rate function"""
        return 0.125 * self.phi * math.exp(-self.membrane_potential / 80)


neuron = Neuron(9.0)
x = np.linspace(0, 10, NUM_STEPS)
Voltage = []

for _ in range(NUM_STEPS):
    neuron.update(DELTA_T)

plt.plot(x, neuron.voltages)
plt.show()

print(neuron.voltages)