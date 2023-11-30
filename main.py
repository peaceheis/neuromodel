# Hodgkin-Huxley Equation Modeling
import math

import numpy as np
import matplotlib.pyplot as plt

T_TOTAL = 1  # (sec)
DELTA_T = .001  # (sec)

C_M = 1 * (10 ** -6)  # General Membrane Conductance (F/cm^2)

TEMP = 18  # temperature for the psi expression?

# Equilibrium Potential Constants (mV)
E_NA = 55  # Sodium Channel Equilibrium
E_K = -77  # Potassium Channel Equilibrium
E_L = -65  # Leak Channel Equilibrium
V_REST = -70  # Rest Voltage

# Voltage Independent Conductance Constants (Ohm^-1cm^-2)
G_NA_MAX = 120  # Sodium
G_K_MAX = 36  # Potassium
G_L_MAX = .3  # Leak

# Initial Parameters (all placeholder for now)
V_m = V_REST
num_iterations = T_TOTAL / DELTA_T


class Neuron:
    def __init__(self, temp: float) -> None:
        self.phi = 3 ** ((temp - 6.3) / 10)

        self.i_external: float = 0  # external current from dendrites

        # sodium activation variable
        self.m_prev: float = 0
        self.m_current: float = 0

        # sodium inactivation variable
        self.h_prev: float = 0
        self.h_current: float = 0

        # k activation variable
        self.n_prev: float = 0
        self.n_current: float = 0

        # membrane voltage
        self.v_prev: float = V_REST
        self.v_current: float = V_REST
        self.membrane_potential: float = self.v_current - V_REST

        self.is_spiking: bool = False
        self.spike_count: int = 0

        self.voltages: list = []

    def update(self, delta_t: float):
        self.voltages.append(self.v_current)

        alpha_m = self.alpha_m()
        alpha_h = self.alpha_h()
        alpha_n = self.alpha_n()

        beta_m = self.beta_m()
        beta_h = self.beta_h()
        beta_n = self.beta_n()
        #dmdt = (alpha_m*(1-m)) - (beta_m*m)
        #dndt = (alpha_n*(1-n)) - (beta_n*n)
        #dhdt = (alpha_h*(1-h)) - (beta_h*h)

        #alpha_m = (.02*(V_m-25)/(1-np.e**(-1*(V_m-25)/9)))
        #alpha_n=.182*(V_m+35)/(1-np.e**(-1*(V_m-35)/9))
        #alpha_h = .25*np.e**(-1*(V_m+90)/12)
        
        #beta_m = (-.0002*(V_m-25)/(1-np.e**(-1*(V_m-25)/9)))
        #beta_n = -.124*(V_m+35)/(1-np.e**(-1*(V_m-35)/9))
        #beta_h = .25*np.e**((V_m+62)/6)/np.e**((V_m+90)/12)




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


# Functions
def current(g: int, V: int, E: int) -> float:
    return g * (V - E)


# Dependent Variables
g_k = G_K_MAX * k_activation ** 4
g_na = G_NA_MAX * na_inactivation * na_activation ** 3
delta_V = V_m - V_REST
I_k = current(g_k, V_m, E_K)
I_Na = current(g_na, V_m, E_NA)
I_L = current(G_L_MAX, V_m, E_L)
psi = 3 ** ((TEMP - 6.3) / 10)
alpha_m = (psi * (2.5 - (.1 * delta_V))) / (-1 + np.e ** (2.5 - (.1 * delta_V)))
alpha_n = (psi * (.1 - (.01 * delta_V))) / (-1 + np.e ** (1 - (.1 * delta_V)))
alpha_h = 0.07 * psi * (np.e ** (-delta_V / 20))
beta_m = 4 * psi * (np.e ** (-delta_V / 20))
beta_n = .125 * psi * (np.e ** (-delta_V / 80))
beta_h = psi / (1 + (np.e ** (3 - (.1 * delta_V))))
m_infinity = alpha_m / (alpha_m + beta_m)
n_infinity = alpha_n / (alpha_n + beta_n)
h_infinity = alpha_h / (alpha_h + beta_h)

tao_m = 1 / (alpha_m + beta_m)
tao_n = 1 / (alpha_m + beta_m)
tao_h = 1 / (alpha_h + beta_h)

dmdt = (1 / tao_m) * (m_infinity - na_activation)
dndt = (1 / tao_n) * (n_infinity - k_activation)
dhdt = (1 / tao_h) * (h_infinity - na_inactivation)
dvdt = -(I_k + I_L + I_Na)
x = np.linspace(0, 10, 10000)
Voltage = []
n_vals = []
for i in range(10000):
    V_m = V_m + (DELTA_T * dvdt)
    na_activation = na_activation + (DELTA_T * dmdt)
    k_activation = k_activation + (DELTA_T * dndt)
    na_inactivation = na_inactivation + (DELTA_T + dhdt)
    I_k = current(g_k, V_m, E_K)
    I_Na = current(g_na, V_m, E_NA)
    I_L = current(G_L_MAX, V_m, E_L)
    g_k = G_K_MAX * (k_activation ** 4)
    g_na = G_NA_MAX * na_inactivation * (na_activation ** 3)
    tao_m = 1 / (alpha_m + beta_m)
    tao_n = 1 / (alpha_m + beta_m)
    tao_h = 1 / (alpha_h + beta_h)
    m_infinity = alpha_m / (alpha_m + beta_m)
    n_infinity = alpha_n / (alpha_n + beta_n)
    h_infinity = alpha_h / (alpha_h + beta_h)
    dmdt = (1 / tao_m) * (m_infinity - na_activation)
    dndt = (1 / tao_n) * (n_infinity - k_activation)
    dhdt = (1 / tao_h) * (h_infinity - na_inactivation)
    dvdt = -(I_k + I_L + I_Na)
    alpha_m = (psi * (2.5 - (.1 * delta_V))) / (-1 + np.e ** (2.5 - (.1 * delta_V)))
    alpha_n = (psi * (.1 - (.01 * delta_V))) / (-1 + np.e ** (1 - (.1 * delta_V)))
    alpha_h = 0.07 * psi * (np.e ** (-delta_V / 20))
    beta_m = 4 * psi * (np.e ** (-delta_V / 20))
    beta_n = .125 * psi * (np.e ** (-delta_V / 80))
    beta_h = psi / (1 + (np.e ** (3 - (.1 * delta_V))))
    Voltage.append(V_m)
    n_vals.append(k_activation)
plt.plot(x, Voltage)
plt.show()
