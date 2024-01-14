import numpy as np
import matplotlib.pyplot as plt

# Hodgkin-Huxley model parameters
C_m = 1.0  # membrane capacitance (uF/cm^2)
g_Na = 120.0  # sodium conductance (mS/cm^2)
g_K = 36.0  # potassium conductance (mS/cm^2)
g_L = 0.3  # leak conductance (mS/cm^2)
E_Na = 50.0  # sodium reversal potential (mV)
E_K = -77.0  # potassium reversal potential (mV)
E_L = -54.4  # leak reversal potential (mV)

# Time parameters
dt = 0.01  # time step (ms)
t_max = 50.0  # total simulation time (ms)
num_steps = int(t_max / dt)

# Adjusted initial conditions for resting potential at -70 mV
V0 = -89.0  # initial membrane potential (mV)
m0 = 0.05   # initial sodium activation variable
h0 = 0.6    # initial sodium inactivation variable
n0 = 0.32   # initial potassium activation variable

# Threshold for action potential initiation
threshold = -55.0  # mV

# External current input (can be modified based on your simulation)
I_ext = 10.0

# Arrays to store results
time = np.arange(0, t_max, dt)
voltage = np.zeros(num_steps)
m_values = np.zeros(num_steps)
h_values = np.zeros(num_steps)
n_values = np.zeros(num_steps)

# Euler's method to solve the Hodgkin-Huxley equations
def hodgkin_huxley(V, m, h, n, I_ext):
    # Ion channel gating kinetics
    alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    beta_n = 0.125 * np.exp(-(V + 65) / 80)
    alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta_m = 4.0 * np.exp(-(V + 65) / 18)
    alpha_h = 0.07 * np.exp(-(V + 65) / 20)
    beta_h = 1.0 / (1 + np.exp(-(V + 35) / 10))

    # Membrane currents
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)

    # Membrane potential dynamics
    dVdt = (I_ext - I_Na - I_K - I_L) / C_m
    # Gating variable dynamics
    dmdt = alpha_m * (1 - m) - beta_m * m
    dhdt = alpha_h * (1 - h) - beta_h * h
    dndt = alpha_n * (1 - n) - beta_n * n

    return dVdt, dmdt, dhdt, dndt

# Initialize state variables with adjusted starting voltage
V = V0
m = m0
h = h0
n = n0

# Simulation loop
for i in range(num_steps):
    dVdt, dmdt, dhdt, dndt = hodgkin_huxley(V, m, h, n, I_ext)

    # Update state variables using Euler's method
    V += dt * dVdt
    m += dt * dmdt
    h += dt * dhdt
    n += dt * dndt

    # Store values
    voltage[i] = V
    m_values[i] = m
    h_values[i] = h
    n_values[i] = n

    # Check for action potential initiation (crossing the threshold)
    if V > threshold:
        print(f"Action potential initiated at t = {i * dt} ms")
        break

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time, voltage, label='Membrane Potential (V)')
plt.title('Hodgkin-Huxley Model with Resting Potential at -70 mV and Threshold at -55 mV')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.legend()
plt.grid(True)
plt.show()