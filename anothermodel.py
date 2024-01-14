import numpy as np
import matplotlib.pyplot as plt

# Hodgkin-Huxley model parameters
C_m = 1.0  # membrane capacitance (uF/cm^2)
g_Na = 120.0  # sodium conductance (mS/cm^2)
g_K = 36.0  # potassium conductance (mS/cm^2)
g_L = 0.3  # leak conductance (mS/cm^2)
E_Na = 50.0  # sodium reversal potential (mV)
E_K = -90.0  # potassium reversal potential (mV)
E_L = -65  # leak reversal potential (mV)

# Time parameters
dt = 0.01  # time step (ms)
t_max = 200.0  # total simulation time (ms)
num_steps = int(t_max / dt)

# Initial conditions
V = -70.0  # initial membrane potential (mV)
m = .05   # initial sodium activation variable
h = .6   # initial sodium inactivation variable
n = .32   # initial potassium activation variable

# External current input (can be modified based on your simulation)
I_ext = 20

# Arrays to store results
time = np.arange(0, t_max, dt)
voltage = np.zeros(num_steps)
m_values = np.zeros(num_steps)
h_values = np.zeros(num_steps)
n_values = np.zeros(num_steps)

# Euler's method to solve the Hodgkin-Huxley equations
for i in range(num_steps):
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
    V += dt * dVdt

    # Gating variable dynamics
    dmdt = alpha_m * (1 - m) - beta_m * m
    dhdt = alpha_h * (1 - h) - beta_h * h
    dndt = alpha_n * (1 - n) - beta_n * n
    m += dt * dmdt
    h += dt * dhdt
    n += dt * dndt
    if I_ext > 0:
        I_ext = I_ext - (.15*dt)
    else:
        I_ext = 0
    print(I_ext)
    # Store values
    voltage[i] = V
    m_values[i] = m
    h_values[i] = h
    n_values[i] = n

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time, voltage, label='Membrane Potential (V)')
plt.title('Hodgkin-Huxley Model using Euler\'s Method')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.legend()
plt.grid(True)
plt.figure(2)
plt.plot(time,m_values)
plt.plot(time, n_values)
plt.plot(time, h_values)
plt.show()