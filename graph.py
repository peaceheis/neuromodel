import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy
import numpy as np

prefix = json.load(open("config.json"))["prefix"]
data = json.load(open(f"{prefix}result.json"))

output_dir = data["dir"]

stim_time = data["stim_time"]
duration = data["duration"]
delta_t = data["delta_t"]

@dataclass
class NeuronRep:
    voltages: np.array
    excitation_vals: np.array
    slow_excitation_vals: np.array
    inhibition_vals: np.array
    slow_inhibition_vals: np.array
    g_sk_vals: np.array
    stim_vals: np.array
    dv_dt_vals: np.array
    spike_times: np.array
    type: str

class GlomerulusRep:
    def __init__(self, pns: list[NeuronRep], lns: list[NeuronRep]):
        self.pns = pns
        self.lns = lns
        self.neurons = self.pns.copy()
        self.neurons.extend(self.lns)

    def get_neurons(self):
        try:
            return self.neurons
        except:
            self.neurons = self.pns.copy()
            self.neurons.extend(self.lns)




glomeruli = []
all_neurons = []
current_pns = []
current_lns = []

for i, neuron in enumerate(data["neurons"]):

    if i%16 < 10:
        neuron = NeuronRep(np.array(neuron["voltages"]), np.array(neuron["excitation_vals"]), np.array(neuron["slow_excitation_vals"]), np.array(neuron["inhibition_vals"]),
                      np.array(neuron["slow_inhibition_vals"]), np.array(neuron["g_sk_vals"]), np.array(neuron["stim_vals"]), np.array(neuron["dv_dt_vals"]), np.array(neuron["spike_times"]), "PN")
        current_pns.append(neuron)
        all_neurons.append(neuron)
    else: # 10 < i%16
        neuron = NeuronRep(np.array(neuron["voltages"]), np.array(neuron["excitation_vals"]), np.array(neuron["slow_excitation_vals"]), np.array(neuron["inhibition_vals"]),
                      np.array(neuron["slow_inhibition_vals"]), np.array(neuron["g_sk_vals"]), np.array(neuron["stim_vals"]), np.array(neuron["dv_dt_vals"]), np.array(neuron["spike_times"]), "LN")
        current_lns.append(neuron)
        all_neurons.append(neuron)
        if i%16 == 15:
            glomeruli.append(GlomerulusRep(current_pns, current_lns))
            current_pns = []
            current_lns = []


# eventplot
colors = ["blue" if neuron.type == "LN" else "red" for neuron in all_neurons]
totaldata = [neuron.spike_times for neuron in all_neurons]
plt.title("Total Glomerular Activity")
plt.eventplot(totaldata, colors=colors, linewidths=0.5, alpha=1)
plt.savefig(f"{output_dir}/total")

for i, glomerulus in enumerate(glomeruli):
    plt.figure()
    colors = ["blue" if neuron.type == "LN" else "red" for neuron in glomerulus.get_neurons()]
    totaldata = [neuron.spike_times for neuron in glomerulus.get_neurons()]
    plt.title(f"Glomerulus {i} Activity")
    plt.eventplot(totaldata, colors=colors, linewidths=0.5, alpha=1)
    plt.savefig(f"{output_dir}/glomerulus_{i}")

x_vals = np.linspace(0, duration, num=int(duration/delta_t))
plt.figure()
neuron_num = 17
neuron_1 = all_neurons[neuron_num]
x = "PN" if neuron_num%16<10 else "LN"
print(f"this is a {x}")

# plt.plot(x_vals, neuron_1.voltages, label="voltage")
# plt.plot(x_vals, neuron_1.dv_dt_vals, label="dv dt")
subtractive = neuron_1.inhibition_vals + neuron_1.slow_inhibition_vals
additive = neuron_1.excitation_vals + neuron_1.slow_excitation_vals + neuron_1.stim_vals
# plt.plot(x_vals, neuron_1.slow_inhibition_vals, label="slow", alpha=0.7)
# plt.plot(x_vals, neuron_1.slow_excitation_vals, label="slow exc", alpha=0.6)
# plt.plot(x_vals, neuron_1.inhibition_vals, label="inh", alpha=0.5)
# plt.plot(x_vals, neuron_1.excitation_vals, label="exc", alpha=0.5)
# plt.plot(x_vals, neuron_1.stim_vals, label="stim", alpha=0.5)
# plt.plot(x_vals, neuron_1.g_sk_vals, label="sk", alpha=0.4)
plt.plot(x_vals, additive, label="add", alpha=0.75)
plt.plot(x_vals, subtractive, label="sub", alpha=0.75)
plt.plot(x_vals, additive - subtractive, label="total", alpha=0.5)

plt.legend()
plt.savefig(f"{output_dir}/neuron_{neuron_num}_values")
plt.figure()
plt.hist(neuron_1.spike_times, bins=[i for i in range(0, int(duration)+500, 500)])
plt.savefig(f"{output_dir}/hist_{neuron_num}")
