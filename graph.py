import json
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy
import numpy as np


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
    binned_spike_counts: np.array
    n_type: str


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


prefix = json.load(open("config.json"))["prefix"]
NUM_PNS = data["num_pns"]
NUM_LNS = data["num_lns"]
stim_time = data["stim_time"]
duration = data["duration"]
delta_t = data["delta_t"]
bin_size = data["bin_size"]
num_bins = data["num_bins"]
skipped_vals = data["skipped_vals"]  # keeps track of whether only every 10th value was recorded for neuron data or not


def initialize():
    global data, output_dir, NUM_PNS, NUM_LNS, duration, delta_t, matrix

    prefix = json.load(open("config.json"))["prefix"]
    data = json.load(open(f"{prefix}result.json"))
    output_dir = data["dir"]

    glomeruli = []
    all_neurons = []
    current_pns = []
    current_lns = []

    for i, neuron in enumerate(data["neurons"]):

        if i % 16 < NUM_PNS:
            neuron = NeuronRep(np.array(neuron["voltages"]), np.array(neuron["excitation_vals"]),
                               np.array(neuron["slow_excitation_vals"]), np.array(neuron["inhibition_vals"]),
                               np.array(neuron["slow_inhibition_vals"]), np.array(neuron["g_sk_vals"]),
                               np.array(neuron["stim_vals"]), np.array(neuron["dv_dt_vals"]),
                               np.array(neuron["spike_times"]),
                               np.array(neuron["binned_spike_counts"]), "PN")

            current_pns.append(neuron)
            all_neurons.append(neuron)
        else:  # 10 < i%16
            neuron = NeuronRep(np.array(neuron["voltages"]), np.array(neuron["excitation_vals"]),
                               np.array(neuron["slow_excitation_vals"]), np.array(neuron["inhibition_vals"]),
                               np.array(neuron["slow_inhibition_vals"]), np.array(neuron["g_sk_vals"]),
                               np.array(neuron["stim_vals"]), np.array(neuron["dv_dt_vals"]),
                               np.array(neuron["spike_times"]),
                               np.array(neuron["binned_spike_counts"]), "LN")

            current_lns.append(neuron)
            all_neurons.append(neuron)
            if i % 16 == 15:
                glomeruli.append(GlomerulusRep(current_pns, current_lns))
                current_pns = []
                current_lns = []

    return glomeruli, current_pns, current_lns, all_neurons, output_dir,


glomeruli, current_pns, current_lns, all_neurons, output_dir = initialize()

# eventplot
colors = ["blue" if neuron.n_type == "LN" else "red" for neuron in all_neurons]
totaldata = [neuron.spike_times for neuron in all_neurons]
plt.title("Total Glomerular Activity")
plt.eventplot(totaldata, colors=colors, linewidths=0.5, alpha=1)
plt.savefig(f"{output_dir}/total")

for i, glomerulus in enumerate(glomeruli):
    plt.figure()
    colors = ["blue" if neuron.n_type == "LN" else "red" for neuron in glomerulus.get_neurons()]
    totaldata = [neuron.spike_times for neuron in glomerulus.get_neurons()]
    plt.title(f"Glomerulus {i} Activity")
    plt.eventplot(totaldata, colors=colors, linewidths=0.5, alpha=1)
    plt.savefig(f"{output_dir}/glomerulus_{i}")

x_vals = np.linspace(0, duration, num=int(duration / (delta_t * (10 - 9 * (not skipped_vals)))))


def generate_vals_graph(nrn, n_index):
    global x_vals
    plt.figure()
    # plt.plot(x_vals, nrn.voltages, label="voltage")
    # plt.plot(x_vals, nrn.dv_dt_vals, label="dv dt")
    subtractive = nrn.inhibition_vals + nrn.slow_inhibition_vals
    additive = nrn.excitation_vals + nrn.slow_excitation_vals + nrn.stim_vals
    plt.plot(x_vals, nrn.slow_inhibition_vals, label="slow", alpha=0.7, color="pink")
    # plt.plot(x_vals, nrn.slow_excitation_vals, label="slow exc", alpha=0.6, color="green")
    plt.plot(x_vals, nrn.inhibition_vals, label="inh", alpha=0.5, color="red")
    # plt.plot(x_vals, nrn.excitation_vals, label="exc", alpha=0.5)
    plt.plot(x_vals, nrn.stim_vals, label="stim", alpha=0.4, color="green")
    # plt.plot(x_vals, nrn.g_sk_vals, label="sk", alpha=0.4)
    plt.plot(x_vals, additive, label="add", alpha=0.5, color="orange")
    plt.plot(x_vals, subtractive, label="sub", alpha=0.5, color="blue")
    plt.plot(x_vals, additive - subtractive, label="total", alpha=0.6, color="purple")
    plt.legend()
    plt.savefig(f"{output_dir}/neuron_{n_index}_{nrn.n_type}_values")
    plt.close()


neuron_num = 36
neuron_1 = all_neurons[neuron_num]
print(f"this is a {neuron_1.n_type}")
generate_vals_graph(neuron_1, neuron_num)

binned_x_var = np.linspace(0, duration, num=num_bins)


def generate_firing_rate_graph(nrn, n_index):
    plt.figure()
    print(binned_x_var)
    plt.bar(binned_x_var, nrn.binned_spike_counts, width=20, align="edge", tick_label=binned_x_var)
    plt.title(f"Neuron {i * 16 + n_index} ({nrn.n_type}) Firing Counts")
    plt.savefig(f"{output_dir}/hist_{nrn.n_type}_{i * 16 + n_index}")
    plt.close()


for i in range(6):
    pn_index = random.randint(0, NUM_PNS - 1)
    pn = glomeruli[i].neurons[pn_index]
    generate_firing_rate_graph(pn, pn_index)
    generate_vals_graph(pn, i * 16 + pn_index)

    ln_index = random.randint(NUM_PNS, 15)
    ln = glomeruli[i].neurons[ln_index]
    generate_firing_rate_graph(ln, ln_index)
    generate_vals_graph(ln, i * 16 + ln_index)

    plt.figure()
    nrns = glomeruli[i].neurons

    plt.title(f"Glomerulus {i} Continuous PN Firing Rate")
    plt.ylabel("Firing Rate, Hz")
    plt.savefig(f"{output_dir}/glom_{i}_pn_firing_rate")
    plt.close()

    plt.figure()

    plt.title(f"Glomerulus {i} Continuous LN Firing Rate")
    plt.ylabel("Firing Rate, Hz")
    plt.savefig(f"{output_dir}/glom_{i}_ln_firing_rate")
    plt.close()

# connectivity matrix

plt.figure()
colors = ["blue" if neuron.n_type == "LN" else "red" for neuron in all_neurons]
plt.eventplot([[connection for connection in row] for row in matrix], colors=colors)
plt.savefig(f"{output_dir}/matrix")
