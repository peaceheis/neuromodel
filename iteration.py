import os
import subprocess
from dataclasses import dataclass
import json

import matplotlib.pyplot as plt
import numpy as np

spike_data = {}




def neuron_key(nrn, n_index):
    return f"{nrn.n_type}_{n_index}"


def serialize_neuron(nrn, n_index, num_runs):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    n_key = neuron_key(nrn, n_index)

    # Store new spike counts
    spike_data[n_key] += nrn.binned_spike_counts


def plot_trials(n_index, n_key, nrn, num_runs):
    all_trials = np.array(spike_data.values())
    all_trials /= num_runs
    print(all_trials)
    x = np.linspace(0, duration, num_bins)
    # histogram
    plt.figure()
    plt.bar(x, all_trials, alpha=0.75, color='blue', edgecolor='black')
    plt.ylabel("Firing Rate (Hz)")
    plt.xlabel("Time (ms)")
    plt.title(f"Neuron {n_index} ({nrn.n_type}) Average Spike Count Histogram")
    # Save figure
    plt.savefig(f"{output_dir}/histogram_{nrn.type}_{i * 16 + n_index}.png")

    print(f"Histogram saved for Neuron {neuron_key}")


def retrieve():
    prefix = json.load(open("config.json"))["prefix"]
    data = json.load(open(f"{prefix}result.json"))

    output_dir = data["dir"]
    NUM_PNS = data["num_pns"]
    NUM_LNS = data["num_lns"]
    stim_time = data["stim_time"]
    duration = data["duration"]
    delta_t = data["delta_t"]
    matrix = data["connectivity_matrix"]
    skipped_vals: bool = data[
        "skipped_vals"]  # keeps track of whether only every 10th value was recorded for neuron data or not

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
        spike_counts: np.array
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


# loop to iterate through multiple trials
trials = 5

for i in range(trials):
    # run neuromodel
    result = subprocess.run(["C:\\Users\\wjtor\\RustRoverProjects\\neuromodel\\target\\release\\neuromodel.exe"])
    # capture the output for the individual run in the same manner as we did in graph.py
    prefix = json.load(open("config.json"))["prefix"]
    data = json.load(open(f"{prefix}result.json"))

    output_dir = data["dir"]
    NUM_PNS = data["num_pns"]
    NUM_LNS = data["num_lns"]
    stim_time = data["stim_time"]
    duration = data["duration"]
    delta_t = data["delta_t"]
    matrix = data["connectivity_matrix"]
    skipped_vals: bool = data[
        "skipped_vals"]  # keeps track of whether only every 10th value was recorded for neuron data or not


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
        spike_counts: np.array
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
    spike_counts_for_sum = []

    for i, neuron in enumerate(data["neurons"]):

        if i % 16 < NUM_PNS:
            neuron = NeuronRep(np.array(neuron["voltages"]), np.array(neuron["excitation_vals"]),
                               np.array(neuron["slow_excitation_vals"]), np.array(neuron["inhibition_vals"]),
                               np.array(neuron["slow_inhibition_vals"]), np.array(neuron["g_sk_vals"]),
                               np.array(neuron["stim_vals"]), np.array(neuron["dv_dt_vals"]),
                               np.array(neuron["spike_times"]),
                               np.array(neuron["spike_counts"]), "PN")
            current_pns.append(neuron)
            all_neurons.append(neuron)
        else:  # 10 < i%16
            neuron = NeuronRep(np.array(neuron["voltages"]), np.array(neuron["excitation_vals"]),
                               np.array(neuron["slow_excitation_vals"]), np.array(neuron["inhibition_vals"]),
                               np.array(neuron["slow_inhibition_vals"]), np.array(neuron["g_sk_vals"]),
                               np.array(neuron["stim_vals"]), np.array(neuron["dv_dt_vals"]),
                               np.array(neuron["spike_times"]), np.array(neuron["spike_counts"]), "LN")
            current_lns.append(neuron)
            all_neurons.append(neuron)
            if i % 16 == 15:
                glomeruli.append(GlomerulusRep(current_pns, current_lns))
                current_pns = []
                current_lns = []

    # simplified version
    for i in range(6):
        pn_index = 2
        pn = glomeruli[i].neurons[pn_index]
        generate_spike_count_histogram(pn, pn_index, i, trials, output_dir="output")
    plt.show()
    # pick some select neurons to examine
    # selected_neurons = [2]
    # for neuron in selected_neurons:
    #     print(neuron.spike_counts)
