import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from graph import duration, num_bins, initialize, output_dir

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
    plt.savefig(f"{output_dir}/histogram_{nrn.n_type}_{n_index}.png")
    print(f"Histogram saved for Neuron {n_key}")


# loop to iterate through multiple trials
trials = 6
glomeruli, all_neurons, current_pns, current_lns = initialize()

for i in range(trials):
    # run neuromodel
    result = subprocess.run(["cargo", "run"])
    # capture the output for the individual run in the same manner as we did in graph.py

    if i == 0:
        for n_index, nrn in enumerate(all_neurons):
            n_key = neuron_key(nrn, n_index)
            spike_data[n_key] = np.zeros((num_bins,))

    for n_index, nrn, in enumerate(all_neurons):
        n_key = neuron_key(nrn, n_index)
        serialize_neuron(nrn, n_index, trials)

for n_index, nrn, in enumerate(all_neurons):
    n_key = neuron_key(nrn, n_index)
    plot_trials(n_index, neuron_key, nrn, trials)

plt.show()
