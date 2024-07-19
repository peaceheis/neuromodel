import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy

prefix = json.load(open("config.json"))["prefix"]
data = json.load(open(f"{prefix}result.json"))

output_dir = data["dir"]

stim_time = data["stim_time"]
duration = data["duration"]
delta_t = data["delta_t"]

@dataclass
class NeuronRep:
    excitation_vals: list[float]
    slow_excitation_vals: list[float]
    inhibition_vals: list[float]
    slow_inhibition_vals: list[float]
    g_sk_vals: list[float]
    spike_times: list[float]

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
        neuron = NeuronRep(neuron["excitation_vals"], neuron["slow_excitation_vals"], neuron["inhibition_vals"],
                      neuron["slow_inhibition_vals"], neuron["g_sk_vals"], neuron["spike_times"])
        current_pns.append(neuron)
        all_neurons.append(neuron)
    else: # 10 < i%16
        neuron = NeuronRep(neuron["excitation_vals"], neuron["slow_excitation_vals"], neuron["inhibition_vals"],
                      neuron["slow_inhibition_vals"], neuron["g_sk_vals"], neuron["spike_times"])
        current_lns.append(neuron)
        all_neurons.append(neuron)
        if i%16 == 15:
            glomeruli.append(GlomerulusRep(current_pns, current_lns))
            current_pns = []
            current_lns = []


# eventplot

totaldata = [neuron.spike_times for neuron in all_neurons]
plt.figure()
plt.title("Total Glomerular Activity")
plt.eventplot(totaldata, colors='blue')
plt.savefig(f"{output_dir}/total")

