import json
import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from model import Network, DELTA_T, Neuron

duration = 5000
stim_time = 500
BIN_SIZE = duration / 10
assert(duration % BIN_SIZE == 0)
steps = int(duration / DELTA_T)
prefix = json.load(open("config.json"))["prefix"]

network = Network(stim_time, "Additive", [0, 1, 2, 3, 4, 5])

vals = np.linspace(0, duration, steps)  # linspace for iteration of the network

for val in vals:
    network.update()
    print(val)


for neuron in network.glomeruli[0].get_neurons():
     neuron.render(vals)

def save_constants(prefix_):
    constants = {
        "V_EXC": Neuron.V_EXC,
        "V_INH": Neuron.V_INH,
        "TAU_INH": Neuron.TAU_INH,
        "TAU_SLOW": Neuron.TAU_SLOW,
        "TAU_EXC": Neuron.TAU_EXC,
        "TAU_EXC_SLOW": Neuron.TAU_EXC_SLOW,
        "TAU_STIM": Neuron.TAU_STIM,
        "TAU_SK": Neuron.TAU_SK,
        "STIM_DURATION": Neuron.STIM_DURATION,

        "SK_MU": Neuron.SK_MU,
        "SK_STDEV": Neuron.SK_STDEV,

        "LN_ODOR_TAU_RISE": Neuron.LN_ODOR_TAU_RISE,
        "LN_MECH_TAU_RISE": Neuron.LN_MECH_TAU_RISE,
        "LN_S_PN": Neuron.LN_S_PN,
        "LN_S_PN_SLOW": Neuron.LN_S_PN_SLOW,
        "LN_S_INH": Neuron.LN_S_INH,
        "LN_S_SLOW": Neuron.LN_S_SLOW,
        "LN_L_STIM": Neuron.LN_L_STIM,

        "PN_ODOR_TAU_RISE": Neuron.PN_ODOR_TAU_RISE,
        "PN_MECH_TAU_RISE": Neuron.PN_MECH_TAU_RISE,
        "PN_S_PN": Neuron.PN_S_PN,
        "PN_S_PN_SLOW": Neuron.PN_S_PN_SLOW,
        "PN_S_INH": Neuron.PN_S_INH,
        "PN_S_SLOW": Neuron.PN_S_SLOW,
        "PN_S_STIM": Neuron.PN_S_STIM

    }

    json.dump(constants, open(prefix_ + "constants.json", "w"), indent=4)


should_serialize = True

if should_serialize:
    rn = datetime.now()
    prefix = prefix + f"{rn.year}{rn.month}{rn.day}-{rn.hour}.{rn.minute}.{rn.second}-{rn.microsecond}/"
    os.mkdir(prefix)
    save_constants(prefix)

totaldata = []
for i, glomerulus in enumerate(network.glomeruli):
    data = []
    for neuron in glomerulus.get_neurons():
        data.append(neuron.spike_times)
        totaldata.append(neuron.spike_times)

    fig, axs = plt.subplots(2, 2, figsize=(14, 5))
    fig.suptitle(f"Glomerulus {i}")

    colors = ["blue" if neuron.neuron_type == "LN" else "red" for neuron in
              glomerulus.get_neurons()]

    axs[0, 0].eventplot(data, colors=colors)
    axs[0, 0].set_title(f"Eventplot")

    axs[0, 1].eventplot([[n.n_id for n in neuron.connected_neurons] for neuron in glomerulus.get_neurons()],
                        colors=colors)
    axs[0, 1].set_title(f"Connectivity")

    axs[1, 0].bar([i for i in range(1, 17)], [neuron.total_excitation for neuron in glomerulus.get_neurons()],
                  width=0.25, color=colors)
    axs[1, 0].set_xlabel("Excitation Amounts")

    axs[1, 1].bar([i for i in range(1, 17)], [neuron.total_inhibition for neuron in glomerulus.get_neurons()],
                  width=0.25, color=colors)
    axs[1, 1].set_xlabel("Inhibition Amounts")
    ''''''
    if should_serialize:
        plt.savefig(prefix + f"{i}.png")
    else:
        pass
    #plt.show()

    for neuron in glomerulus.get_neurons():
        print(
            f"Glomerulus {i} - Neuron {neuron.n_id} Inhibition {neuron.total_inhibition}, len {len(neuron.inh_times)} SK {neuron.s_sk if neuron.neuron_type == 'PN' else 0}, inh {len(neuron.inh_times)}, exc {len(neuron.exc_times)}")
        print(
            f"Glomerulus {i} - Neuron {neuron.n_id} Excitation {neuron.total_excitation}, len {len(neuron.inh_times)} SK {neuron.s_sk if neuron.neuron_type == 'PN' else 0}, inh {len(neuron.inh_times)}, exc {len(neuron.exc_times)}")

plt.figure()
plt.title("Total Glomerular Activity")
plt.eventplot(totaldata, colors='blue')
if should_serialize:
    plt.savefig(prefix + f"total")
else:
    pass
    #plt.show()

# for neuron in network.glomeruli[3].get_neurons():
#     plt.figure()
#     plt.title(f"Neuron {neuron.n_id} G Sk")
#     plt.plot(np.multiply(10, neuron.g_sk_vals))
#     plt.plot(neuron.voltages)
#     plt.show()
for glomerulus in network.glomeruli:
    plt.figure()
    plt.title(f"Glomerulus {glomerulus.g_id} PN Firing Rates")
    x = np.linspace(0, duration, num=int(duration / BIN_SIZE))

    pn_rates, ln_rates = glomerulus.get_normalized_average_firing_rates(duration, BIN_SIZE)
    plt.bar(x, pn_rates, width=10, align='edge', ec='blue')
    if should_serialize:
        plt.savefig(prefix + f"{glomerulus.g_id}_rates_pn")
    else:
        pass
        #plt.show()

    plt.figure()
    plt.title(f"Glomerulus {glomerulus.g_id} LN Firing Rates")
    print(x)
    plt.bar(x, ln_rates, width=10, align='edge', ec='blue')
    if should_serialize:
        plt.savefig(prefix + f"{glomerulus.g_id}_rates_ln")
    else:
        pass
#plt.show()

# plt.figure(16 + i)
# plt.title(f"Glomerulus {i} PN Rates")
# plt.bar(rate_bins, pn_rates)
#
# plt.show()
#
#
# plt.figure(32 + i)
# plt.title(f"Glomerulus {i} Ln Rates")
# plt.bar(rate_bins, ln_rates)
# print("ok here")
# plt.show()
# print("yay!")

# for i in range(6):
# glomerulus = Glomerulus(1000, Neuron.LAMBDA_ODOR_MAX, Neuron.LAMBDA_MECH_MAX)
# neuron = Neuron(250, 0, Neuron.LAMBDA_MECH_MAX, "PN")
#
# steps = np.linspace(0, duration, num=int(duration / DELTA_T))
#
# for step in steps:
#     print(step)
#     glomerulus.update()
#
# neurons = glomerulus.pns
# neurons.extend(glomerulus.lns)
# plt.figure(figsize=(6.4, 4.8))
# plt.title(f"Spike Count - {neuron.neuron_type} - Lambda Odor: {neuron.lambda_odor}, Lambda Mech: {neuron.lambda_mech}")
# plt.plot(steps, neuron.spike_counts, color="red" if neuron.neuron_type == "LN" else "blue")
# plt.show()
# plt.figure()
# plt.title(f"Voltage - {neuron.neuron_type} - Lambda Odor: {neuron.lambda_odor}, Lambda Mech: {neuron.lambda_mech}")
# plt.plot(steps, neuron.voltages, color="red" if neuron.neuron_type == "LN" else "blue")
# plt.show()

'''
neuron2.connected_neurons.append(neuron)
neuron.connected_neurons.append(neuron2)

for i, neuron in enumerate(neurons):
    plt.figure(figsize=(8, 6))
    plt.plot(steps, neuron.spike_counts, color=([f'C{i}' for i in range(10)] + [f'C{i}' for i in range(6)])[i], linewidth=1)
    plt.show()
    data.append(neuron.spike_times)
plt.eventplot(data, colors=[f'C{i}' for i in range(10)] + [f'C{i}' for i in range(6)], lineoffsets=2, linewidths=1)
print(data)
plt.show()

for step in steps:
    neuron.update()
    neuron2.update()
    if int(neuron.t) == 1000:
        bg_count = len(neuron.spike_times)
        bg_count_2 = len(neuron2.spike_times)
    if int(neuron.t) == 1500:
        last_500 = len(neuron.spike_times)
        last_500_2 = len(neuron2.spike_times)

bg_rate = bg_count / 1000
bg_rate_2 = bg_count_2 / 1000
end_count = len(neuron.spike_times) - last_500
end_count_2 = len(neuron2.spike_times) - last_500_2
end_rate = end_count / 500
end_rate_2 = end_count_2 / 500

print(f"FIRING COUNTS - NEURON 1 - 1s bg {bg_count}, last 500ms {end_count}")
print(f"FIRING COUNTS - NEURON 2 - 1s bg {bg_count_2}, last 500ms {end_count_2}")
print(f"FIRING RATES - NEURON 1 - 1s bg {bg_rate}, last 500ms {end_rate}")
print(f"FIRING RATES - NEURON 2 - 1s bg {bg_rate_2}, last 500ms {end_rate_2}")
print(f"NORMALIZED RATE - NEURON 1 - {end_rate / bg_rate}")
print(f"NORMALIZED RATE - NEURON 2 - {end_rate_2 / bg_rate_2}")
'''
plt.show()
