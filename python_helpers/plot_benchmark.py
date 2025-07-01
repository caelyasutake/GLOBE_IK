import numpy as np
import matplotlib.pyplot as plt

# Batch sizes (categories)
batch_sizes = [1, 10, 100, 1000]
indices = np.arange(len(batch_sizes))
bar_width = 0.25

# Average IK times (in ms) for each method
globe_ik_times = [4.93084, 4.85361, 4.83469, 7.6796]
curobo_times   = [14.515423774719238, 28.21211814880371, 25.28095245361328, 96.06382846832275]
ikflow_times   = [23.04, 23.877, 33.153, 130.22]

plt.figure(figsize=(8, 6))
# Plot the bars for each method
plt.bar(indices - bar_width, globe_ik_times, width=bar_width, label='GLOBE-IK')
plt.bar(indices, curobo_times, width=bar_width, label='curobo')
plt.bar(indices + bar_width, ikflow_times, width=bar_width, label='IKFlow')

plt.xlabel("Batch Size")
plt.ylabel("Average IK Time (ms)")
plt.title("Average IK Time vs Batch Size")
plt.xticks(indices, batch_sizes)
plt.legend()
plt.grid(axis='y', linestyle='--', linewidth=0.5)

# Annotate the speedup factors at the top of the Curobo and IKFLOW bars.
# The speedup factor is computed as (other_method_time / GLOBE-IK_time).
offset = 1  # offset in ms above each bar for the text annotation

for i, bs in enumerate(batch_sizes):
    # Calculate speedup factors relative to GLOBE-IK
    speedup_curobo = curobo_times[i] / globe_ik_times[i]
    speedup_ikflow = ikflow_times[i] / globe_ik_times[i]
    
    # Annotate above the Curobo bar (centered on the bar)
    plt.text(indices[i],
             curobo_times[i] + offset,
             f"{speedup_curobo:.2f}x",
             ha='center', va='bottom', color='blue', fontsize=9)
    
    # Annotate above the IKFLOW bar (centered on the bar)
    plt.text(indices[i] + bar_width,
             ikflow_times[i] + offset,
             f"{speedup_ikflow:.2f}x",
             ha='center', va='bottom', color='green', fontsize=9)

# Save the plot as "time_vs_bs.png"
plt.savefig("time_vs_bs.png")
plt.show()
