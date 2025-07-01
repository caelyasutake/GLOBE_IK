import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Read error values from ik.csv (assumes a header with a column named "Error", already in mm)
ik_data = pd.read_csv('ik.csv')
errors_IKFlow = ik_data['Error'].values  # errors in mm

# Read error values from globeik_iiwa_errors.txt by parsing lines that contain "Error:"
errors_GlobeIK = []
with open('globeik_iiwa_errors.txt', 'r') as f:
    for line in f:
        if "Error:" in line:
            match = re.search(r"Error:\s*([\deE\.\-]+)", line)
            if match:
                errors_GlobeIK.append(float(match.group(1)))
errors_GlobeIK = np.array(errors_GlobeIK)

# Read error values from curobo_iiwa_errors.csv
# The "Error" column may contain strings like "[0.0013383734039962292]" or numeric values
curobo_data = pd.read_csv(r'curobo_iiwa_errors.csv')

def parse_error(val):
    if isinstance(val, str):
        return float(val.strip('[]'))
    return float(val)

errors_Curobo = curobo_data['Error'].apply(parse_error).values

# Define x positions for the scatter plot groups (closer together)
x_IKFlow = np.full(len(errors_IKFlow), 1)
x_GlobeIK = np.full(len(errors_GlobeIK), 1.2)
x_Curobo  = np.full(len(errors_Curobo), 1.4)

plt.figure(figsize=(8, 6))

# Plot the individual error values for each solver
plt.scatter(x_IKFlow, errors_IKFlow, color='blue', alpha=0.6, label='IKFlow')
plt.scatter(x_GlobeIK, errors_GlobeIK, color='red', alpha=0.6, label='GlobeIK')
plt.scatter(x_Curobo, errors_Curobo, color='green', alpha=0.6, label='Curobo')

# Set y-axis limits to 0 - 250 mm with evenly spaced ticks (e.g., every 50 mm)
plt.ylim(0, 250)
plt.yticks(np.linspace(0, 250, 6))

# Set the x-axis ticks and labels
plt.xticks([1, 1.2, 1.4], ['IKFlow', 'GlobeIK', 'Curobo'])
plt.xlabel('Solver')
plt.ylabel('Error (mm)')
plt.title('Solver Error Comparison')
plt.legend()

# Remove horizontal grid lines
plt.grid(False)

# Save the figure as a PNG file
plt.savefig('scatter_plot.png', dpi=300)
plt.show()
