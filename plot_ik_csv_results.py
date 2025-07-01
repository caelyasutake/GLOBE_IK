import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
csv_file = "globeik_iiwa_robometrics.csv"
df = pd.read_csv(csv_file)

# Strip leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

# Print column names for debugging
print("Column names:", df.columns.tolist())

# Access columns (use normalized names)
solve_time = df["Solve Time (ms)"]
pos_error = df["Pos Error (mm)"]
ang_error = df["Ang Error (deg)"]

# Plot: Solve Time vs Positional Error
plt.figure(figsize=(10, 5))
plt.scatter(solve_time, pos_error, alpha=0.7)
plt.title("Solve Time vs Positional Error")
plt.xlabel("Solve Time (ms)")
plt.ylabel("Positional Error (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Solve Time vs Angular Error
plt.figure(figsize=(10, 5))
plt.scatter(solve_time, ang_error, alpha=0.7, color='orange')
plt.title("Solve Time vs Angular Error")
plt.xlabel("Solve Time (ms)")
plt.ylabel("Angular Error (deg)")
plt.grid(True)
plt.tight_layout()
plt.show()
