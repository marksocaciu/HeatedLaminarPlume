import numpy as np
import matplotlib.pyplot as plt

# Data
height = np.array([5e-3, 1e-2, 1.5e-2, 2e-2, 2.5e-2, 3e-2, 3.5e-2])
enthalpy_flow = np.array([2.678796e-01, 3.080905e-01, 3.437198e-01, 3.802539e-01, 4.157289e-01, 4.472166e-01, 4.724517e-01])

# Supplied heating power
Q_supplied = 0.2697

# Plot
plt.figure(figsize=(6, 4))

plt.scatter(
    height,
    enthalpy_flow,
    marker="o",
    label="Plume enthalpy flow"
)

plt.axhline(
    y=Q_supplied,
    linestyle="--",
    label="Supplied heating power"
)

plt.xlabel("Height above wire [m]", size=22)
plt.ylabel("Plume enthalpy flow [W/m]", size=22)
plt.xticks(size=18)
plt.yticks(size=18)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
