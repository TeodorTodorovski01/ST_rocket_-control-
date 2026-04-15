import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# ================= INPUT =================
launch_angle_deg = float(
    input("Vnesi agol na ispaluvanje (deg, 0 = vertikalno): "))
v_max_kmh = float(input("Vnesi maksimalna brzina (km/h): "))
S_cm2 = float(input("Vnesi povrsina na zalistoci (cm^2): "))

# ================= KONSTANTI =================
rho = 1.225  # gustina na vozduh (kg/m^3)
CL_alpha = 6.0  # tipicna vrednost

# konverzii
S = S_cm2 / 10000  # cm^2 -> m^2
v_max = v_max_kmh / 3.6  # km/h -> m/s

# ================= OPSEZI =================
angles_deg = np.arange(0, 15.5, 0.5)
angles_rad = np.deg2rad(angles_deg)

velocities_kmh = np.arange(0, v_max_kmh + 10, 10)
velocities = velocities_kmh / 3.6  # m/s

# ================= PRESMETKA =================
results = []

for v_kmh, v in zip(velocities_kmh, velocities):
    for angle_deg, angle_rad in zip(angles_deg, angles_rad):

        # CL (linearen model)
        CL = CL_alpha * angle_rad

        # sila
        F = 0.5 * rho * v**2 * S * CL

        results.append([v_kmh, angle_deg, F])

# ================= DATAFRAME =================
df = pd.DataFrame(results, columns=["Brzina_kmh", "Agol_deg", "Sila_N"])

# ================= PRIKAZ =================
print("\n--- Rezultati (prvi 20 reda) ---")
print(df.head(20))

# ================= SNIMANJE =================
df.to_csv("aerodynamic_force_results.csv", index=False)

print("\nFajlot e zacuvan kako 'aerodynamic_force_results.csv'")


# pivot za mesh
pivot = df.pivot(index="Agol_deg", columns="Brzina_kmh", values="Sila_N")

X = pivot.columns.values
Y = pivot.index.values
X, Y = np.meshgrid(X, Y)
Z = pivot.values

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z)

ax.set_xlabel("Brzina (km/h)")
ax.set_ylabel("Agol (deg)")
ax.set_zlabel("Sila (N)")

plt.title("3D Surface: F(V, delta)")
plt.show()
3
