import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# USER SETTINGS
# ==============================

launch_angle_deg = 84
launch_azimuth_deg = 133

wind_speed_kmh = 20     # mean wind speed
wind_direction_deg = 60  # wind direction (from north)
wind_turbulence = 2.0   # random gust strength (m/s)

parachute_drogue_alt = 120
parachute_main_alt = 50

# ==============================
# ROCKET PARAMETERS
# ==============================

mass0 = 6.0
prop_mass = 2.0
burn_time = 4.0
thrust_force = 250

Cd_body = 0.5
A_body = 0.012

Cd_drogue = 2.0
A_drogue = 0.3

Cd_main = 6.0
A_main = 0.8

Ix = 0.04
Iy = 0.04
Iz = 0.01

# ==============================
# SIMULATION PARAMETERS
# ==============================

dt = 0.01
t_max = 60

# ==============================
# INITIAL STATE
# ==============================

pos = np.array([0.0, 0.0, 0.0])
vel = np.array([0.0, 0.0, 0.0])

angles = np.array([0.0, np.radians(launch_angle_deg) -
                  np.pi/2, np.radians(launch_azimuth_deg)])
rates = np.array([0.0, 0.0, 0.0])

mass = mass0
drogue_deployed = False
main_deployed = False

# ==============================
# LOGS
# ==============================

x_log = []
y_log = []
z_log = []
time_log = []
angle_log = []

# ==============================
# HELPER FUNCTIONS
# ==============================


def rotation_matrix(phi, theta, psi):
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    return Rz @ Ry @ Rx


def thrust(t):
    return thrust_force if t < burn_time else 0


def air_density(alt):
    return 1.225*np.exp(-alt/8500)


def wind_model(alt):
    # mean wind vector
    wind_speed = wind_speed_kmh / 3.6
    wind_dir = np.radians(wind_direction_deg)
    base = np.array([wind_speed*np.cos(wind_dir),
                    wind_speed*np.sin(wind_dir), 0])
    shear = np.array([0.001*alt, 0.0005*alt, 0])
    gust = np.random.normal(0, wind_turbulence, 3)
    return base + shear + gust

# ==============================
# SIMULATION LOOP
# ==============================


for step in range(int(t_max/dt)):
    t = step*dt
    alt = pos[2]

    if alt < 0 and t > 0.5:
        break

    # mass
    if t < burn_time:
        mass = mass0 - prop_mass*(t/burn_time)

    # thrust
    T = thrust(t)

    # parachutes
    if not drogue_deployed and alt < parachute_drogue_alt and vel[2] < 0:
        drogue_deployed = True
    if drogue_deployed and not main_deployed and alt < parachute_main_alt:
        main_deployed = True

    # choose drag
    if main_deployed:
        Cd = Cd_main
        A = A_main
    elif drogue_deployed:
        Cd = Cd_drogue
        A = A_drogue
    else:
        Cd = Cd_body
        A = A_body

    # wind
    wind = wind_model(alt)
    rel_vel = vel - wind
    V = np.linalg.norm(rel_vel)
    drag = -0.5*air_density(alt)*V**2*Cd*A * \
        (rel_vel/V) if V > 0 else np.zeros(3)

    thrust_body = np.array([0, 0, T])
    R = rotation_matrix(*angles)
    thrust_world = R @ thrust_body
    gravity = np.array([0, 0, -mass*9.81])

    F = thrust_world + drag + gravity
    acc = F/mass

    vel += acc*dt
    pos += vel*dt

    # attitude stabilization
    phi, theta, psi = angles
    p, q, r = rates
    Mx = -8*phi - 2*p
    My = -8*theta - 2*q
    Mz = -0.3*r
    rates += np.array([Mx/Ix, My/Iy, Mz/Iz])*dt
    angles += rates*dt

    # logging
    time_log.append(t)
    x_log.append(pos[0])
    y_log.append(pos[1])
    z_log.append(pos[2])
    angle_log.append(theta)

# ==============================
# PLOTS
# ==============================

# Altitude
plt.figure()
plt.plot(time_log, z_log)
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.title("Rocket Altitude")
plt.grid()

# Side view
plt.figure()
plt.plot(x_log, z_log)
plt.xlabel("X Distance (m)")
plt.ylabel("Altitude (m)")
plt.title("Side Flight Path")
plt.grid()

# Top-down ground track
plt.figure()
plt.plot(x_log, y_log, label="Flight Path")
plt.scatter(0, 0, color='red', label="Launch")
plt.scatter(x_log[-1], y_log[-1], color='green', label="Landing")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Top-Down Ground Track")
plt.legend()
plt.grid()

# 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_log, y_log, z_log, label="Rocket Path")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Altitude (m)")
ax.set_title("3D Trajectory")
ax.legend()

plt.show()

# ==============================
# REPORT
# ==============================

print("Max altitude:", max(z_log), "m")
landing_distance = np.sqrt(x_log[-1]**2 + y_log[-1]**2)
print("Landing distance:", landing_distance, "m")
