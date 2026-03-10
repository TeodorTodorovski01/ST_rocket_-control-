import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# USER SETTINGS
# ==============================

launch_angle_deg = 84        # launch elevation [deg]
launch_azimuth_deg = 133     # launch direction [deg]

wind_speed_kmh = 20          # mean wind speed [km/h]
wind_direction_deg = 100      # wind direction [deg]
wind_turbulence = 2.0        # gust strength [m/s]

parachute_drogue_alt = 0   # drogue deploy altitude [m]
parachute_main_alt = 0    # main deploy altitude [m]

# Control gains (PD controller)
Kp_pitch = 0  # 8
Kd_pitch = 0  # 2
Kp_roll = 0  # 8
Kd_roll = 0  # 2
Kd_yaw = 0  # 1

# ==============================
# ROCKET PARAMETERS
# ==============================

mass0 = 6.0          # initial mass [kg]
prop_mass = 2.0      # propellant mass [kg]
burn_time = 8.0      # burn time [s]
thrust_force = 400   # thrust [N]

Cd_body = 0.5        # drag coefficient rocket body [-]
A_body = 0.012       # reference area [m²]

Cd_drogue = 2.0      # drogue parachute drag coefficient [-]
A_drogue = 0.3       # drogue area [m²]

Cd_main = 6.0        # main parachute drag coefficient [-]
A_main = 0.8         # main parachute area [m²]

Ix = 0.04            # moment of inertia X [kg·m²]
Iy = 0.04            # moment of inertia Y [kg·m²]
Iz = 0.01            # moment of inertia Z [kg·m²]

# ==============================
# SIMULATION PARAMETERS
# ==============================

dt = 0.01            # simulation timestep [s]
t_max = 500          # maximum simulation time [s]

g = 9.81             # gravity [m/s²]

# ==============================
# INITIAL STATE
# ==============================

pos = np.array([0.0, 0.0, 0.0])    # position [m]
vel = np.array([0.0, 0.0, 0.0])    # velocity [m/s]

angles = np.array([
    0.0,
    np.radians(launch_angle_deg) - np.pi/2,
    np.radians(launch_azimuth_deg)
])                                # Euler angles [rad]

rates = np.array([0.0, 0.0, 0.0])   # angular rates [rad/s]

mass = mass0                      # rocket mass [kg]

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

speed_log = []
acc_log = []

# ==============================
# HELPER FUNCTIONS
# ==============================


def rotation_matrix(phi, theta, psi):

    Rz = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    return Rz @ Ry @ Rx


def thrust(t):
    return thrust_force if t < burn_time else 0


def air_density(alt):
    return 1.225 * np.exp(-alt/8500)   # [kg/m³]


def wind_model(alt):

    wind_speed = wind_speed_kmh / 3.6  # convert km/h → m/s
    wind_dir = np.radians(wind_direction_deg)

    base = np.array([
        wind_speed*np.cos(wind_dir),
        wind_speed*np.sin(wind_dir),
        0
    ])

    shear = np.array([
        0.001*alt,
        0.0005*alt,
        0
    ])

    gust = np.random.normal(0, wind_turbulence, 3)

    return base + shear + gust


# ==============================
# SIMULATION LOOP
# ==============================

for step in range(int(t_max/dt)):

    t = step * dt
    alt = pos[2]

    if alt < 0 and t > 0.5:
        break

    if t < burn_time:
        mass = mass0 - prop_mass*(t/burn_time)

    T = thrust(t)

    if not drogue_deployed and alt < parachute_drogue_alt and vel[2] < 0:
        drogue_deployed = True

    if drogue_deployed and not main_deployed and alt < parachute_main_alt:
        main_deployed = True

    if main_deployed:
        Cd = Cd_main
        A = A_main
    elif drogue_deployed:
        Cd = Cd_drogue
        A = A_drogue
    else:
        Cd = Cd_body
        A = A_body

    wind = wind_model(alt)

    rel_vel = vel - wind
    V = np.linalg.norm(rel_vel)

    if V > 0:
        drag = -0.5 * air_density(alt) * V**2 * Cd * A * rel_vel / V   # [N]
    else:
        drag = np.zeros(3)

    thrust_body = np.array([0, 0, T])
    R = rotation_matrix(*angles)
    thrust_world = R @ thrust_body
    gravity = np.array([0, 0, -mass*g])
    F = thrust_world + drag + gravity
    acc = F / mass

    vel += acc * dt
    pos += vel * dt

    speed = np.linalg.norm(vel)
    acc_mag = np.linalg.norm(acc)

    phi, theta, psi = angles
    p, q, r = rates

    # ==============================
    # PD attitude control using USER SETTINGS
    Mx = -Kp_roll*phi - Kd_roll*p
    My = -Kp_pitch*theta - Kd_pitch*q
    Mz = -Kd_yaw*r

    rates += np.array([Mx/Ix, My/Iy, Mz/Iz]) * dt
    angles += rates * dt
    # ==============================

    time_log.append(t)
    x_log.append(pos[0])
    y_log.append(pos[1])
    z_log.append(pos[2])
    angle_log.append(theta)
    speed_log.append(speed)
    acc_log.append(acc_mag)

# ==============================
# PLOTS
# ==============================

speed_scale = 5
acc_scale = 20

plt.figure()
plt.plot(time_log, z_log, label="Height [m]")
plt.plot(time_log, np.array(speed_log)*speed_scale, label="Speed x5")
plt.plot(time_log, np.array(acc_log)*acc_scale, label="Acceleration x20")
plt.xlabel("Time [s]")
plt.ylabel("Scaled values")
plt.title("Rocket Height, Speed and Acceleration vs Time")
plt.legend()
plt.grid()

plt.figure()
plt.plot(x_log, z_log)
plt.xlabel("Downrange Distance X [m]")
plt.ylabel("Altitude [m]")
plt.title("Side Flight Path")
plt.grid()

plt.figure()
plt.plot(x_log, y_log)
plt.scatter(0, 0)
plt.scatter(x_log[-1], y_log[-1])
plt.xlabel("X Position [m]")
plt.ylabel("Y Position [m]")
plt.title("Top-Down Ground Track")
plt.grid()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_log, y_log, z_log)
ax.set_xlabel("X Position [m]")
ax.set_ylabel("Y Position [m]")
ax.set_zlabel("Altitude [m]")
ax.set_title("3D Rocket Trajectory")
plt.show()

# ==============================
# FLIGHT REPORT

print("\nFlight Report")
print("-------------------------")
print("Maximum altitude:", round(max(z_log), 2), "m")
landing_distance = np.sqrt(x_log[-1]**2 + y_log[-1]**2)
print("Landing distance:", round(landing_distance, 2), "m")
flight_time = time_log[-1]
print("Total flight time:", round(flight_time, 2), "s")
