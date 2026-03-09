import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# ==============================
# USER SETTINGS
# ==============================

num_flights = 50           # Monte Carlo flights
launch_angle_deg = 84
launch_azimuth_deg = 133
wind_speed_kmh = 20
wind_dir_deg = 60
g = 9.81

parachute_drogue_alt = 900
parachute_main_alt = 300

show_animation = True

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
    if t < burn_time:
        return thrust_force
    return 0


def air_density(alt):
    return 1.225*np.exp(-alt/8500)


def wind_model(alt, base_wind):
    shear = np.array([0.001*alt, 0.0005*alt, 0])
    gust = np.random.normal(0, 0.3, 3)
    return base_wind + shear + gust

# ==============================
# MONTE CARLO SIMULATION
# ==============================


# wind base vector
wind_speed = wind_speed_kmh / 3.6
wind_dir = np.radians(wind_dir_deg)
wind_base = np.array([wind_speed*np.cos(wind_dir),
                      wind_speed*np.sin(wind_dir), 0])

# store all landing points
landing_points = []

# flight animation data
all_flights_xyz = []

for flight in range(num_flights):

    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])
    angles = np.array([0.0, np.radians(launch_angle_deg) -
                      np.pi/2, np.radians(launch_azimuth_deg)])
    rates = np.array([0.0, 0.0, 0.0])
    mass = mass0

    drogue_deployed = False
    main_deployed = False

    x_log = []
    y_log = []
    z_log = []

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
        wind = wind_model(alt, wind_base)
        rel_vel = vel - wind
        V = np.linalg.norm(rel_vel)
        if V > 0:
            drag = -0.5*air_density(alt)*V**2*Cd*A*rel_vel/V
        else:
            drag = np.zeros(3)

        thrust_body = np.array([0, 0, T])
        R = rotation_matrix(*angles)
        thrust_world = R @ thrust_body
        gravity = np.array([0, 0, -mass*g])

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

        x_log.append(pos[0])
        y_log.append(pos[1])
        z_log.append(pos[2])

    landing_points.append([pos[0], pos[1]])
    all_flights_xyz.append([x_log, y_log, z_log])

# ==============================
# PLOTS
# ==============================

# Top-down ground tracks
plt.figure()
for flight_data in all_flights_xyz:
    plt.plot(flight_data[0], flight_data[1], alpha=0.5)
plt.scatter(0, 0, color='red', label='Launch')
plt.scatter([p[0] for p in landing_points], [p[1]
            for p in landing_points], color='green', label='Landing')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Top-Down Ground Tracks - Monte Carlo")
plt.grid()
plt.legend()

# Landing spread
plt.figure()
plt.scatter([p[0] for p in landing_points], [p[1] for p in landing_points])
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Landing Points Distribution")
plt.grid()
plt.axis('equal')

# 3D trajectory of last flight
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = all_flights_xyz[-1]
ax.plot(x, y, z, label='Last Flight')
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Altitude (m)")
ax.set_title("3D Trajectory")
ax.legend()

plt.show()

# ==============================
# OPTIONAL LIVE ANIMATION
# ==============================

if show_animation:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(0, max([max(f[2]) for f in all_flights_xyz])+50)

    line, = ax.plot([], [], [], 'b-', lw=2)

    def update(num):
        line.set_data(all_flights_xyz[-1][0][:num],
                      all_flights_xyz[-1][1][:num])
        line.set_3d_properties(all_flights_xyz[-1][2][:num])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(
        all_flights_xyz[-1][0]), interval=10, blit=True)
    plt.show()
