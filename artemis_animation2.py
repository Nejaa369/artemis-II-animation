"""
Artemis II — Free-Return Trajectory  (physics-based, clean)
=============================================================
Verified trajectory: launch → Earth orbit → TLI → lunar flyby (day 3.1)
→ swings around Moon → returns to Earth (day 6.6).

Phase colours:
  Orange  — Launch
  Cyan    — Earth Orbit
  Red     — TLI Burn
  Yellow  — Coast to Moon
  Pink    — Lunar Flyby
  Green   — Return to Earth

Labels: placed in a clean side panel, never overlapping the animation.

Requirements:  pip install numpy matplotlib
Run:           python artemis_ii.py
Controls:      Space = pause/resume | R = restart | +/- = speed | Q = quit
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation

# ═══════════════════════════════════════════════════════════════
# 1.  PHYSICS
# ═══════════════════════════════════════════════════════════════
MU_E        = 3.986004418e14
MU_M        = 4.9048695e12
R_E         = 6_378_137.0
R_M         = 1_737_100.0
D_EM        = 384_400_000.0
MOON_OMEGA  = 2*np.pi / (27.3 * 86400)
MOON_PHASE0 = np.radians(130.0)   # tuned: flyby day 3.14, splashdown day 6.6

R_PARK  = R_E + 300_000.0
V_PARK  = np.sqrt(MU_E / R_PARK)
V_TLI   = 10_850.0

def moon_pos(t):
    a = MOON_PHASE0 + MOON_OMEGA * t
    return D_EM * np.cos(a), D_EM * np.sin(a)

def deriv(s, t):
    x, y, vx, vy = s
    mx, my = moon_pos(t)
    re3 = (x*x + y*y)**1.5
    dx, dy = x-mx, y-my
    rm3 = (dx*dx + dy*dy)**1.5
    return np.array([vx, vy,
                     -MU_E*x/re3 - MU_M*dx/rm3,
                     -MU_E*y/re3 - MU_M*dy/rm3])

def rk4(s, t, dt):
    k1 = deriv(s, t)
    k2 = deriv(s + 0.5*dt*k1, t + 0.5*dt)
    k3 = deriv(s + 0.5*dt*k2, t + 0.5*dt)
    k4 = deriv(s +     dt*k3, t +     dt)
    return s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ═══════════════════════════════════════════════════════════════
# 2.  BUILD TRAJECTORY
# ═══════════════════════════════════════════════════════════════
print("Building trajectory ...")
M = 1e6   # metres to Mm

# Phase 0: Launch — straight up along +X from surface to parking orbit
N_LAUNCH = 80
r_launch = np.linspace(R_E, R_PARK, N_LAUNCH)
L_x  = r_launch
L_y  = np.zeros(N_LAUNCH)
L_vx = np.zeros(N_LAUNCH)
L_vy = np.linspace(0, V_PARK, N_LAUNCH)

# Phase 1: Earth parking orbit — 2 full CCW loops starting at angle 0
N_ORBIT = 700
angles  = np.linspace(0, 2*np.pi*2.0, N_ORBIT)
O_x  =  R_PARK * np.cos(angles)
O_y  =  R_PARK * np.sin(angles)
O_vx = -V_PARK * np.sin(angles)
O_vy =  V_PARK * np.cos(angles)

# Phase 2: TLI burn — short outward spiral from angle 0
N_BURN  = 60
burn_r  = np.linspace(R_PARK, R_PARK * 1.02, N_BURN)
burn_a  = np.linspace(0, np.radians(3), N_BURN)
B_x  =  burn_r * np.cos(burn_a)
B_y  =  burn_r * np.sin(burn_a)
B_vx = -V_TLI  * np.sin(burn_a)
B_vy =  V_TLI  * np.cos(burn_a)

# Phases 3-5: Free-return (real gravity integration)
# Start: (R_PARK, 0), velocity (0, V_TLI)
DT_FINE = 60.0
DAYS    = 7.0
N_CRUS  = int(DAYS * 86400 / DT_FINE)

cx  = np.zeros(N_CRUS);  cy  = np.zeros(N_CRUS)
cvx = np.zeros(N_CRUS);  cvy = np.zeros(N_CRUS)
cmx = np.zeros(N_CRUS);  cmy = np.zeros(N_CRUS)

s = np.array([R_PARK, 0.0, 0.0, V_TLI])
for i in range(N_CRUS):
    t = i * DT_FINE
    cx[i], cy[i]   = s[0], s[1]
    cvx[i], cvy[i] = s[2], s[3]
    cmx[i], cmy[i] = moon_pos(t)
    s = rk4(s, t, DT_FINE)
    if i > N_CRUS // 3 and np.hypot(s[0], s[1]) < R_E + 100_000:
        N_CRUS = i + 1
        cx=cx[:N_CRUS]; cy=cy[:N_CRUS]
        cvx=cvx[:N_CRUS]; cvy=cvy[:N_CRUS]
        cmx=cmx[:N_CRUS]; cmy=cmy[:N_CRUS]
        break

print(f"Cruise: {N_CRUS} steps ({N_CRUS*DT_FINE/86400:.2f} days)")

rm_c      = np.hypot(cx-cmx, cy-cmy)
closest_i = int(np.argmin(rm_c))

def cruise_phase(i):
    if rm_c[i] < 20e6: return 4
    if i < closest_i:  return 3
    return 5

cruise_ph = np.array([cruise_phase(i) for i in range(N_CRUS)])

# Downsample cruise 4x
DS = 4
cx_d  = cx[::DS];  cy_d  = cy[::DS]
cvx_d = cvx[::DS]; cvy_d = cvy[::DS]
cmx_d = cmx[::DS]; cmy_d = cmy[::DS]
cph_d = cruise_ph[::DS]
NC    = len(cx_d)

all_x  = np.concatenate([L_x,  O_x,  B_x,  cx_d])  / M
all_y  = np.concatenate([L_y,  O_y,  B_y,  cy_d])  / M
all_vx = np.concatenate([L_vx, O_vx, B_vx, cvx_d])
all_vy = np.concatenate([L_vy, O_vy, B_vy, cvy_d])
all_ph = np.concatenate([
    np.full(N_LAUNCH, 0, int),
    np.full(N_ORBIT,  1, int),
    np.full(N_BURN,   2, int),
    cph_d
])

moon0 = np.array(moon_pos(0)) / M
moon_xa = np.concatenate([
    np.full(N_LAUNCH + N_ORBIT + N_BURN, moon0[0]),
    cmx_d / M
])
moon_ya = np.concatenate([
    np.full(N_LAUNCH + N_ORBIT + N_BURN, moon0[1]),
    cmy_d / M
])

N_TOT = len(all_x)
print(f"Total animation steps: {N_TOT}")

# ═══════════════════════════════════════════════════════════════
# 3.  PHASE METADATA
# ═══════════════════════════════════════════════════════════════
PHASES = {
    0: ("① Launch",           "#ff8c00"),
    1: ("② Earth Orbit",      "#40c8ff"),
    2: ("③ TLI Burn",         "#ff3030"),
    3: ("④ Coast to Moon",    "#ffe066"),
    4: ("⑤ Lunar Flyby",      "#ff60a0"),
    5: ("⑥ Return to Earth",  "#80ff90"),
}

# ═══════════════════════════════════════════════════════════════
# 4.  FIGURE — left 78% animation, right 22% label panel
# ═══════════════════════════════════════════════════════════════
BG = "#03050f"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": "#c8e6ff", "axes.edgecolor": "#0e2a3d",
    "xtick.color": "#2a5a70", "ytick.color": "#2a5a70",
    "xtick.labelsize": 7,     "ytick.labelsize": 7,
})

fig = plt.figure(figsize=(15, 10))
fig.suptitle("Artemis II  —  Free-Return Mission",
             fontsize=14, fontweight="bold", color="#7ecfff",
             fontfamily="monospace", y=0.984)

gs  = gridspec.GridSpec(1, 2, width_ratios=[3.5, 1],
                        left=0.04, right=0.98,
                        top=0.95, bottom=0.05, wspace=0.02)
ax  = fig.add_subplot(gs[0])
axL = fig.add_subplot(gs[1])

ax.set_aspect("equal")
ax.set_xlabel("X  (10³ km)", fontsize=8, color="#2a5a70")
ax.set_ylabel("Y  (10³ km)", fontsize=8, color="#2a5a70")

# Axis limits
pad  = 25_000 / M
alx  = np.concatenate([all_x, moon_xa])
aly  = np.concatenate([all_y, moon_ya])
ax.set_xlim(alx.min()-pad, alx.max()+pad)
ax.set_ylim(aly.min()-pad, aly.max()+pad)
xlim, ylim = ax.get_xlim(), ax.get_ylim()
span = xlim[1] - xlim[0]

# Starfield
rng = np.random.default_rng(42)
ax.scatter(rng.uniform(*xlim, 400), rng.uniform(*ylim, 400),
           s=rng.uniform(0.2, 2.2, 400), c="white",
           alpha=0.28, zorder=0, linewidths=0)

# Ghost path
for ph, (_, col) in PHASES.items():
    mask = all_ph == ph
    if mask.any():
        ax.plot(all_x[mask], all_y[mask], color=col,
                alpha=0.07, lw=0.7, zorder=1)

# Moon orbit ring
th = np.linspace(0, 2*np.pi, 400)
ax.plot(D_EM/M*np.cos(th), D_EM/M*np.sin(th),
        color="#1a3a50", lw=0.6, ls="--", alpha=0.3, zorder=1)

# Display radii
RE_D = max(R_E/M, span * 0.022)
RM_D = max(R_M/M, span * 0.011)

# Earth
ax.add_patch(plt.Circle((0,0), RE_D*2.5, color="#1a5cbf", alpha=0.08, zorder=2))
ax.add_patch(plt.Circle((0,0), RE_D,     color="#1565c0", ec="#4a9fff", lw=1.8, zorder=3))
ax.text(0, -RE_D*3.8, "EARTH", color="#7ecfff", fontsize=10,
        ha="center", va="top", fontfamily="monospace", fontweight="bold", zorder=4)

# Moon (animated)
moon_patch = plt.Circle((moon_xa[0], moon_ya[0]), RM_D,
                         color="#263238", ec="#607080", lw=1.2, zorder=3)
ax.add_patch(moon_patch)
moon_lbl = ax.text(moon_xa[0], moon_ya[0] + RM_D*2.8, "MOON",
                   color="#90a4ae", fontsize=10, ha="center", va="bottom",
                   fontfamily="monospace", fontweight="bold", zorder=4)

# Trail + spacecraft
TAIL = 400
trail_lines = {ph: ax.plot([], [], lw=2.0, solid_capstyle="round",
                             color=PHASES[ph][1], alpha=0.92, zorder=5)[0]
               for ph in PHASES}

craft, = ax.plot([], [], "^", color="white", ms=11, zorder=9,
                  path_effects=[pe.withStroke(linewidth=3,
                                              foreground=PHASES[0][1])])
glow, = ax.plot([], [], "o", ms=26, color=PHASES[0][1], alpha=0.18, zorder=8)

craft_lbl = ax.text(0, 0, "Orion", color="white", fontsize=8.5,
                     fontfamily="monospace", fontweight="bold",
                     va="bottom", ha="left", zorder=10,
                     path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

# HUD
hkw = dict(transform=ax.transAxes, fontsize=8.5, fontfamily="monospace",
           va="bottom",
           path_effects=[pe.withStroke(linewidth=2, foreground=BG)])
t_day = ax.text(0.02, 0.13, "", color="#546e7a", **hkw)
t_alt = ax.text(0.02, 0.08, "", color="#80cbc4", **hkw)
t_spd = ax.text(0.02, 0.03, "", color="#80cbc4", **hkw)

# ═══════════════════════════════════════════════════════════════
# 5.  LABEL PANEL  (completely isolated from animation)
# ═══════════════════════════════════════════════════════════════
axL.set_facecolor("#040c18")
axL.set_xlim(0, 1)
axL.set_ylim(0, 1)
axL.axis("off")

# Border
for spine in ["top","bottom","left","right"]:
    axL.spines[spine].set_visible(False)
axL.add_patch(mpatches.FancyBboxPatch((0,0), 1, 1,
              boxstyle="round,pad=0.01",
              fc="#040c18", ec="#0e2a3d", lw=1.5,
              transform=axL.transAxes, zorder=0))

axL.text(0.5, 0.96, "MISSION PHASES",
         color="#7ecfff", fontsize=10, fontweight="bold",
         fontfamily="monospace", ha="center", va="top",
         transform=axL.transAxes)

axL.axhline(0.89, color="#0e2a3d", lw=1.2, xmin=0.05, xmax=0.95)

# 6 phase rows, evenly spaced between y=0.85 and y=0.08
phase_ys = np.linspace(0.82, 0.10, 6)
phase_artists = []

for idx, (ph, (name, col)) in enumerate(PHASES.items()):
    y = phase_ys[idx]

    # Colour bar on left
    bar = mpatches.FancyBboxPatch((0.04, y-0.04), 0.06, 0.07,
                                   boxstyle="round,pad=0.005",
                                   fc=col, ec="none", alpha=0.20,
                                   transform=axL.transAxes, zorder=2)
    axL.add_patch(bar)

    # Active indicator dot
    dot, = axL.plot([0.07], [y], "o", ms=7, color=col, alpha=0.0,
                    transform=axL.transAxes, zorder=4)

    # Phase name text
    txt = axL.text(0.16, y, name, color=col, fontsize=9,
                   fontweight="bold", fontfamily="monospace",
                   ha="left", va="center", alpha=0.28,
                   transform=axL.transAxes)

    phase_artists.append((ph, txt, dot, bar))

# Controls hint
axL.text(0.5, 0.03,
         "SPACE pause  R restart\n+/− speed   Q quit",
         color="#1a4060", fontsize=7, fontfamily="monospace",
         ha="center", va="bottom", linespacing=1.5,
         transform=axL.transAxes)

# ═══════════════════════════════════════════════════════════════
# 6.  PLAYBACK
# ═══════════════════════════════════════════════════════════════
state = {"frame": 0, "playing": True, "speed": 6}

def on_key(event):
    k = event.key
    if k == " ":               state["playing"] = not state["playing"]
    elif k in ("r","R"):       state["frame"] = 0; state["playing"] = True
    elif k in ("q","Q","escape"): plt.close(fig)
    elif k in ("+","="):       state["speed"] = min(state["speed"]+2, 50)
    elif k == "-":              state["speed"] = max(state["speed"]-2, 1)

fig.canvas.mpl_connect("key_press_event", on_key)

def update(_):
    if state["playing"]:
        state["frame"] = (state["frame"] + state["speed"]) % N_TOT
    fi  = state["frame"]
    ox  = all_x[fi];  oy  = all_y[fi]
    ph  = int(all_ph[fi])

    # Moon
    mxi, myi = moon_xa[fi], moon_ya[fi]
    moon_patch.set_center((mxi, myi))
    moon_lbl.set_position((mxi, myi + RM_D*2.8))

    # Trail
    s0 = max(0, fi - TAIL)
    for p in PHASES:
        xs, ys = [], []
        for j in range(s0, fi):
            if all_ph[j] == p:
                j1 = min(j+1, N_TOT-1)
                xs += [all_x[j], all_x[j1], np.nan]
                ys += [all_y[j], all_y[j1], np.nan]
        trail_lines[p].set_data(xs, ys)

    # Spacecraft
    if fi > 0:
        ddx = all_x[fi] - all_x[fi-1]
        ddy = all_y[fi] - all_y[fi-1]
        angle = np.degrees(np.arctan2(ddy, ddx)) - 90
        craft.set_marker((3, 0, angle))
        craft.set_markeredgecolor(PHASES[ph][1])

    craft.set_data([ox], [oy])
    glow.set_data([ox], [oy])
    glow.set_color(PHASES[ph][1])

    la = np.arctan2(oy, ox) if (ox != 0 or oy != 0) else np.pi/4
    craft_lbl.set_position((ox + RE_D*0.6*np.cos(la+0.5),
                             oy + RE_D*0.6*np.sin(la+0.5)))

    # Phase panel
    for p, txt, dot, bar in phase_artists:
        active = (p == ph)
        txt.set_alpha(1.0  if active else 0.28)
        dot.set_alpha(0.95 if active else 0.0)
        bar.set_alpha(0.55 if active else 0.15)

    # HUD
    re   = np.hypot(ox, oy) * M
    alt  = (re - R_E) / 1e3
    spd  = np.hypot(all_vx[fi], all_vy[fi]) / 1e3
    orb_dur = 2.0 * 2*np.pi*R_PARK / V_PARK / 86400
    day  = fi / N_TOT * (orb_dur + DAYS)
    t_day.set_text(f"Day   : {day:5.2f}")
    t_alt.set_text(f"Alt   : {alt:>10,.0f} km")
    t_spd.set_text(f"Speed : {spd:>10.2f} km/s")

    return (
        *trail_lines.values(),
        craft, glow, craft_lbl,
        moon_patch, moon_lbl,
        t_day, t_alt, t_spd,
        *[dot for _, _, dot, _ in phase_artists],
        *[txt for _, txt, _, _ in phase_artists],
    )

ani = FuncAnimation(fig, update,
                    frames=N_TOT // max(state["speed"], 1),
                    interval=30, blit=True)

print("Space = pause/resume  |  R = restart  |  +/- = speed  |  Q = quit")
plt.show()