"""Graph dual_log.csv: per-camera u/v jitter + inter-camera sync (dt).

    python3 plot_jitter.py [dual_log.csv]

Saves jitter.png and prints per-camera u/v std-dev + dt stats. Std-dev is the
jitter only when the stylus is held STILL. u std is the number that matters most
(u drives triangulation). Runs headless (saves PNG).
"""
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "dual_log.csv"
rows = list(csv.DictReader(open(path)))
if not rows:
    sys.exit("empty log")


def col(rs, key):
    return np.array([float(r[key]) for r in rs if r[key] != ""])


fig, ax = plt.subplots(2, 2, figsize=(13, 9))

for cam in (0, 1):
    rs = [r for r in rows if r[f"u{cam}"] != ""]
    fr = np.array([float(r["frame"]) for r in rs])
    u, v = col(rs, f"u{cam}"), col(rs, f"v{cam}")
    ax[0][0].plot(fr, u, ".-", ms=3, label=f"cam{cam} u")
    ax[0][1].plot(fr, v, ".-", ms=3, label=f"cam{cam} v")
    ax[1][0].scatter(u, v, s=6, label=f"cam{cam}")
    drop = sum(1 for r in rows if r[f"u{cam}"] == "")
    print(f"cam{cam}: n={len(u)} dropouts={drop} "
          f"u_std={u.std():.2f}px v_std={v.std():.2f}px "
          f"u_range={np.ptp(u):.1f} v_range={np.ptp(v):.1f}")

ax[0][0].set_title("u vs frame (u = the one that matters)")
ax[0][0].legend()
ax[0][1].set_title("v vs frame")
ax[0][1].legend()
ax[1][0].set_title("u-v scatter per camera")
ax[1][0].invert_yaxis()
ax[1][0].legend()

dt = col(rows, "dt_us")
acc = col(rows, "accept")
ax[1][1].hist(dt, bins=40)
ax[1][1].set_title(f"inter-cam dt (us)  median={np.median(dt):.0f} "
                   f"accept={100 * acc.mean():.0f}%")
ax[1][1].set_xlabel("dt_us")
print(f"dt: median={np.median(dt):.0f}us max={dt.max():.0f}us "
      f"accept_rate={100 * acc.mean():.0f}%")

plt.tight_layout()
plt.savefig("jitter.png", dpi=110)
print("saved jitter.png")
