# C3 — Calibration + Position
> Parent: ../PROJECT.md | Owners: JAM + DAN (DAN = Pi/microSD/hardware) | Updated: 2026-07-18

## Goal
Recover the pen's 2D screen position by **two-bearing intersection** from two in-plane
top-corner cameras, and produce the calibration artifacts that make the bearing map exact.
Target sub-mm interior, ≤~0.7mm top-edge strip. Runs anywhere on synthetic geometry; only
calibration *capture* needs the real rig.

## Interface (fixed — from parent PROJECT)
- **In (runtime):** `Detection` per camera = `{cam_id, u, v, timestamp_us, confidence}` | None.
  Only **u** (horizontal centroid) carries screen info in this geometry; v is not relied upon.
- **Out:** `ScreenPoint = {x_norm∈[0,1], y_norm∈[0,1], pen_down: bool, timestamp_us}`.
- **Artifacts:** `intrinsics_cam{0,1}.npz`, `bearing_map_cam{0,1}.npz` (pixel→angle + fit),
  `screen_config.yaml` (screen px dims, corner positions, baseline). Screen-region mask is
  derived from the tap grid (software, not physical tape).
- `pen_down` = blob present (LED lit only on tip contact via series tip-switch) — no separate signal.

## Approach — locked
**Mount:** two CM3 Wide **in the plane of the screen surface**, both top corners, **max baseline
= screen width**, each sightline **grazing/edge-on** across the flat screen. Deliberate: in-plane
presents near-zero reflective area → **structurally eliminates screen glare**. Consequences:
- Glare/frame-differencing/**modulation demoted to backup**; **continuous superbright-LED + tip-switch
  is primary** (HSV threshold on the LED's saturated hue).
- Aim so **screen width spans as much of the horizontal sensor axis as possible** (max pixels-per-mm
  on the u/bearing axis). v axis carries little info here.
- Mask everything outside the calibrated screen region **in software**, not tape.

**Position — two-bearing intersection (NOT 3D triangulation, NOT homography):**
Per camera per frame: capture → HSV threshold inside masked screen region → largest-blob **sub-pixel
horizontal centroid (u)** → `u → bearing angle` via calibrated pixel→angle model → cast bearing from
that camera's known corner. **Intersect the two bearings on the 2D screen plane → (x,y).**

**Method A calibration (two stages, both captured once with the REAL stylus so tip-height offset bakes in):**
1. **Intrinsic/bearing cal** — checkerboard waved to frame edges (screen fills frame edge-to-edge here),
   `calibrateCamera`/`undistortPoints` → correct wide-lens barrel distortion → accurate pixel→bearing
   across full width.
2. **Tap-grid cal** — tap a dense grid of known screen points **out to the edges**, record `(blob-u →
   known x,y)` per camera → fit the final bearing/position map.

**Transport:** output (x,y) to laptop over **ethernet**. *(Deviation: parent PROJECT says USB-C
CDC-serial primary / WiFi-UDP fallback — reconcile in C4/C6.)*

## Known accuracy structure (drives what tests target)
With cameras **in sync** and glare removed by in-plane mounting, the two biggest general-case errors
(camera desync 5–20mm; glare-corrupted centroid) are **structurally zeroed**. Leaves:
- **Calibration residual = dominant real-world term** (biggest lever now).
- **GDOP = spatial modulator.** Interior well-conditioned; top edge / top corners near-collinear
  (sin γ → small) → weakest, intentionally the least-used writing area.
- Expected: **~0.2mm @1080p, ~0.08mm full-res interior; ~0.6–0.7mm top-edge strip.**
- **[CONFIRM screen dimensions]** and **[CONFIRM resolution/framerate mode]** — both scale these linearly.

**Load-bearing risks the tests must attack:**
- **R1 — best-accuracy geometry loads the bearing onto the lens's WORST distortion zone.** Screen L/R
  extremes sit at the horizontal frame edges = max barrel distortion (~130px @ 2304-crop), where the
  pinhole radial polynomial is least reliable. The bearing map's edge accuracy rides entirely on
  `undistortPoints` being good there → **gated by an explicit distortion test before tap-grid.**
- **R2 — grazing incidence INCREASES specular reflectance (Fresnel).** "Near-zero reflective area" is
  about screen *area*; the LED is a point source whose mirror image can still appear as a competing
  blob. Reflection-reject was dropped → **verify no secondary blob**; if present, restore an air-side
  reject rule.

## Architecture — C3/ files
- `C3/position.py` — solver + math, no I/O. `pixel_to_bearing(map,u)`, `intersect(b0,c0,b1,c1)→(x,y)`,
  `solve(det0,det1)→ScreenPoint`. Importable, runs on synthetic geometry.
- `C3/calibrate.py` — thin entry point. Stage-1 checkerboard → `intrinsics_camN.npz`; stage-2 tap grid
  (Mac displays dots, C2 supplies detections) → `bearing_map_camN.npz` + screen mask + `screen_config.yaml`.
- `C3/test_position.py` — synthetic two-bearing test (no rig, no downloads).

## Testing Strategy
### Gate before tap-grid — distortion isolation (R1)
- `calibrateCamera` **RMS reprojection < 1px** (checkerboard corners must reach frame edges).
- **Straight-edge residual at the L/R frame edge**: undistort a real straight edge at the horizontal
  extreme, measure max deviation from fitted line.
- Run **pinhole AND `cv2.fisheye`** on the same images; keep whichever has the lower edge residual.
- One-shot **R2 check**: lit stylus static at several screen positions, look for a secondary reflected blob.

### On-rig (after calibration)
1. **Static reprojection residual (ground-truth accuracy).** Tap held-out known points (not in cal grid);
   measure predicted-vs-actual (x,y). Report RMS + max **per region** (center, bottom, top-edge strip,
   corners) → confirm the GDOP map matches prediction.
2. **Bearing/centroid stability.** Hold lit stylus stationary at several points; measure centroid jitter
   (px) and resulting (x,y) jitter (mm). Confirms ±0.3px centroid assumption; exposes noise/exposure issues.
3. **Sync verification.** Move stylus at known speed across a straight edge; check for desync artifact
   (triangulated point lagging/leading between cameras). Confirms the "in sync" assumption the whole budget rests on.

### Added
4. **Synthetic two-bearing test (no rig, runs anywhere — do FIRST, fast CI).** Inject ±0.3px centroid noise
   into the intersection math across a screen grid → produce the (x,y) error map → **assert it matches the
   ~0.2mm interior / ~0.6–0.7mm top-edge prediction and the GDOP structure.** Validates intersection math +
   the paper error budget before any hardware. Run: `python C3/test_position.py`.
5. **Blob visibility + single-camera dropout across full screen.** Confirm the LED blob is detected with
   adequate size/SNR at the **farthest reach from each camera** (bottom-far corners — foreshortened, dimmest).
   Two-bearing needs BOTH cameras to see it → test where one camera loses the blob (hand occlusion, blob
   exits a FOV near a corner) and confirm **graceful dropout, not a garbage intersection.**

## Environment for best accuracy
- Dense tap-grid extending fully to screen edges — single biggest lever now that glare/desync are gone.
- Rigid, vibration-free mounts held exactly in-plane; **calibrate after final mounting, don't bump** (any
  post-cal flex directly corrupts the bearing map).
- Small, bright, well-focused LED → compact clean blob (superbright OK — in-plane removed the glare
  downside — but keep it **below sensor clipping**).
- Controlled ambient light, **no same-hue sources in view** (HSV gating needs the LED hue unique).
- Keep primary writing area in the well-conditioned interior, away from the top-edge GDOP strip.

## TODO (ordered)
### Solver + synthetic (no rig — do first)
- [ ] `position.py`: `Detection`/`ScreenPoint` shapes per contract; u-only bearing.
- [ ] `position.py`: `pixel_to_bearing(map, u)` — calibrated pixel→angle.
- [ ] `position.py`: `intersect(bearing0, corner0, bearing1, corner1) → (x,y)`; flag near-collinear (small sin γ).
- [ ] `position.py`: `solve(det0, det1) → ScreenPoint` — pen_down = blob present; single-cam ⇒ dropout, not garbage.
- [ ] `test_position.py` (test 4): synthetic grid, inject ±0.3px centroid noise → error map;
      assert interior ~0.2mm-equiv, top-edge ~0.6–0.7mm-equiv, GDOP structure matches.
- [ ] `test_position.py` passes: `python C3/test_position.py`.

### Calibration + distortion gate (needs rig / checkerboard — coordinate with DAN/C1/C2)
- [ ] **[BLOCKING]** Confirm capture res/framerate mode WITH C1 (DAN) — bearing map valid ONLY in that mode.
- [ ] **[BLOCKING]** Confirm screen dimensions — scales the whole accuracy budget.
- [ ] Distortion gate: `calibrateCamera` (checkerboard to edges) RMS < 1px; L/R-edge straight-line residual;
      pinhole vs `cv2.fisheye`, keep lower-edge-residual → `intrinsics_camN.npz`.
- [ ] R2 one-shot: check for secondary reflected LED blob; restore air-side reject rule if present.
- [ ] `calibrate.py`: full-screen dot grid on Mac (to edges), auto-advance on stable detection, reject noisy taps.
- [ ] `calibrate.py`: collect `(blob-u → known x,y)` per cam (real lit stylus) → fit → `bearing_map_camN.npz`.
- [ ] `calibrate.py`: derive software screen-region mask + write `screen_config.yaml` (dims, corners, baseline).
- [ ] On-rig test 1 (held-out residual, per-region RMS/max) within budget.
- [ ] On-rig test 2 (centroid + (x,y) jitter) within ±0.3px assumption.
- [ ] On-rig test 3 (sync verification) — no desync artifact at speed.
- [ ] On-rig test 5 (blob visibility + single-cam dropout) across full screen incl. far corners.

## Done
- [2026-07-18] C3 planned; CM3-Wide distortion research captured; design locked to in-plane mount,
  two-bearing intersection, Method A calibration.

## Open Questions
- **[CONFIRM] screen dimensions** — scales accuracy budget. [BLOCKING synthetic-test numbers]
- **[CONFIRM] capture resolution/framerate mode** (from C1/DAN) — bearing map valid only in that mode. [BLOCKING]
- Pinhole vs `cv2.fisheye` at the horizontal frame edges — resolve empirically in the distortion gate. [R1]
- Does the LED produce a secondary reflected blob at grazing? — resolve in R2 one-shot. [R2]
- Ethernet transport vs parent's USB-C serial — reconcile in C4/C6.

## Key Decisions
- [2026-07-19] **COMMITTED TO A (two-bearing triangulation). B (homography) dropped** — at the ~2° grazing
  mount B's homography Jacobian is ill-conditioned: ±0.3px centroid noise → 2.4mm mean / 7mm max (blows
  the ≤5mm budget); A same-noise ~0.35mm. B archived to `archive/B_homography/` (recoverable if A fails on
  the real rig). Shared harness stripped of B-only checkerboard/intrinsics capture.
- [2026-07-18] **In-plane grazing mount** — presents near-zero reflective area, structurally eliminates
  glare; demotes modulation to backup, makes continuous LED + tip-switch primary.
- [2026-07-18] **Two-bearing intersection (u-only), Method A** — undistort → pixel→bearing → cast from
  known corners → intersect on 2D plane; v axis not relied upon.
- [2026-07-18] **Continuous superbright LED + HSV threshold primary**, modulation backup.
- [2026-07-18] Software screen mask, not physical tape.
- [2026-07-18] Error budget: desync + glare structurally zeroed → calibration residual dominant, GDOP
  spatial modulator; top-edge strip is the intentional weak zone.

## Notes
- Intrinsics/maps are **crop-mode specific** — recalibrate if C1 changes capture mode.
- Tap grid + checkerboard MUST reach the frame/screen edges — the accuracy-critical bearing zone is
  exactly the horizontal frame edges (R1).
- Calibrate with the ACTUAL stylus (tip-height offset bakes in) AND after final rigid mounting.
- Sources: jig/charuco-calibration (CM3-Wide coeffs, sanity baseline); picamera2 #630 + imx708_wide.json
  (libcamera applies no geometric correction); OpenCV fisheye docs; ROS camera_calibration (straight-line
  test); reprojection thresholds = community convention.
