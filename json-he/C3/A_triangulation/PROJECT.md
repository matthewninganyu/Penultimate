# C3-A — Classic Two-Bearing Triangulation
> Candidate A (sibling: ../B_homography). Parent: ../../PROJECT.md | Owners: JAM + DAN | Updated: 2026-07-19
> Pick A or B after the bench bake-off (both share the detection front-end + calibration capture).

## Goal
Recover pen (x,y) on the screen by intersecting two **azimuth bearings**, one per top-corner
camera. Collapse each camera to a single horizontal coordinate `u` (flatten v), map `u → bearing`,
cast from the known corner, intersect on the 2D screen plane.

## Camera reality (from the 3 sample frames — design tailored to this)
- Mounted top corners, **few mm above the screen plane, slight down-tilt, toed-in** toward center.
  NOT in-plane (couldn't get flush). Screen fills the **bottom ~third** of the frame.
- **Blue LED**, saturates to a white core with blue halo. Reflection appears as a **blue streak
  running DOWN** from the contact point (glossy screen).
- **Heavy barrel distortion** — straight edges curve. Screen L/R extremes sit near the horizontal
  frame edges = worst distortion (**R1**).
- Overhead-light glare on screen = **white / low-saturation** → rejected by the blue-hue gate.
- Black letterbox bars top+bottom (crop mode).

## Honest caveat (JAM's accepted tradeoff)
Cameras are out-of-plane, so `u` is not a pure azimuth — the on-plane LED's `u` has a small,
spatially-varying dependence on screen depth (a **systematic bias**, not noise, worst toward edges).
A accepts this: the per-camera `u→bearing` fit is calibrated against **real screen taps**, absorbing
most of it; residual is measured in test 1. If residual blows the ≤5mm edge budget, fall to B.

## Interface (from parent)
- **In:** `Detection` per camera `{cam_id, u, v, timestamp_us, confidence}` | None. A uses **u only**
  (v used solely to pick the topmost blob).
- **Out:** `ScreenPoint {x_norm∈[0,1], y_norm∈[0,1], pen_down, timestamp_us}`.
- **Artifacts:** `bearing_map_cam{0,1}.npz` (1-D u→bearing fit), `screen_config.yaml`
  (screen dims, corner positions, baseline), screen-region mask per camera.
- `pen_down` = blob present (LED lit only on tip contact).

## Detection front-end (SHARED with B — BUILT: ../detect.py, validated on 4 real frames)
Discriminator = **blueness×brightness peak** (not literal "topmost"): the LED contact is the most
intense blue AND brightest point. Selecting the blob containing that peak rejects the **blue pen
barrel** (above the tip, dimmer), the **reflection streak** (below, dimmer), and **white glare**
(neutral). Steps: score `(B−R)·V` blurred → peak-floor gate (pen up/down) → peak-containing blob →
score-weighted centroid. A-specific: **flatten to u**. `../tests/test_detect.py` passes.
- **Resolution-relative** (sigma ∝ width, floor = normalized peak) → auto-scales across 1852×1422
  fixtures / 720p runtime / full-res, no per-mode tuning. Verified <0.04% normalized centroid drift.
- C2/GAY integration still pending (mask-from-calibration, streaming).

## Calibration (once, with the REAL stylus so tip-height offset bakes in)
- **`u → bearing` per camera, 1-D polynomial fit from the tap grid.** No checkerboard/intrinsics
  needed: the barrel distortion along the horizontal axis is a smooth monotonic warp; a flexible
  1-D fit absorbs it *and* the toe-in, directly from known screen taps. (This sidesteps the
  fisheye-vs-pinhole fight for the u-axis — a real simplification A buys over B.)
- Tap a **dense grid to the screen edges** (edges are the accuracy-critical, worst-distortion zone).
  Record `(chosen-blob-u → known screen x,y)` per camera.
- Write corner positions + baseline (= screen width) to `screen_config.yaml`.
- **Screen active area = 30.4 × 19.7 cm** (3024×1964 px, 14.2" diag). Baseline (A) = 30.4 cm width.
  **Capture = 720p (1280×720) @ 100 FPS.** Bearing map valid ONLY in this mode; recalibrate if changed.

## Position solver
- `pixel_to_bearing(map, u) → angle` (apply 1-D fit).
- `intersect(bearing0, corner0, bearing1, corner1) → (x,y)`; flag near-collinear (small sin γ) at
  the top strip → raise uncertainty / downweight.
- `solve(det0, det1) → ScreenPoint`: topmost-blob u each, map to bearings, intersect, normalize.
  Single camera present ⇒ **no intersection possible → dropout** (A's structural weakness vs B).
  `pen_down` = blob present. Δt guard: skip/flag if inter-cam Δ too large (C1 targets <8ms).

## Testing
### Synthetic (no rig — do FIRST, fast CI): `test_position.py`
- Two-bearing intersection over a screen grid, inject **±0.3px u-noise** AND a **small v→u coupling
  term** modeling the out-of-plane tilt → error map. Assert interior budget + bounded top-edge
  degradation; **quantify the tilt-bias term** so we know A's ceiling before hardware.
### On rig (after calibration)
1. **Static reprojection residual (ground truth):** tap held-out points, RMS + max **per region**
   (center/bottom/top-strip/corners) vs GDOP prediction.
2. **Centroid/bearing stability:** static stylus, centroid jitter (px) → (x,y) jitter (mm); confirms ±0.3px.
3. **Sync verification:** move stylus at known speed across a straight edge; check desync lag/lead.
4. **Blob visibility + single-cam dropout:** confirm blue blob detected at farthest reach from each
   camera (bottom-far, dim/foreshortened); confirm dropout (one cam loses blob) fails gracefully.
5. **Reflection-reject correctness:** verify topmost-blob pick always selects contact, never the
   down-streak, across the screen (use the sample-frame geometry as fixtures).

## TODO
- [ ] `position.py`: 1-D `pixel_to_bearing`, `intersect`, `solve` (dropout-safe, near-collinear flag).
- [ ] `test_position.py`: synthetic + tilt-bias term; assert budget + measure ceiling.
- [ ] Detection front-end w/ C2: mask-top, blue-HSV, highest-blob, u-centroid. Validate on sample frames.
- [x] Confirm screen dims + capture mode (DAN/C1) — 30.4×19.7cm active, 720p@100FPS.
- [ ] `calibrate.py`: tap grid to edges → 1-D `u→bearing` fit → `bearing_map_camN.npz` + mask + config.
- [ ] On-rig tests 1–5 within budget.

## Key Decisions
- [2026-07-19] **Blue LED + blue HSV** (green plan void) — white glare hue-rejected. [→ C5/C6]
- [2026-07-19] Reflection/barrel-reject via **blueness×brightness peak** (LED core = brightest blue).
  Replaced literal "topmost" — the blue pen barrel sits ABOVE the contact, so topmost picks the barrel.
- [2026-07-19] **1-D u→bearing poly from taps** — absorbs horizontal distortion + toe-in, no checkerboard.
- [2026-07-19] Accept out-of-plane u-bias as A's known ceiling; measure it; fall to B if it breaks edge budget.
- [2026-07-19] **Screen active area 30.4×19.7 cm** (3024×1964, 14.2" MBP) — glass tap area, NOT chassis; baseline=30.4cm.
- [2026-07-19] **Capture 720p@100FPS** — balance u-resolution vs inter-cam sync (10ms/frame) for moving stylus.

## Notes
- A's weaknesses vs B: needs BOTH cameras (no single-cam fallback); carries the out-of-plane u-bias.
- A's strengths vs B: no intrinsics/undistort, geometric & interpretable, 1-D calibration.
- Mask/bearing map are crop-mode specific; recalibrate if C1 changes mode or the rig is bumped.

## Build result [2026-07-19]
Synthetic solver PASS. interior 0.35mm / edge 0.23mm / top-strip 0.50mm (bounded); tilt-bias ceiling 0.36mm.
**Bake-off finding:** at the ~2deg grazing mount actually built, A (u-only bearing) is noise-ROBUST (+/-0.3px -> ~0.35mm). Sibling B is not (see B doc). A favored for the current rig.
