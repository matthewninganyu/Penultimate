# C3-B — 4-Point Plane Homography
> Candidate B (sibling: ../A_triangulation). Parent: ../../PROJECT.md | Owners: JAM + DAN | Updated: 2026-07-19
> Pick A or B after the bench bake-off (both share the detection front-end + calibration capture).

## How a homography works (primer — B uses this)
The screen surface is a flat plane. The camera image is a flat plane. For a pinhole camera, any
point that lies **on** the screen plane maps to its image pixel through a single fixed 3×3 projective
matrix **H** — the *homography*. This holds for **any camera pose** (our few-mm-high, down-tilted,
toed-in mount included) — that's the whole reason B beats triangulation here: it doesn't care that
the cameras aren't in-plane. Our LED is kept on the plane by the tip switch, so the mapping is exact.

- Solve H from **≥4 known correspondences**: tap ≥4 screen points, record their image pixels →
  `cv2.findHomography(image_pts, screen_pts)`. 4 is the minimum; we use the **whole tap grid** so
  RANSAC/least-squares averages noise and distortion residual.
- Runtime = one matrix-multiply per camera: `(u,v,1)·H → (x,y,w) → (x/w, y/w)`. Full (x,y) from a
  **single** camera → two cameras are redundancy, not a requirement (single-cam fallback = B's edge over A).

**Critical caveat:** a homography is *linear projective* — it does **NOT** model lens distortion.
The frames show heavy barrel distortion, so B **MUST undistort (u,v) before applying H** (see R1).

## Goal
Recover pen (x,y) via a per-camera plane homography `image(u,v) → screen(x,y)`, fit from the tap
grid, combined across two cameras. Handles the out-of-plane mount exactly.

## Camera reality (from the 3 sample frames — design tailored to this)
- Top corners, **few mm above plane, slight down-tilt, toed-in**. Screen = **bottom ~third** of frame.
- **Blue LED**, white-hot core + blue halo. Reflection = **blue streak running DOWN** from contact.
- **Heavy barrel distortion** — L/R screen extremes at the horizontal frame edges = worst distortion
  (**R1**). B is *more* exposed to this than A (H can't absorb distortion) → undistort quality at the
  edges is load-bearing.
- Overhead glare = **white/low-sat** → blue-hue rejected. Black letterbox bars top+bottom.

## Interface (from parent)
- **In:** `Detection` per camera `{cam_id, u, v, timestamp_us, confidence}` | None. B uses **full (u,v)**.
- **Out:** `ScreenPoint {x_norm∈[0,1], y_norm∈[0,1], pen_down, timestamp_us}`.
- **Artifacts:** `intrinsics_cam{0,1}.npz` (undistort), `homography_cam{0,1}.npy` (H, image→screen),
  `screen_config.yaml`, screen-region mask per camera.
- `pen_down` = blob present.

## Detection front-end (SHARED with A — BUILT: ../detect.py, validated on 4 real frames)
Discriminator = **blueness×brightness peak** (not literal "topmost"): the LED contact is the most
intense blue AND brightest point. The peak-containing blob rejects the **blue pen barrel** (above,
dimmer), the **reflection streak** (below, dimmer), and **white glare** (neutral). Steps: score
`(B−R)·V` blurred → peak-floor gate (pen up/down) → peak-containing blob → score-weighted centroid.
B keeps **both u and v**. `../tests/test_detect.py` passes.
- **Resolution-relative** (sigma ∝ width, floor = normalized peak) → auto-scales across 1852×1422
  fixtures / 720p runtime / full-res, no per-mode tuning. Verified <0.04% normalized centroid drift.
- C2/GAY integration still pending (mask-from-calibration, streaming).

## Calibration (once, with the REAL stylus so tip-height offset bakes in)
1. **Intrinsics / undistort (R1 gate — do before homography):** checkerboard waved to the frame
   edges → `cv2.calibrateCamera`; **run pinhole AND `cv2.fisheye`, keep whichever has lower
   straight-line residual at the L/R frame edges**; RMS reprojection < 1px. Barrel distortion is
   large (~130px @ crop) — an unmodeled edge is a direct (x,y) error B cannot recover.
2. **Homography:** tap a **dense grid to the screen edges** with the real stylus; undistort each
   chosen-blob pixel; `findHomography(undistorted_image_pts, screen_pts)` per camera (grid + RANSAC,
   not just 4). Store H image→screen so runtime needs no inversion.
- **Screen active area = 30.4 × 19.7 cm** (3024×1964 px, 14.2" diag) → screen_pts space for findHomography.
  **Capture = 720p (1280×720) @ 100 FPS.** Intrinsics/H valid ONLY in this mode; recalibrate if changed.

## Position solver
- `apply_homography(H, u, v) → (x,y)` after `undistortPoints`.
- `combine(est0, conf0, est1, conf1) → (x,y)`: confidence/conditioning-weighted average of the two
  per-camera estimates; downweight the camera whose view is most foreshortened / grazing there.
- `solve(det0, det1) → ScreenPoint`: undistort→H each present camera, combine.
  **One camera present ⇒ still outputs (x,y)** (degraded) — no hard dropout (B's advantage over A).
  `pen_down` = blob present. Δt guard: skip/flag if inter-cam Δ too large.

## Testing
### Synthetic (no rig — do FIRST, fast CI): `test_position.py`
- Project a known screen grid through a synthetic pose **+ injected barrel distortion** (jig CM3-Wide
  coeffs, scaled to crop res) → undistort → fit H on the grid → `solve` on held-out points.
- Assert recovery within budget **with distortion present**; a **no-undistort baseline must FAIL**
  (proves the undistort step earns its place and the test has teeth).
- Sweep undistort quality (inject residual distortion) → map how edge error grows → quantify R1 risk.
- Single-camera path recovers (x,y) within (degraded) budget.
### On rig (after calibration)
1. **Static reprojection residual (ground truth):** held-out taps, RMS+max **per region**.
2. **Centroid stability:** static stylus → centroid & (x,y) jitter.
3. **Sync verification:** stylus at known speed across a straight edge → desync artifact.
4. **Blob visibility + single-cam dropout:** blue blob at farthest reach; confirm single-cam path
   still outputs a usable point (B should degrade, not die).
5. **Reflection-reject correctness:** topmost-blob pick selects contact not the down-streak.
6. **Undistort straight-line check (R1):** real straight edge at the L/R frame extreme stays straight.

## TODO
- [ ] `position.py`: `apply_homography` (+undistort), `combine`, `solve` (single-cam fallback, Δt guard).
- [ ] `test_position.py`: synthetic w/ injected distortion; no-undistort baseline FAILS; single-cam path; R1 sweep.
- [ ] Detection front-end w/ C2: mask-top, blue-HSV, highest-blob, (u,v)-centroid. Validate on sample frames.
- [x] Confirm screen dims + capture mode (DAN/C1) — 30.4×19.7cm active, 720p@100FPS.
- [ ] R1 gate: checkerboard-to-edges; pinhole vs fisheye, keep lower edge residual; RMS<1px → `intrinsics_camN.npz`.
- [ ] `calibrate.py`: tap grid to edges → undistort → `findHomography` (grid+RANSAC) → `homography_camN.npy` + mask + config.
- [ ] On-rig tests 1–6 within budget.

## Key Decisions
- [2026-07-19] **Blue LED + blue HSV** (green plan void) — white glare hue-rejected. [→ C5/C6]
- [2026-07-19] Reflection/barrel-reject via **blueness×brightness peak** (LED core = brightest blue).
  Replaced literal "topmost" — the blue pen barrel sits ABOVE the contact, so topmost picks the barrel.
- [2026-07-19] **Undistort is mandatory before H** (H can't model the large barrel distortion); pinhole
  vs fisheye chosen by L/R-edge residual.
- [2026-07-19] H stored **image→screen** (no runtime inversion); fit over the full grid, not just 4 points.
- [2026-07-19] **Screen active area 30.4×19.7 cm** (3024×1964, 14.2" MBP) — glass tap area = screen_pts space, NOT chassis.
- [2026-07-19] **Capture 720p@100FPS** — barrel ~130px scales to this crop res; sync 10ms/frame for moving stylus.

## Notes
- B's strengths vs A: uses full (u,v) → handles out-of-plane mount exactly; single-camera fallback;
  no per-camera azimuth-bias term.
- B's weakness vs A: fully exposed to lens distortion → undistort quality at the frame edges (R1) is
  the dominant risk; needs the checkerboard intrinsics step A skips.
- Intrinsics/H are crop-mode specific; recalibrate if C1 changes mode or the rig is bumped.

## Build result [2026-07-19]
Synthetic solver PASS 7/7. Distortion handled (~2e-5mm exact-centroid); no-undistort baseline 2.56mm (teeth ok); single-cam fallback ok; R1 edge/int ~1.3x.
**Bake-off finding [BLOCKING for B]:** the ~2deg grazing mount makes the plane-homography Jacobian ill-conditioned. Geometrically exact, BUT +/-0.3px centroid noise -> 2.4mm mean / 7mm max (median Jacobian cond 40) = blows budget. B needs a LESS-GRAZING/higher vantage or sub-0.05px centroids to hit sub-mm. Decide before committing to B.
