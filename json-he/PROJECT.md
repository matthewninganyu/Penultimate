# Camera-triangulated stylus — turn any screen into a tablet
> Updated: 2026-07-18 | Demo target: 36h hackathon, ~$100, MacBook (Windows if time)
> Transport: direct Pi↔Mac ethernet (UDP); USB-C now power-only

## Goal
Two top-corner Raspberry Pi Camera Module 3 cameras track a blinking visible-LED
stylus at grazing angle across a non-touchscreen laptop display, map its position
to screen pixels, and drive the OS cursor for seamless drawing/notes. Optimize for
a working, accurate demo — not production. Accuracy is the selling point
(target ≤2mm center, ≤5mm edges).

## Stack
- OpenCV: `calibrateCamera`/`undistortPoints` (intrinsics), `findHomography` (plane map), blob/centroid
- NumPy: geometry, least-squares position solve
- Transport: direct Pi↔Mac ethernet UDP (Pi 5 gigabit + USB-C→ethernet dongle on Mac). Power via separate USB-C 5V/3A

## Approach (locked decisions)
- Mount: **adjacent top corners**, max baseline, slight down-tilt. Degenerate ray zone
  falls on the unused top edge; center stays well-conditioned.
- Detection primary: **temporal modulation** — LED free-runs blinking at fixed freq;
  detect by its blink signature (rejects static glare AND live-screen content); no
  hardware sync. Backup swap: HSV-green + brightness threshold.
- Reflection reject: pick blinking blob furthest from the screen-surface line (air-side).
- Position: **undistort + dual-homography least-squares**. Per camera: `undistortPoints`
  then screen→cam homography; solve (x,y) over both cameras. No extrinsics, no metric triangulation.
- Pen-down: LED only powered when tip switch closed (contact) → detection present = pen down.
  Also enforces the on-plane assumption that makes the homography exact.
- Green LED + matching bandpass/gel filter on each lens (extra glare margin).
- One-Euro filter on output for smooth cursor.

## Architecture — components + interfaces
Independent components. Each has a fixed I/O contract and is testable alone with a
fake/synthetic input. Build against the contract, not the other person's code.

**Data contracts (the interfaces between people):**
- `FrameBundle`  = { cam_id, frame: ndarray, timestamp_us, frame_index }  — Capture emits a
  short rolling buffer per camera (modulation needs recent N frames).
- `Detection`    = { cam_id, u, v, timestamp_us, confidence } | None       — one per camera.
- `ScreenPoint`  = { x_norm∈[0,1], y_norm∈[0,1], pen_down: bool, timestamp_us }.
- Calibration artifacts (shared files): `intrinsics_cam{0,1}.npz`, `homography_cam{0,1}.npy`,
  `screen_config.yaml`, `surface_line_cam{0,1}` (for reflection reject).

**Components:**
- **C1 Capture** [DAN] — dual CM3 on Pi 5, cropped high-fps mode, monotonic per-frame
  timestamps, rolling frame buffer. Emits `FrameBundle`. Owns picamera2 config only.
- **C2 Detection** [GAY] — `FrameBundle` buffer → `Detection` per camera. Modulation (primary)
  / threshold (backup) as a swappable frame-source stage; subpixel centroid; reflection reject.
  Runs anywhere on synthetic frames.
- **C3 Calibration+Position** [UNASSIGNED] — intrinsic + homography calibration scripts (produce
  the shared artifacts) and runtime solver: `(u0,v0,u1,v1)+timestamps → ScreenPoint`. Includes
  timestamp interpolation to a common time. Runs anywhere on synthetic geometry.
- **C4 Output** [LIF] — `ScreenPoint` stream → OS cursor move + click (Quartz working) + One-Euro
  smoothing. Owns transport receive (serial/UDP). Runs on Mac.
- **C5 Stylus HW** [UNASSIGNED] — green LED + tip switch + blink driver. Deliverable = physical pen.
- **C6 Rig+Integration** [SHARED] — rigid top-corner camera bar, wiring, transport link Pi↔Mac,
  end-to-end glue script, config. Rig must stay rigid (homography valid only while cams+lid fixed).

## Testing Strategy (black-box, per component)
- **C1** `test_capture` — two streams, timestamps monotonic, inter-camera delta < 8ms, fps ≥ target. On Pi.
- **C2** `test_detection` — synthetic frames: bright blinking dot on animated+glare background →
  assert centroid within tol, rejects static glare, rejects live-screen content, rejects mirror blob. Anywhere.
- **C3** `test_position` — synthetic homographies: known (x,y) → (u0,v0,u1,v1) → assert recovered
  (x,y) within tol; check top-band degradation is bounded. Anywhere.
- **C4** `test_output` — scripted `ScreenPoint` stream → assert cursor lands at expected pixels. On Mac.
- **INT** `test_integration` — static pen at known screen taps → measure mean error + jitter std-dev
  against ≤2mm center / ≤5mm edge target. On rig, after calibration.

## TODO (ordered; independent tracks run in parallel)

### C1 Capture [DAN]
- [x] Pi 5 + 2× CM3 capturing
- [ ] Cropped high-fps mode (target ~120fps) — record actual fps/res achieved
- [ ] Per-frame `timestamp_us` from libcamera; expose `frame_index`
- [ ] Rolling N-frame buffer per camera exposed to C2
- [ ] Emit `FrameBundle` over the agreed in-process API
- [ ] `test_capture` passes on Pi
- [ ] [BLOCKING] Verify stable under dual-cam load on **5V/3A** (no brownout); disable unused peripherals, cap fps/res if needed

### C2 Detection [GAY]
- [x] Basic LED detection working, deployed to Pi 5
- [ ] Subpixel intensity-weighted centroid on largest blob
- [ ] Modulation frame-source: detect blink signature, reject static glare + live-screen content
  - [ ] Pick algorithm (adjacent-diff vs sliding max−min vs Goertzel lock-in) empirically
  - [ ] Choose + document LED blink frequency (avoid screen-refresh harmonics)
- [ ] Threshold+HSV-green frame-source (backup swap) behind same interface
- [ ] Reflection reject: select blob furthest from `surface_line_camN` (air-side)
- [ ] Emit `Detection` (with confidence) per camera
- [ ] `test_detection` passes on synthetic frames

### C3 Calibration + Position [UNASSIGNED]
- [ ] `calibrate_intrinsics.py` — checkerboard → `intrinsics_camN.npz`
- [ ] `calibrate_homography.py` — tap known screen points with real stylus → `homography_camN.npy`
  - [ ] Define screen tap grid + capture surface line per camera
- [ ] Runtime: `undistortPoints` → dual-homography least-squares → `ScreenPoint`
- [ ] Timestamp interpolation of the two detections to a common time before solving
- [ ] `test_position` passes on synthetic geometry

### C4 Output [LIF]
- [x] Coordinates → cursor movement on laptop (Quartz)
- [ ] Consume `ScreenPoint` (normalized) → absolute cursor + pen-down = click/drag
- [ ] One-Euro filter smoothing on stream
- [ ] Transport receive: USB-C CDC-serial primary, WiFi-UDP fallback
- [ ] `test_output` passes on Mac
- [ ] Windows path (SendInput absolute) — only if time

### C5 Stylus HW [UNASSIGNED]
- [ ] Green superbright LED + tip momentary switch (LED powered only on contact)
- [ ] Blink driver at chosen freq (555/ATtiny), fits in marker-barrel body
- [ ] Battery (coin/AAA); untethered; verify camera sees blob at grazing angle across full screen

### C6 Rig + Integration [SHARED]
- [ ] Rigid top-corner camera bar, max baseline, slight down-tilt; clip to laptop bezel
- [ ] Bandpass/gel filter on each lens
- [ ] [BLOCKING] Single USB-C Mac→Pi: power + CDC-serial data (`dtoverlay=dwc2,dr_mode=peripheral` + gadget)
  - [ ] Fallback: WiFi UDP for coordinates if same-cable data fails
- [ ] End-to-end glue: Capture→Detection→Position→transport→Output
- [ ] First live draw on laptop
- [ ] `test_integration`: measure error + jitter vs target
- [ ] Ergonomic cleanup + quick recalibration UX (serves ease-of-use)
- [ ] Demo dry-run + buffer

## Done
- [2026-07-18] Pi 5 + 2× CM3 capturing (DAN)
- [2026-07-18] Basic CV LED detection deployed to Pi 5 (GAY)
- [2026-07-18] Coordinates → cursor movement working on laptop (LIF)
- [2026-07-18] Plan finalized: adjacent-top mount, modulation-primary detection, undistort+dual-homography position, software cursor over USB-C

## Open Questions
- Blink frequency + modulation algorithm — resolve empirically in C2 [BLOCKING for C2 accuracy]
- Does Pi 5 hold stable on 5V/3A under dual-camera load? [BLOCKING — C1]
- Same USB-C cable for power + CDC-serial data, or split data to WiFi? [BLOCKING — C6]
- Cropped mode actual fps/res on CM3 — sets time-sync error budget
- C3 and C5 owners

## Key Decisions
- [2026-07-18] Adjacent-top mount — keeps degenerate ray zone on unused top edge, center well-conditioned
- [2026-07-18] Modulation primary (blink-signature detection), threshold backup — rejects static glare + live-screen content, no hardware sync
- [2026-07-18] Undistort + dual-homography LSQ — avoids extrinsic calibration, absorbs mounting error, single repeatable tap-cal per camera
- [2026-07-18] Pen-down via tip switch gating LED power — no data link, tiny stylus, enforces on-plane homography validity
- [2026-07-18] Software cursor over USB-C serial from Pi — USB-HID gadget dropped (already have Quartz cursor control working)

## Notes
- Rig rigidity is load-bearing: homography valid only while cameras + laptop lid stay fixed. Recalibrate if moved → keep tap-cal fast.
- Homography assumes on-plane LED; pen tilt shifts LED offset → keep LED at the tip, accept small tilt error.
- Calibrate with the actual stylus so tip-height offset bakes into the homography.
- Build against the data contracts above; that is what lets all six tracks proceed in parallel.
