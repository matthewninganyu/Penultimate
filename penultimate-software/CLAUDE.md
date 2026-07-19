# Penultimate project notes

Penultimate is a laptop-side app for making a non-touchscreen laptop feel touchscreen-like. A Raspberry Pi with two camera modules will eventually track the pen and send input data to the laptop. For now, the laptop app should work with simulated data.

## Product direction

The pen should feel ambient, not mode-heavy.

Core principle:

```text
Penultimate is running -> the pen moves the laptop cursor.
```

Do not build a persistent mode overlay or mode selector. The user should not have to think in terms of "Cursor Mode", "Annotation Mode", or "Notebook Mode". Cursor movement is the baseline behavior.

Annotation and notebook behavior should be treated as surfaces/contexts, not top-level modes:

- No active surface: pen moves the cursor only.
- Screen annotation surface: pen moves the cursor and pen-down can draw over the screen.
- Notebook surface: pen moves the cursor and pen-down can write inside the notebook app.

## UX decisions

- A cold app launch starts ambiently with cursor injection active and no visible
  Notebook, annotation overlay, or toolbar. Opening the Notebook must be an
  explicit action after launch.
- On macOS, Penultimate remains present in the Dock so it can be opened as a
  normal note-taking app. Cold launch must still leave Notebook hidden and must
  not take focus; a Dock re-open shows and focuses Notebook. The status HUD is
  non-focusable and may appear without activating it.
- Provide a native menu-bar icon as the discoverable companion to shortcuts.
  Its menu contains Open Notebook, Start/Stop Annotation (with
  `Cmd + Option + A` shown), Calibrate Screen, live pen connection status, and
  Quit Penultimate. Menu actions must work while every app window is hidden.
- Notebook should behave like a regular desktop app window.
- On macOS, use an overlay title bar so the traffic lights sit within the
  notebook's unified pale blue-gray header rather than above it in an empty
  white strip. Use the native traffic-light inset x=14, y=28 to center the
  controls in the 52 px header; this inset is not equivalent to a CSS top
  coordinate. Keep the left 86 px clear.
- The Notebook writing surface is pageless and fills the available workspace.
  Use a restrained 24 px dot grid rather than ruled lines or a bordered paper
  sheet; do not show page numbers, page navigation, or thumbnails.
- The notebook eraser offers `Stroke` and `Pixel` modes. Stroke removes whole
  annotation objects. Pixel records an undoable compositing path that removes
  only the rendered pixels beneath it, with an adjustable 6-64 px size.
- The Notebook toolbar begins with a Cursor tool. It is a non-marking state
  that lets the user stop drawing and use a normal pointer within the notebook;
  it is a surface tool, not a global Penultimate mode. Pressing Escape while
  the Notebook is focused selects Cursor.
- Notebook text is presented as editable, resizable content boxes over the
  canvas rather than immediately flattened text. Existing boxes remain
  directly editable, and Enter inserts line breaks within a box.
- Keep the notebook toolbar compact: tool and color selection remain in the
  row, while clicking the already-selected tool opens a small anchored settings
  popover with a live preview and only that tool's relevant controls.
- The transparent overlay should not be the main app experience.
- The overlay should remain transparent, click-through, and unfocusable when used.
- Avoid persistent UI that obstructs the screen.
- A mode/status overlay is not needed.
- Temporary feedback is limited to pen connection state.
- Present status in a dedicated top-center, transparent, click-through HUD.
  Check connection state every 50 ms and show `Pen connected` immediately for
  1.5 seconds. Debounce only disconnection for 400 ms, then keep
  `Pen disconnected` visible until reconnection. Do not show paused, resumed,
  or trackpad override states. Start the native HUD window visible but
  transparent so its webview initializes; the component hides it after
  transient messages.
- Style the HUD as a flat 38 px blue-gray tag with 9 px corners, ink-blue text,
  and a separate muted green or coral state dot. Use a quiet
  solid surface and thin border with no shadow. Position it 48 px below the
  screen top. Enter over 320 ms by revealing outward from the center while
  settling down 8 px; stagger the label and dot behind the surface. Exit upward
  over 170 ms before hiding the native window. It should match the notebook
  rather than resemble a system toast.

## Internal state model

Internal state is still needed, but it should not be presented as a user-facing mode selector.

Preferred direction:

```ts
type ActiveSurface = "none" | "overlay" | "notebook";

type RuntimeState = {
  enabled: boolean;
  pausedByUser: boolean;
  pausedByTrackpad: boolean;
  inputSource: "simulated" | "pi";
  activeSurface: ActiveSurface;
};
```

Derived behavior:

```ts
const cursorInjectionActive =
  enabled &&
  !pausedByUser &&
  !pausedByTrackpad;

const inkActive =
  cursorInjectionActive &&
  activeSurface !== "none";
```

Cursor injection must never depend on whether the annotation overlay is
visible. Closing or disabling annotation returns to cursor-only behavior; it
does not pause the pen.

Route pen input to exactly one active writing surface: the overlay while screen
annotation is enabled, otherwise the visible Notebook. The Notebook must
receive `penDown` events so a real pen can start and continue note strokes.

The Pi UDP adapter accepts `normalized_x`, `normalized_y`, `touching`, and a
fractional Unix-seconds timestamp directly. Map them to normalized cursor
coordinates, `penDown`, and Unix milliseconds; reject packets where
`valid: false`.

## Current prototype behavior

- Tauri 2 + React/TypeScript frontend.
- Rust backend handles the high-frequency simulated cursor loop.
- The app currently supports simulated input data because Raspberry Pi data is not ready yet.
- The overlay window is transparent and click-through.
- The normal Dock window is the Notes surface. The old overlay controls window is not part of the product UI.
- `Cmd + Option + A` toggles screen annotation.
- The annotation toolbar ends with an icon-only Close control at the far right
  of its top row, separated from drawing actions. Close disables annotation,
  exits trackpad drawing, hides both overlay and toolbar, and leaves cursor
  injection running.
- The pointer icon in the annotation toolbar selects the non-marking Cursor
  tool (`V`). Do not map a pointer-shaped selector icon to the red Laser tool.
- Center the floating annotation toolbar horizontally on its current monitor at
  runtime, while keeping it 40 px below the top edge.
- The floating annotation toolbar is frosted paper: use a near-neutral 76%
  surface over native macOS Popover vibrancy so it can actually blur the apps
  behind the transparent window. Keep the CSS tint near 38%, with 12 px
  corners, a thin light edge, and one soft shadow. Tool hover states use
  translucent white and selection uses a softened ink blue; preserve contrast
  without making it decorative.
- `Escape` is registered as a global pause/resume shortcut, with a focused-window fallback.
- Trackpad/mouse override is implemented in Rust:
  - Penultimate remembers the last cursor position it injected.
  - If the real cursor later appears noticeably away from that position, it assumes the user moved the trackpad/mouse.
  - Cursor injection pauses briefly, then resumes automatically.

## Implementation caution

Do not reintroduce a user-facing mode toggle unless the product direction changes explicitly.

Future work should focus on:

1. Cleaning the internal runtime state around enabled/paused/surface/source.
2. Making the Dock icon open/focus a normal Notebook window.
3. Keeping the transparent overlay as an invisible annotation/input layer.
4. Keeping diagnostics out of the user-facing flow.
5. Adding a proper input event shape for Raspberry Pi data:

```ts
type PenInputEvent = {
  x: number;
  y: number;
  penDown: boolean;
  pressure?: number;
  timestamp: number;
};
```

## Calibration decisions

Calibration is automatic. It is a temporary full-screen surface that lets the
Raspberry Pi locate the display without asking the user to touch individual
points. The laptop UI presents the visual markers and status; the Pi owns
detection, confidence checks, coordinate measurement, and the resulting
homography.

After a 130 ms near-black reference frame, show four large, solid, symmetrical
L-shaped markers simultaneously. Each L has two equal 128 px arms that are 28
px thick and points inward from its corner. Keep a 12 px horizontal inset; the
bottom pair sits visually flush with the bottom edge while allowing for the
native window's seam-hiding overscan:

```text
top-left:     red    #FF3B30
top-right:    green  #34C759
bottom-right: blue   #0A84FF
bottom-left:  yellow #FFD60A
```

The background is flat near-black `#080B10`. Marker colors, screen position,
and timing provide redundant identity; detection must not depend on exact RGB
values because camera exposure and white balance will shift them.

```ts
type CalibrationMessage = {
  level?: "success" | "warning" | "error";
  message?: string;
  progress?: number; // 0-1
};
```

Status semantics:

- Capture should feel nearly immediate. At 30 fps, the Pi should normally
  confirm several stable frames while the markers remain steady for roughly
  250-400 ms. Do not fade or animate marker colors during capture.
- The current laptop-side sequence treats the steady 350 ms marker interval as
  captured automatically. Remove the markers, show a restrained confirmation
  for 250 ms, then close the surface automatically. A blocking Pi error still
  prevents completion.
- Warning is recoverable. Use the calm status `Adjusting…` while the Pi
  continues looking; clear it automatically when usable samples resume.
- Error is blocking and requires Retry or Cancel. While an error is active,
  later progress or success messages must be ignored. Losing the Pi connection
  is an error.
- Retry restarts the reference frame and four-corner scan.
- Cancel preserves any previously valid saved calibration.

The calibration UI should remain flat and restrained: no gradients, glow,
glass effects, or decorative overlay graphics.

Calibration presentation details:

- Keep a small, stationary central message: `Calibrating your screen` and
  `Keep the screen in view.` A subtle indeterminate line can communicate
  activity without implying a long multi-step process.
- Show all four markers at once. They should appear immediately after the
  reference frame and remain completely steady until success or a blocking
  error.
- The expected perceived duration is roughly 0.6-0.9 seconds including the
  reference, capture, and confirmation states. Do not add artificial waits.
- Hide simulated Pi controls by default. `Cmd + Shift + D` reveals them during
  development; the R, W, E, and S keyboard controls remain available.
- Show fatal errors in a focused Retry/Cancel panel. Avoid layering additional
  progress graphics or decorative completion animations onto the flow.
