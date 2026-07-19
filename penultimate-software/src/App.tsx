import { useEffect, useRef, useState } from "react";
import { emitTo, listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { register, unregister } from "@tauri-apps/plugin-global-shortcut";
import {
  getCurrentWindow,
  PhysicalPosition,
  PhysicalSize,
  primaryMonitor,
} from "@tauri-apps/api/window";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import {
  BookOpen,
  Circle,
  Crosshair,
  Eraser,
  Highlighter,
  MousePointer2,
  PenTool,
  Pencil,
  RectangleHorizontal,
  Redo2,
  Settings2,
  Type,
  Undo2,
} from "lucide-react";
import CalibrationScreen from "./CalibrationScreen";
import StatusHud from "./StatusHud";
import AnnotationOverlay from "./AnnotationOverlay";
import AnnotationToolbar from "./AnnotationToolbar";
import NotebookCanvas, { type NotebookCanvasHandle } from "./NotebookCanvas";
import {
  defaultStyleForTool,
  type AnnotationStyle,
  type AnnotationTool,
  type EraserMode,
} from "./annotationModel";
import "./App.css";

type OverlaySnapshot = {
  overlayEnabled: boolean;
  cursorActive: boolean;
  trackpadOverrideEnabled: boolean;
  trackpadDrawingEnabled: boolean;
  simulated: boolean;
  rate: number;
  appliedRate: number;
  moveLatencyMs: string;
  connected: boolean;
  latencyMs: string;
  source: "simulated" | "pi";
};

type RuntimeFlagsPayload = {
  overlayEnabled?: boolean;
  cursorActive?: boolean;
  trackpadOverrideEnabled?: boolean;
  trackpadDrawingEnabled?: boolean;
  simulated?: boolean;
};

type RuntimeControlPayload = Pick<
  OverlaySnapshot,
  | "overlayEnabled"
  | "cursorActive"
  | "simulated"
  | "trackpadOverrideEnabled"
  | "trackpadDrawingEnabled"
>;

type DisplayMetrics = {
  loopHz: string;
  moveHz: string;
  moveLatencyMs: string;
};

const OVERLAY_LABEL = "overlay";
const NOTEBOOK_LABEL = "notebook";
const CALIBRATION_LABEL = "calibration";
const STATUS_LABEL = "status";
const TOOLBAR_LABEL = "toolbar";
const METRIC_POLL_MS = 250;
const METRIC_RENDER_MS = 1000;

function createInitialSnapshot(): OverlaySnapshot {
  return {
    overlayEnabled: false,
    cursorActive: true,
    trackpadOverrideEnabled: true,
    trackpadDrawingEnabled: false,
    simulated: true,
    rate: 0,
    appliedRate: 0,
    moveLatencyMs: "—",
    connected: false,
    latencyMs: "—",
    source: "simulated",
  };
}

function isRealMeasurement(value: string) {
  return value !== "—" && value !== "poll failed" && value.length > 0;
}

function createInitialDisplayMetrics(): DisplayMetrics {
  return {
    loopHz: "measuring…",
    moveHz: "measuring…",
    moveLatencyMs: "measuring…",
  };
}

function snapshotsRenderTheSame(a: OverlaySnapshot, b: OverlaySnapshot) {
  return (
    a.overlayEnabled === b.overlayEnabled &&
    a.cursorActive === b.cursorActive &&
    a.trackpadOverrideEnabled === b.trackpadOverrideEnabled &&
    a.trackpadDrawingEnabled === b.trackpadDrawingEnabled &&
    a.simulated === b.simulated &&
    a.connected === b.connected &&
    a.source === b.source
  );
}

function metricsAreTheSame(a: DisplayMetrics, b: DisplayMetrics) {
  return (
    a.loopHz === b.loopHz &&
    a.moveHz === b.moveHz &&
    a.moveLatencyMs === b.moveLatencyMs
  );
}

function App() {
  const windowLabel = getCurrentWebviewWindow().label;
  const isOverlayWindow = windowLabel === OVERLAY_LABEL;
  const isNotebookWindow = windowLabel === NOTEBOOK_LABEL;
  const isCalibrationWindow = windowLabel === CALIBRATION_LABEL;
  const isToolbarWindow = windowLabel === TOOLBAR_LABEL;
  const isStatusWindow = windowLabel === STATUS_LABEL;
  const [snapshot, setSnapshot] = useState<OverlaySnapshot>(() =>
    createInitialSnapshot(),
  );
  const [, setDisplayMetrics] = useState<DisplayMetrics>(() =>
    createInitialDisplayMetrics(),
  );
  const [calibrationLaunchError, setCalibrationLaunchError] = useState("");
  const [notebookTool, setNotebookTool] = useState<AnnotationTool>("pen");
  const [notebookStyle, setNotebookStyle] = useState<AnnotationStyle>(() =>
    defaultStyleForTool("pen"),
  );
  const notebookText = "";
  const [eraserMode, setEraserMode] = useState<EraserMode>("stroke");
  const [toolSettingsOpen, setToolSettingsOpen] = useState(false);
  const notebookCanvasRef = useRef<NotebookCanvasHandle>(null);
  const snapshotRef = useRef<OverlaySnapshot>(createInitialSnapshot());
  const lastMetricRenderAtRef = useRef(0);
  const lastShortcutToggleAtRef = useRef(0);
  const lastEscapeToggleAtRef = useRef(0);

  const applySnapshot = (nextSnapshot: OverlaySnapshot) => {
    snapshotRef.current = nextSnapshot;
    setSnapshot((previousSnapshot) =>
      snapshotsRenderTheSame(previousSnapshot, nextSnapshot)
        ? previousSnapshot
        : nextSnapshot,
    );
  };

  const applyRuntimeSnapshot = (nextSnapshot: OverlaySnapshot) => {
    const previous = snapshotRef.current;
    const now = Date.now();
    const overlayIsMoving =
      nextSnapshot.overlayEnabled &&
      nextSnapshot.cursorActive &&
      nextSnapshot.simulated;

    if (now - lastMetricRenderAtRef.current >= METRIC_RENDER_MS) {
      lastMetricRenderAtRef.current = now;

      setDisplayMetrics((previousMetrics) => {
        const nextMetrics = {
          loopHz:
            nextSnapshot.simulated && nextSnapshot.rate > 0
              ? `${Math.round(nextSnapshot.rate)} loop Hz`
              : previousMetrics.loopHz,
          moveHz:
            overlayIsMoving && nextSnapshot.appliedRate > 0
              ? `${Math.round(nextSnapshot.appliedRate)} move Hz`
              : previousMetrics.moveHz,
          moveLatencyMs:
            overlayIsMoving && isRealMeasurement(nextSnapshot.moveLatencyMs)
              ? nextSnapshot.moveLatencyMs
              : previousMetrics.moveLatencyMs,
        };

        return metricsAreTheSame(previousMetrics, nextMetrics)
          ? previousMetrics
          : nextMetrics;
      });
    }

    applySnapshot({
      ...nextSnapshot,
      rate:
        nextSnapshot.simulated && nextSnapshot.rate === 0 && previous.rate > 0
          ? previous.rate
          : nextSnapshot.rate,
      appliedRate:
        overlayIsMoving &&
        nextSnapshot.appliedRate === 0 &&
        previous.appliedRate > 0
          ? previous.appliedRate
          : nextSnapshot.appliedRate,
      moveLatencyMs:
        overlayIsMoving &&
        nextSnapshot.moveLatencyMs === "—" &&
        isRealMeasurement(previous.moveLatencyMs)
          ? previous.moveLatencyMs
          : nextSnapshot.moveLatencyMs,
    });
  };

  const applyControls = (payload: Partial<RuntimeControlPayload>) => {
    applySnapshot({
      ...snapshotRef.current,
      ...payload,
    });
  };

  const pushRuntimeFlags = async (payload: RuntimeFlagsPayload) => {
    await invoke("set_runtime_flags", { payload });
  };

  useEffect(() => {
    void (async () => {
      const window = getCurrentWindow();

      if (isOverlayWindow) {
        const monitor = await primaryMonitor();

        await window.setDecorations(false);
        await window.setAlwaysOnTop(true);
        await window.setSkipTaskbar(true);
        await window.setResizable(false);
        await window.setFocusable(false);
        await window.setIgnoreCursorEvents(true);

        if (monitor) {
          await window.setPosition(
            new PhysicalPosition(monitor.position.x, monitor.position.y),
          );
          await window.setSize(
            new PhysicalSize(monitor.size.width, monitor.size.height),
          );
        }

        if (snapshotRef.current.overlayEnabled) {
          await window.show();
        } else {
          await window.hide();
        }
      } else if (isToolbarWindow) {
        await window.setDecorations(false);
        await window.setAlwaysOnTop(true);
        await window.setSkipTaskbar(true);
        await window.setResizable(false);
      }
    })();
  }, [isOverlayWindow, isToolbarWindow]);

  useEffect(() => {
    if (!isOverlayWindow) {
      return;
    }

    const unlisteners: Array<() => void> = [];

    void listen<boolean>("penultimate:set-overlay-enabled", async (event) => {
      const next = event.payload;
      const window = getCurrentWindow();
      if (next) {
        await window.show();
        await window.setFocusable(false);
        await window.setIgnoreCursorEvents(true);
      } else {
        await window.hide();
      }
    }).then((unlisten) => unlisteners.push(unlisten));

    return () => {
      for (const unlisten of unlisteners) {
        unlisten();
      }
    };
  }, [isOverlayWindow]);

  useEffect(() => {
    if (!isNotebookWindow) {
      return;
    }

    const unlisteners: Array<() => void> = [];

    void listen<boolean>(
      "penultimate:set-notebook-visible",
      async (event) => {
        const notebookWindow = getCurrentWindow();
        if (event.payload) {
          await notebookWindow.show();
          await notebookWindow.setFocus();
        } else {
          await notebookWindow.hide();
        }
      },
    ).then((unlisten) => unlisteners.push(unlisten));

    const toggleCursorPause = () => {
      const current = snapshotRef.current;

      if (current.trackpadDrawingEnabled) {
        applyControls({ trackpadDrawingEnabled: false });
        void pushRuntimeFlags({ trackpadDrawingEnabled: false });
        return;
      }

      const nextCursorActive = !current.cursorActive;
      applyControls({ cursorActive: nextCursorActive });
      void pushRuntimeFlags({ cursorActive: nextCursorActive });
    };

    void register("CommandOrControl+Alt+A", () => {
      const now = Date.now();
      if (now - lastShortcutToggleAtRef.current < 350) {
        return;
      }
      lastShortcutToggleAtRef.current = now;

      const current = snapshotRef.current;
      const nextOverlayEnabled = !current.overlayEnabled;
      applyControls({
        overlayEnabled: nextOverlayEnabled,
      });
      void emitTo(OVERLAY_LABEL, "penultimate:set-overlay-enabled", nextOverlayEnabled);
      void emitTo(TOOLBAR_LABEL, "penultimate:set-toolbar-visible", nextOverlayEnabled);
      void pushRuntimeFlags({
        overlayEnabled: nextOverlayEnabled,
      });
    });

    void register("Escape", () => {
      const now = Date.now();
      if (now - lastEscapeToggleAtRef.current < 350) {
        return;
      }
      lastEscapeToggleAtRef.current = now;

      void getCurrentWindow().isFocused().then((focused) => {
        if (focused) {
          setNotebookTool("cursor");
          setToolSettingsOpen(false);
        } else {
          toggleCursorPause();
        }
      });
    });

    void register("CommandOrControl+Alt+C", () => {
      void emitTo(OVERLAY_LABEL, "penultimate:clear-annotation");
    });

    void register("CommandOrControl+Alt+Z", () => {
      void emitTo(OVERLAY_LABEL, "penultimate:undo-annotation");
    });

    void register("CommandOrControl+Alt+Shift+Z", () => {
      void emitTo(OVERLAY_LABEL, "penultimate:redo-annotation");
    });

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setNotebookTool("cursor");
        setToolSettingsOpen(false);
      }
    };

    window.addEventListener("keydown", onKeyDown);

    return () => {
      window.removeEventListener("keydown", onKeyDown);
      void unregister("CommandOrControl+Alt+A");
      void unregister("Escape");
      void unregister("CommandOrControl+Alt+C");
      void unregister("CommandOrControl+Alt+Z");
      void unregister("CommandOrControl+Alt+Shift+Z");
      for (const unlisten of unlisteners) {
        unlisten();
      }
    };
  }, [isNotebookWindow]);

  useEffect(() => {
    if (!isNotebookWindow) {
      return;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const nextSnapshot = (await invoke(
          "get_runtime_snapshot",
        )) as OverlaySnapshot;
        if (!cancelled) {
          applyRuntimeSnapshot(nextSnapshot);
        }
      } catch (error) {
        if (!cancelled) {
          applySnapshot({
            ...snapshotRef.current,
            connected: false,
            latencyMs: "poll failed",
            moveLatencyMs:
              error instanceof Error ? error.message : "poll failed",
          });
        }
      }
    };

    void poll();
    const timer = window.setInterval(() => {
      void poll();
    }, METRIC_POLL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [isNotebookWindow]);

  if (isOverlayWindow) {
    return <AnnotationOverlay />;
  }

  if (isCalibrationWindow) {
    return <CalibrationScreen />;
  }

  if (isToolbarWindow) {
    return <AnnotationToolbar />;
  }

  if (isStatusWindow) {
    return <StatusHud />;
  }

  const toggleAnnotation = () => {
    const nextOverlayEnabled = !snapshotRef.current.overlayEnabled;
    applyControls({
      overlayEnabled: nextOverlayEnabled,
    });
    void emitTo(
      OVERLAY_LABEL,
      "penultimate:set-overlay-enabled",
      nextOverlayEnabled,
    );
    void emitTo(
      TOOLBAR_LABEL,
      "penultimate:set-toolbar-visible",
      nextOverlayEnabled,
    );
    void pushRuntimeFlags({
      overlayEnabled: nextOverlayEnabled,
    });
  };

  const launchCalibration = async () => {
    setCalibrationLaunchError("");
    try {
      await invoke("open_calibration");
    } catch (error) {
      setCalibrationLaunchError(String(error));
    }
  };

  const selectNotebookTool = (tool: AnnotationTool) => {
    if (tool === notebookTool) {
      if (tool !== "cursor") setToolSettingsOpen((open) => !open);
      return;
    }
    setNotebookTool(tool);
    setNotebookStyle(defaultStyleForTool(tool));
    setToolSettingsOpen(false);
  };

  const updateNotebookStyle = (patch: Partial<AnnotationStyle>) => {
    setNotebookStyle((current) => ({ ...current, ...patch }));
  };

  const notebookTools: Array<{
    tool: AnnotationTool;
    label: string;
    icon: typeof PenTool;
  }> = [
    { tool: "cursor", label: "Cursor", icon: MousePointer2 },
    { tool: "pen", label: "Pen", icon: PenTool },
    { tool: "pencil", label: "Pencil", icon: Pencil },
    { tool: "highlighter", label: "Highlighter", icon: Highlighter },
    { tool: "eraser", label: "Eraser", icon: Eraser },
    { tool: "rounded-rectangle", label: "Rectangle", icon: RectangleHorizontal },
    { tool: "ellipse", label: "Ellipse", icon: Circle },
    { tool: "text", label: "Text", icon: Type },
  ];

  const notebookColors = ["#20252b", "#1769aa", "#d83b3b", "#16845b", "#f1bf30"];
  const canAdjustNotebookStroke =
    notebookTool !== "cursor" &&
    notebookTool !== "eraser" &&
    notebookTool !== "text";

  const maxStrokeWidth = (tool: AnnotationTool) => {
    switch (tool) {
      case "pencil":
        return 20;
      case "pen":
        return 32;
      case "marker":
        return 48;
      case "highlighter":
        return 64;
      default:
        return 32;
    }
  };

  return (
    <main className="notebook-shell">
      <header className="notebook-header" data-tauri-drag-region>
        <div className="notebook-identity">
          <BookOpen size={18} strokeWidth={1.8} aria-hidden="true" />
          <div className="status-label">Penultimate</div>
        </div>
        <div className="notebook-tools">
          <span className={`connection-status idle ${snapshot.connected ? "is-hidden" : ""}`}>
            <span aria-hidden="true" />
            Waiting for pen
          </span>
          <button
            className={`toolbar-command ${snapshot.overlayEnabled ? "active" : ""}`}
            onClick={toggleAnnotation}
            title="Toggle screen annotation"
            aria-pressed={snapshot.overlayEnabled}
          >
            <Crosshair size={16} aria-hidden="true" />
            <span>Annotate</span>
          </button>
          <button
            className="icon-command"
            onClick={() => void launchCalibration()}
            aria-label="Calibrate screen"
            title="Calibrate screen"
          >
            <Settings2 size={17} aria-hidden="true" />
          </button>
        </div>
      </header>

      <div className="notebook-workspace">
        <div className="notebook-instrument-bar">
          <div className="notebook-tool-group" aria-label="Writing tools">
            {notebookTools.map(({ tool, label, icon: Icon }) => (
              <div className="notebook-tool-anchor" key={tool}>
                <button
                  className={`notebook-tool ${notebookTool === tool ? "selected" : ""}`}
                  onClick={() => selectNotebookTool(tool)}
                  aria-label={label}
                  title={notebookTool === tool && tool !== "cursor" ? `${label} settings` : label}
                  aria-expanded={notebookTool === tool && toolSettingsOpen}
                >
                  <Icon size={19} strokeWidth={1.8} />
                </button>
                {notebookTool === tool && toolSettingsOpen && (
                  <section className="tool-settings-popover" aria-label={`${label} settings`}>
                    <div className="tool-settings-heading">{label}</div>
                    <div className={`tool-live-preview ${tool === "eraser" ? "eraser-preview" : ""}`}>
                      <span
                        style={{
                          backgroundColor: tool === "eraser" ? "#9aa3ad" : notebookStyle.color,
                          height: tool === "eraser"
                            ? `${Math.min(42, notebookStyle.width)}px`
                            : `${Math.min(10, notebookStyle.width)}px`,
                          width: tool === "eraser"
                            ? `${Math.min(42, notebookStyle.width)}px`
                            : undefined,
                          opacity: notebookStyle.opacity,
                        }}
                      />
                    </div>
                    {tool === "eraser" ? (
                      <>
                        <div className="eraser-mode-options" aria-label="Eraser mode">
                          <button className={eraserMode === "stroke" ? "selected" : ""} onClick={() => setEraserMode("stroke")}>Stroke</button>
                          <button className={eraserMode === "pixel" ? "selected" : ""} onClick={() => setEraserMode("pixel")}>Pixel</button>
                        </div>
                        {eraserMode === "pixel" && (
                          <label className="tool-setting-row">
                            <span>Size</span><output>{Math.round(notebookStyle.width)} px</output>
                            <input type="range" min="6" max="64" step="2" value={notebookStyle.width} onChange={(event) => updateNotebookStyle({ width: Number(event.currentTarget.value) })} />
                          </label>
                        )}
                      </>
                    ) : tool === "text" ? (
                      <p className="tool-settings-hint">Click anywhere on the canvas, then type. Press Enter for a new line.</p>
                    ) : (
                      <label className="tool-setting-row">
                        <span>Thickness</span><output>{notebookStyle.width}px</output>
                        <input type="range" min="1" max={maxStrokeWidth(tool)} step="0.5" value={notebookStyle.width} onChange={(event) => updateNotebookStyle({ width: Number(event.currentTarget.value) })} />
                      </label>
                    )}
                  </section>
                )}
              </div>
            ))}
          </div>
          <span className="instrument-divider" />
          {notebookTool !== "eraser" && notebookTool !== "cursor" && (
            <div className="notebook-color-group" aria-label="Ink color">
              {notebookColors.map((color) => (
                <button
                  key={color}
                  className={`notebook-color ${notebookStyle.color === color ? "selected" : ""}`}
                  style={{ backgroundColor: color }}
                  onClick={() => updateNotebookStyle({ color })}
                  aria-label={`Ink color ${color}`}
                />
              ))}
            </div>
          )}
          {canAdjustNotebookStroke && (
            <label className="notebook-size" title="Stroke thickness">
              <span
                className="stroke-preview"
                style={{
                  backgroundColor: notebookStyle.color,
                  height: `${Math.min(14, notebookStyle.width)}px`,
                  opacity: notebookStyle.opacity,
                }}
                aria-hidden="true"
              />
              <input
                type="range"
                min="1"
                max={maxStrokeWidth(notebookTool)}
                step="0.5"
                value={notebookStyle.width}
                onChange={(event) =>
                  updateNotebookStyle({ width: Number(event.currentTarget.value) })
                }
                aria-label="Stroke thickness"
              />
              <output>{notebookStyle.width}px</output>
            </label>
          )}
          <span className="instrument-spacer" />
          <button className="notebook-tool" onClick={() => notebookCanvasRef.current?.undo()} aria-label="Undo" title="Undo">
            <Undo2 size={18} />
          </button>
          <button className="notebook-tool" onClick={() => notebookCanvasRef.current?.redo()} aria-label="Redo" title="Redo">
            <Redo2 size={18} />
          </button>
        </div>

        <section className="notebook-stage" aria-label="Notes workspace">
          <div className="paper-page dotted-paper">
            <NotebookCanvas
              ref={notebookCanvasRef}
              tool={notebookTool}
              style={notebookStyle}
              text={notebookText}
              eraserMode={eraserMode}
            />
          </div>
          {calibrationLaunchError && (
            <div className="control-error" role="alert">
              {calibrationLaunchError}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

export default App;
