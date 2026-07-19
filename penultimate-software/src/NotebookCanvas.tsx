import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { listen } from "@tauri-apps/api/event";
import { currentMonitor, getCurrentWindow } from "@tauri-apps/api/window";
import {
  createAnnotationWorkspace,
  type AnnotationPoint,
  type AnnotationStyle,
  type AnnotationTool,
  type EraserMode,
  type PenInputEvent,
} from "./annotationModel";
import { renderAnnotationDocument } from "./annotationRenderer";
import { AnnotationToolEngine } from "./annotationTools";

export type NotebookCanvasHandle = {
  undo(): void;
  redo(): void;
  clear(): void;
};

type NotebookCanvasProps = {
  tool: AnnotationTool;
  style: AnnotationStyle;
  text: string;
  eraserMode: EraserMode;
};

type ScreenTransform = {
  monitorX: number;
  monitorY: number;
  monitorWidth: number;
  monitorHeight: number;
  innerX: number;
  innerY: number;
  scaleFactor: number;
};

type ReceivedPenPoint = {
  point: AnnotationPoint;
  rawPoint: AnnotationPoint;
  receivedAt: number;
  speed: number;
};

const MAX_CONTIGUOUS_PEN_DISTANCE = 0.14;
const MAX_CONTIGUOUS_PEN_INTERVAL_MS = 100;
const MIN_CUTOFF_HZ = 0.8;
const SPEED_CUTOFF_BETA = 2.0;
const DERIVATIVE_CUTOFF_HZ = 1.5;

function smoothingAlpha(cutoffHz: number, elapsedMs: number) {
  const elapsedSeconds = Math.max(0.001, elapsedMs / 1_000);
  return 1 - Math.exp(-2 * Math.PI * cutoffHz * elapsedSeconds);
}

const NotebookCanvas = forwardRef<NotebookCanvasHandle, NotebookCanvasProps>(
  function NotebookCanvas({ tool, style, text, eraserMode }, ref) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [, setDocumentRevision] = useState(0);
    const workspaceRef = useRef(createAnnotationWorkspace());
    const styleRef = useRef(style);
    const textRef = useRef(text);
    const toolRef = useRef(tool);
    const eraserModeRef = useRef(eraserMode);
    const drawingRef = useRef(false);
    const screenTransformRef = useRef<ScreenTransform | null>(null);
    const lastPenPointRef = useRef<ReceivedPenPoint | null>(null);
    const shiftRef = useRef(false);
    const redrawRef = useRef<() => void>(() => undefined);
    const engineRef = useRef(
      new AnnotationToolEngine({
        workspace: workspaceRef.current,
        getStyle: () => ({ ...styleRef.current }),
        getText: () => textRef.current,
        getEraserMode: () => eraserModeRef.current,
        getConstrainProportions: () => shiftRef.current,
      }),
    );

    styleRef.current = style;
    textRef.current = text;
    toolRef.current = tool;
    eraserModeRef.current = eraserMode;

    useEffect(() => {
      engineRef.current.select(tool);
    }, [tool]);

    useImperativeHandle(ref, () => ({
      undo() {
        if (engineRef.current.undo()) {
          redrawRef.current();
          setDocumentRevision((revision) => revision + 1);
        }
      },
      redo() {
        if (engineRef.current.redo()) {
          redrawRef.current();
          setDocumentRevision((revision) => revision + 1);
        }
      },
      clear() {
        engineRef.current.clear();
        redrawRef.current();
        setDocumentRevision((revision) => revision + 1);
      },
    }));

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const context = canvas.getContext("2d");
      if (!context) return;

      let renderFrame = 0;
      const render = () => {
        renderAnnotationDocument(context, workspaceRef.current.document, {
          width: canvas.clientWidth,
          height: canvas.clientHeight,
        }, { renderText: false });
      };
      const redraw = () => {
        if (renderFrame) return;
        renderFrame = window.requestAnimationFrame(() => {
          renderFrame = 0;
          render();
        });
      };
      redrawRef.current = redraw;

      const resize = () => {
        const ratio = window.devicePixelRatio || 1;
        const bounds = canvas.getBoundingClientRect();
        canvas.width = Math.round(bounds.width * ratio);
        canvas.height = Math.round(bounds.height * ratio);
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        redraw();
      };
      const observer = new ResizeObserver(resize);
      observer.observe(canvas);
      resize();

      const applyPoint = (point: AnnotationPoint, down: boolean) => {
        if (down && !drawingRef.current) {
          engineRef.current.begin(point);
          drawingRef.current = true;
        } else if (down) {
          engineRef.current.update(point);
        } else if (drawingRef.current) {
          engineRef.current.finish(point);
          drawingRef.current = false;
          // Keep React out of the active ink path. It only needs to update
          // after a completed stroke, for example when a text box was added.
          setDocumentRevision((revision) => revision + 1);
        }
        redraw();
      };

      const refreshScreenTransform = async () => {
        const penultimateWindow = getCurrentWindow();
        const [monitor, innerPosition, scaleFactor] = await Promise.all([
          currentMonitor(),
          penultimateWindow.innerPosition(),
          penultimateWindow.scaleFactor(),
        ]);
        if (!monitor) return;

        screenTransformRef.current = {
          monitorX: monitor.position.x,
          monitorY: monitor.position.y,
          monitorWidth: monitor.size.width,
          monitorHeight: monitor.size.height,
          innerX: innerPosition.x,
          innerY: innerPosition.y,
          scaleFactor,
        };
      };

      const transformTimer = window.setInterval(() => {
        void refreshScreenTransform();
      }, 250);
      void refreshScreenTransform();

      const pointerPoint = (event: PointerEvent): AnnotationPoint => {
        const bounds = canvas.getBoundingClientRect();
        return {
          x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
          y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
          pressure: event.pressure > 0 ? event.pressure : 0.55,
          timestamp: Date.now(),
        };
      };

      const applySmoothedPoint = (rawPoint: AnnotationPoint, down: boolean) => {
        const receivedAt = performance.now();
        const previous = lastPenPointRef.current;
        const rawDistance = previous
          ? Math.hypot(
            rawPoint.x - previous.rawPoint.x,
            rawPoint.y - previous.rawPoint.y,
          )
          : 0;
        const elapsedMs = previous ? receivedAt - previous.receivedAt : 0;

        if (
          down &&
          drawingRef.current &&
          previous &&
          rawDistance > MAX_CONTIGUOUS_PEN_DISTANCE &&
          elapsedMs < MAX_CONTIGUOUS_PEN_INTERVAL_MS
        ) {
          // A tracking jump must not become an ink segment across the page.
          applyPoint(previous.point, false);
          applyPoint(rawPoint, true);
          lastPenPointRef.current = {
            point: rawPoint,
            rawPoint,
            receivedAt,
            speed: 0,
          };
        } else if (down) {
          const rawSpeed = previous && elapsedMs > 0
            ? rawDistance / (elapsedMs / 1_000)
            : 0;
          const speed = previous
            ? previous.speed + (
              rawSpeed - previous.speed
            ) * smoothingAlpha(DERIVATIVE_CUTOFF_HZ, elapsedMs)
            : 0;
          const alpha = previous
            ? smoothingAlpha(MIN_CUTOFF_HZ + SPEED_CUTOFF_BETA * speed, elapsedMs)
            : 1;
          const point = previous
            ? {
              ...rawPoint,
              x: previous.point.x + (rawPoint.x - previous.point.x) * alpha,
              y: previous.point.y + (rawPoint.y - previous.point.y) * alpha,
            }
            : rawPoint;
          applyPoint(point, true);
          lastPenPointRef.current = { point, rawPoint, receivedAt, speed };
        } else {
          applyPoint(previous?.point ?? rawPoint, false);
          lastPenPointRef.current = null;
        }
      };

      const onPointerDown = (event: PointerEvent) => {
        if (toolRef.current === "cursor") return;
        event.preventDefault();
        canvas.setPointerCapture(event.pointerId);
        applySmoothedPoint(pointerPoint(event), true);
      };
      const onPointerMove = (event: PointerEvent) => {
        if (!drawingRef.current) return;
        event.preventDefault();
        applySmoothedPoint(pointerPoint(event), true);
      };
      const onPointerUp = (event: PointerEvent) => {
        if (!drawingRef.current) return;
        const placedText = toolRef.current === "text";
        applySmoothedPoint(pointerPoint(event), false);
        if (canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
        if (placedText) {
          window.requestAnimationFrame(() => {
            const boxes = canvas.parentElement?.querySelectorAll<HTMLTextAreaElement>(
              ".notebook-content-box",
            );
            boxes?.[boxes.length - 1]?.focus();
          });
        }
      };
      canvas.addEventListener("pointerdown", onPointerDown);
      canvas.addEventListener("pointermove", onPointerMove);
      canvas.addEventListener("pointerup", onPointerUp);
      canvas.addEventListener("pointercancel", onPointerUp);

      const onKey = (event: KeyboardEvent) => {
        shiftRef.current = event.shiftKey;
      };
      window.addEventListener("keydown", onKey);
      window.addEventListener("keyup", onKey);

      let unlistenPen: (() => void) | undefined;
      // The Notebook is the active writing surface whenever it is the visible,
      // non-overlay surface; the Rust side only routes pen input here in that
      // case. A real pen never clicks to focus the window, so gating ink on OS
      // focus would silently drop every stroke — route on delivery instead.
      void listen<PenInputEvent>("penultimate:pen-input", ({ payload }) => {
        const bounds = canvas.getBoundingClientRect();
        const transform = screenTransformRef.current;
        if (!transform) return;
        const clientX = (
          transform.monitorX + payload.x * transform.monitorWidth - transform.innerX
        ) / transform.scaleFactor;
        const clientY = (
          transform.monitorY + payload.y * transform.monitorHeight - transform.innerY
        ) / transform.scaleFactor;
        const inside =
          clientX >= bounds.left && clientX <= bounds.right &&
          clientY >= bounds.top && clientY <= bounds.bottom;
        if (!inside && !drawingRef.current) return;
        const rawPoint = {
          x: Math.max(0, Math.min(1, (clientX - bounds.left) / bounds.width)),
          y: Math.max(0, Math.min(1, (clientY - bounds.top) / bounds.height)),
          pressure: Math.max(0, Math.min(1, payload.pressure ?? 0.55)),
          timestamp: payload.timestamp,
        };
        // A stroke may start only inside the paper, but once it is in progress
        // let it continue while the pen wanders past the edge (points clamp to
        // the border) rather than chopping it into separate annotations.
        applySmoothedPoint(rawPoint, payload.penDown && (inside || drawingRef.current));
      }).then((unlisten) => {
        unlistenPen = unlisten;
      });

      return () => {
        observer.disconnect();
        if (renderFrame) window.cancelAnimationFrame(renderFrame);
        window.clearInterval(transformTimer);
        unlistenPen?.();
        window.removeEventListener("keydown", onKey);
        window.removeEventListener("keyup", onKey);
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("pointercancel", onPointerUp);
      };
    }, []);

    const textAnnotations = workspaceRef.current.document.annotations.filter(
      (annotation) => annotation.kind === "text",
    );

    return (
      <>
        <canvas
          ref={canvasRef}
          className={`notebook-canvas ${tool === "cursor" ? "cursor-mode" : ""}`}
        />
        {textAnnotations.map((annotation) => (
          <textarea
            key={annotation.id}
            className="notebook-content-box"
            style={{
              left: `${annotation.position.x * 100}%`,
              top: `${annotation.position.y * 100}%`,
              color: annotation.style.color,
              fontSize: `${annotation.fontSize}px`,
              opacity: annotation.style.opacity,
            }}
            value={annotation.text}
            placeholder="Type here"
            onFocus={() => {
              engineRef.current.history.begin(workspaceRef.current.document);
            }}
            onChange={(event) => {
              annotation.text = event.currentTarget.value;
              redrawRef.current();
              setDocumentRevision((revision) => revision + 1);
            }}
            onBlur={() => {
              engineRef.current.history.commit(workspaceRef.current.document);
              redrawRef.current();
              setDocumentRevision((revision) => revision + 1);
            }}
            onPointerDown={(event) => event.stopPropagation()}
            aria-label="Editable text box"
          />
        ))}
      </>
    );
  },
);

export default NotebookCanvas;
