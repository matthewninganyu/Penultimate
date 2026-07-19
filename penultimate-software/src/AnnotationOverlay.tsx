import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import {
  createAnnotationWorkspace,
  defaultStyleForTool,
  pointFromPenEvent,
  type AnnotationTool,
  type AnnotationStyle,
  type PenInputEvent,
} from "./annotationModel";
import { AnnotationToolEngine } from "./annotationTools";
import {
  drawFreehandSegment,
  drawLaserAnnotations,
  renderAnnotationDocument,
} from "./annotationRenderer";

export default function AnnotationOverlay() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [trackpadDrawing, setTrackpadDrawing] = useState(false);
  const trackpadDrawingRef = useRef(false);
  const workspaceRef = useRef(createAnnotationWorkspace());
  const constrainProportionsRef = useRef(false);
  const activeStyleRef = useRef(defaultStyleForTool("pen"));
  const annotationTextRef = useRef("Note");
  const toolEngineRef = useRef(
    new AnnotationToolEngine({
      workspace: workspaceRef.current,
      getStyle: () => ({ ...activeStyleRef.current }),
      getConstrainProportions: () => constrainProportionsRef.current,
      getText: () => annotationTextRef.current,
    }),
  );
  const lastPenDownRef = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const redraw = () => {
      renderAnnotationDocument(context, workspaceRef.current.document, {
        width: canvas.clientWidth,
        height: canvas.clientHeight,
      });
      drawLaserAnnotations(
        context,
        workspaceRef.current.transientAnnotations,
        { width: canvas.clientWidth, height: canvas.clientHeight },
        Date.now(),
      );
    };

    const resize = () => {
      const ratio = window.devicePixelRatio || 1;
      const width = window.innerWidth;
      const height = window.innerHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      redraw();
    };

    resize();
    window.addEventListener("resize", resize);

    const applyPenInput = (payload: PenInputEvent) => {
      const point = pointFromPenEvent(payload);

      if (payload.penDown) {
        if (!lastPenDownRef.current) {
          toolEngineRef.current.begin(point);
          if (toolEngineRef.current.tool === "eraser") redraw();
        } else {
          const activeTool = toolEngineRef.current.tool;
          const annotations = workspaceRef.current.document.annotations;
          const stroke = annotations[annotations.length - 1];
          const previous =
            stroke?.kind === "freehand" && stroke.tool === activeTool
              ? stroke.points[stroke.points.length - 1]
              : null;
          const beforePrevious =
            stroke?.kind === "freehand" && stroke.tool === activeTool
              ? stroke.points[stroke.points.length - 2]
              : undefined;
          toolEngineRef.current.update(point);
          if (
            previous &&
            stroke.kind === "freehand" &&
            stroke.tool === activeTool
          ) {
            drawFreehandSegment(context, stroke, previous, point, {
              width: canvas.clientWidth,
              height: canvas.clientHeight,
            }, beforePrevious);
          } else if (stroke?.kind === "shape" && stroke.tool === activeTool) {
            redraw();
          } else {
            redraw();
          }
        }
      } else {
        if (lastPenDownRef.current) {
          toolEngineRef.current.finish(point);
          redraw();
        }
      }
      lastPenDownRef.current = payload.penDown;
    };

    const unlisteners: Array<() => void> = [];
    void listen<PenInputEvent>("penultimate:pen-input", ({ payload }) => {
      applyPenInput(payload);
    }).then((unlisten) => unlisteners.push(unlisten));

    void listen<boolean>("penultimate:set-trackpad-drawing", (event) => {
      trackpadDrawingRef.current = event.payload;
      setTrackpadDrawing(event.payload);
      if (!event.payload && lastPenDownRef.current) {
        toolEngineRef.current.cancel();
        lastPenDownRef.current = false;
        redraw();
      }
    }).then((unlisten) => unlisteners.push(unlisten));

    void listen<boolean>("penultimate:set-overlay-enabled", (event) => {
      if (event.payload) return;
      toolEngineRef.current.cancel();
      lastPenDownRef.current = false;
      redraw();
    }).then((unlisten) => unlisteners.push(unlisten));

    void listen("penultimate:clear-annotation", () => {
      toolEngineRef.current.clear();
      workspaceRef.current.transientAnnotations.length = 0;
      lastPenDownRef.current = false;
      context.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    }).then((unlisten) => unlisteners.push(unlisten));

    void listen("penultimate:undo-annotation", () => {
      if (toolEngineRef.current.undo()) redraw();
    }).then((unlisten) => unlisteners.push(unlisten));

    void listen("penultimate:redo-annotation", () => {
      if (toolEngineRef.current.redo()) redraw();
    }).then((unlisten) => unlisteners.push(unlisten));

    void listen<AnnotationTool>("penultimate:select-annotation-tool", (event) => {
      toolEngineRef.current.select(event.payload);
      lastPenDownRef.current = false;
      redraw();
    }).then((unlisten) => unlisteners.push(unlisten));

    void listen<boolean>("penultimate:set-constrain-proportions", (event) => {
      constrainProportionsRef.current = event.payload;
    }).then((unlisten) => unlisteners.push(unlisten));

    void listen<{
      tool: AnnotationTool;
      style: AnnotationStyle;
      text: string;
    }>("penultimate:annotation-settings", (event) => {
      activeStyleRef.current = event.payload.style;
      annotationTextRef.current = event.payload.text;
    }).then((unlisten) => unlisteners.push(unlisten));

    let laserAnimationFrame = 0;
    const animateLaser = () => {
      const transient = workspaceRef.current.transientAnnotations;
      const now = Date.now();
      const next = transient.filter((annotation) => annotation.expiresAt > now);
      if (next.length !== transient.length) {
        transient.splice(0, transient.length, ...next);
      }
      if (transient.length > 0) redraw();
      laserAnimationFrame = window.requestAnimationFrame(animateLaser);
    };
    laserAnimationFrame = window.requestAnimationFrame(animateLaser);

    const updateShiftConstraint = (event: KeyboardEvent) => {
      if (event.key === "Escape" && trackpadDrawingRef.current) {
        event.preventDefault();
        event.stopImmediatePropagation();
        void invoke("set_runtime_flags", {
          payload: { trackpadDrawingEnabled: false },
        });
        return;
      }
      constrainProportionsRef.current = event.shiftKey;
    };
    window.addEventListener("keydown", updateShiftConstraint);
    window.addEventListener("keyup", updateShiftConstraint);

    const pointerEventToPenInput = (
      event: PointerEvent,
      penDown: boolean,
    ): PenInputEvent => {
      const bounds = canvas.getBoundingClientRect();
      return {
        x: (event.clientX - bounds.left) / bounds.width,
        y: (event.clientY - bounds.top) / bounds.height,
        penDown,
        pressure: event.pressure > 0 ? event.pressure : 0.55,
        timestamp: Date.now(),
      };
    };

    const onPointerDown = (event: PointerEvent) => {
      if (!trackpadDrawingRef.current) return;
      event.preventDefault();
      canvas.setPointerCapture(event.pointerId);
      applyPenInput(pointerEventToPenInput(event, true));
    };
    const onPointerMove = (event: PointerEvent) => {
      if (!trackpadDrawingRef.current || !lastPenDownRef.current) return;
      event.preventDefault();
      applyPenInput(pointerEventToPenInput(event, true));
    };
    const onPointerUp = (event: PointerEvent) => {
      if (!trackpadDrawingRef.current || !lastPenDownRef.current) return;
      event.preventDefault();
      applyPenInput(pointerEventToPenInput(event, false));
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    };
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);

    return () => {
      window.removeEventListener("resize", resize);
      window.removeEventListener("keydown", updateShiftConstraint);
      window.removeEventListener("keyup", updateShiftConstraint);
      window.cancelAnimationFrame(laserAnimationFrame);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerUp);
      for (const unlisten of unlisteners) unlisten();
    };
  }, []);

  return (
    <main
      className={`overlay-shell ${trackpadDrawing ? "overlay-interactive" : "overlay-passive"}`}
      aria-hidden="true"
    >
      <canvas ref={canvasRef} className="annotation-canvas" />
    </main>
  );
}
