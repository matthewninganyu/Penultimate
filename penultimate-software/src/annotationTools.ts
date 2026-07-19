import {
  createFreehandAnnotation,
  type AnnotationPoint,
  type AnnotationStyle,
  type AnnotationTool,
  type AnnotationWorkspace,
  type EraserAnnotation,
  type EraserMode,
  type FreehandAnnotation,
  type FreehandTool,
  type LaserAnnotation,
  type PersistentAnnotation,
  type ShapeAnnotation,
  type ShapeTool,
  type TextAnnotation,
} from "./annotationModel";
import { AnnotationHistory } from "./annotationHistory";

export type ToolPhase = "idle" | "drawing";

export interface AnnotationToolController {
  readonly tool: AnnotationTool;
  readonly phase: ToolPhase;
  begin(point: AnnotationPoint): void;
  update(point: AnnotationPoint): void;
  finish(point: AnnotationPoint): void;
  cancel(): void;
}

type ControllerOptions = {
  workspace: AnnotationWorkspace;
  getStyle: (tool: AnnotationTool) => AnnotationStyle;
  getConstrainProportions?: () => boolean;
  getText?: () => string;
  getEraserMode?: () => EraserMode;
};

function createId(point: AnnotationPoint) {
  return globalThis.crypto?.randomUUID?.() ??
    `annotation-${point.timestamp}-${Math.random().toString(16).slice(2)}`;
}

abstract class BaseController implements AnnotationToolController {
  abstract readonly tool: AnnotationTool;
  protected readonly workspace: AnnotationWorkspace;
  protected readonly getStyle: (tool: AnnotationTool) => AnnotationStyle;
  protected readonly getConstrainProportions: () => boolean;
  protected readonly getText: () => string;
  protected readonly getEraserMode: () => EraserMode;
  protected active = false;

  constructor(options: ControllerOptions) {
    this.workspace = options.workspace;
    this.getStyle = options.getStyle;
    this.getConstrainProportions =
      options.getConstrainProportions ?? (() => false);
    this.getText = options.getText ?? (() => "Text");
    this.getEraserMode = options.getEraserMode ?? (() => "stroke");
  }

  get phase(): ToolPhase {
    return this.active ? "drawing" : "idle";
  }

  abstract begin(point: AnnotationPoint): void;
  abstract update(point: AnnotationPoint): void;
  abstract finish(point: AnnotationPoint): void;
  abstract cancel(): void;
}

class FreehandController extends BaseController {
  readonly tool: FreehandTool;
  private draft: FreehandAnnotation | null = null;

  constructor(tool: FreehandTool, options: ControllerOptions) {
    super(options);
    this.tool = tool;
  }

  begin(point: AnnotationPoint) {
    this.cancel();
    this.draft = createFreehandAnnotation(
      this.tool,
      point,
      this.getStyle(this.tool),
    );
    this.workspace.document.annotations.push(this.draft);
    this.active = true;
  }

  update(point: AnnotationPoint) {
    if (this.draft) this.draft.points.push(point);
  }

  finish(point: AnnotationPoint) {
    if (!this.draft) return;
    const last = this.draft.points[this.draft.points.length - 1];
    if (last.timestamp !== point.timestamp) this.draft.points.push(point);
    this.draft = null;
    this.active = false;
  }

  cancel() {
    if (this.draft) removeAnnotation(this.workspace, this.draft);
    this.draft = null;
    this.active = false;
  }
}

class ShapeController extends BaseController {
  readonly tool: ShapeTool;
  private draft: ShapeAnnotation | null = null;

  constructor(tool: ShapeTool, options: ControllerOptions) {
    super(options);
    this.tool = tool;
  }

  begin(point: AnnotationPoint) {
    this.cancel();
    this.draft = {
      id: createId(point),
      kind: "shape",
      tool: this.tool,
      start: point,
      end: point,
      cornerRadius: this.tool === "rounded-rectangle" ? 12 : undefined,
      constrainProportions: this.getConstrainProportions(),
      style: { ...this.getStyle(this.tool) },
      createdAt: point.timestamp,
    };
    this.workspace.document.annotations.push(this.draft);
    this.active = true;
  }

  update(point: AnnotationPoint) {
    if (this.draft) {
      this.draft.end = point;
      this.draft.constrainProportions = this.getConstrainProportions();
    }
  }

  finish(point: AnnotationPoint) {
    if (!this.draft) return;
    this.draft.end = point;
    this.draft.constrainProportions = this.getConstrainProportions();
    this.draft = null;
    this.active = false;
  }

  cancel() {
    if (this.draft) removeAnnotation(this.workspace, this.draft);
    this.draft = null;
    this.active = false;
  }
}

class TextController extends BaseController {
  readonly tool = "text" as const;
  private draft: TextAnnotation | null = null;

  begin(point: AnnotationPoint) {
    this.cancel();
    this.draft = {
      id: createId(point),
      kind: "text",
      position: point,
      text: this.getText(),
      fontSize: 18,
      style: { ...this.getStyle(this.tool) },
      createdAt: point.timestamp,
    };
    this.active = true;
  }

  update() {}

  finish() {
    if (this.draft) {
      this.workspace.document.annotations.push(this.draft);
    }
    this.draft = null;
    this.active = false;
  }

  cancel() {
    this.draft = null;
    this.active = false;
  }
}

class CursorController extends BaseController {
  readonly tool = "cursor" as const;

  begin() {}
  update() {}
  finish() {}
  cancel() {
    this.active = false;
  }
}

class EraserController extends BaseController {
  readonly tool = "eraser" as const;
  private removed: Array<{ annotation: PersistentAnnotation; index: number }> = [];
  private draft: EraserAnnotation | null = null;

  begin(point: AnnotationPoint) {
    this.removed = [];
    this.active = true;
    if (this.getEraserMode() === "pixel") {
      this.draft = {
        id: createId(point),
        kind: "eraser",
        points: [point],
        style: { ...this.getStyle(this.tool) },
        createdAt: point.timestamp,
      };
      this.workspace.document.annotations.push(this.draft);
    } else {
      this.eraseAt(point);
    }
  }

  update(point: AnnotationPoint) {
    if (!this.active) return;
    if (this.draft) this.draft.points.push(point);
    else this.eraseAt(point);
  }

  finish(point: AnnotationPoint) {
    if (this.draft) {
      const last = this.draft.points[this.draft.points.length - 1];
      if (last.timestamp !== point.timestamp) this.draft.points.push(point);
    } else {
      this.eraseAt(point);
    }
    this.draft = null;
    this.removed = [];
    this.active = false;
  }

  cancel() {
    if (this.draft) removeAnnotation(this.workspace, this.draft);
    for (const { annotation, index } of this.removed.sort(
      (left, right) => left.index - right.index,
    )) {
      this.workspace.document.annotations.splice(index, 0, annotation);
    }
    this.draft = null;
    this.removed = [];
    this.active = false;
  }

  private eraseAt(point: AnnotationPoint) {
    for (
      let index = this.workspace.document.annotations.length - 1;
      index >= 0;
      index -= 1
    ) {
      const annotation = this.workspace.document.annotations[index];
      if (!annotationHitTest(annotation, point)) continue;
      this.removed.push({ annotation, index });
      this.workspace.document.annotations.splice(index, 1);
    }
  }
}

class LaserController extends BaseController {
  readonly tool = "laser" as const;
  private draft: LaserAnnotation | null = null;

  begin(point: AnnotationPoint) {
    this.cancel();
    this.draft = {
      id: createId(point),
      kind: "laser",
      points: [point],
      expiresAt: Date.now() + 900,
      style: { ...this.getStyle(this.tool) },
      createdAt: point.timestamp,
    };
    this.workspace.transientAnnotations.push(this.draft);
    this.active = true;
  }

  update(point: AnnotationPoint) {
    if (!this.draft) return;
    this.draft.points.push(point);
    this.draft.expiresAt = Date.now() + 900;
  }

  finish(point: AnnotationPoint) {
    if (!this.draft) return;
    this.update(point);
    this.draft = null;
    this.active = false;
  }

  cancel() {
    if (this.draft) {
      const index = this.workspace.transientAnnotations.indexOf(this.draft);
      if (index >= 0) this.workspace.transientAnnotations.splice(index, 1);
    }
    this.draft = null;
    this.active = false;
  }
}

function removeAnnotation(
  workspace: AnnotationWorkspace,
  annotation: PersistentAnnotation,
) {
  const index = workspace.document.annotations.indexOf(annotation);
  if (index >= 0) workspace.document.annotations.splice(index, 1);
}

function pointDistance(left: AnnotationPoint, right: AnnotationPoint) {
  return Math.hypot(left.x - right.x, left.y - right.y);
}

function segmentDistance(
  point: AnnotationPoint,
  start: AnnotationPoint,
  end: AnnotationPoint,
) {
  const deltaX = end.x - start.x;
  const deltaY = end.y - start.y;
  const lengthSquared = deltaX * deltaX + deltaY * deltaY;
  if (lengthSquared === 0) return pointDistance(point, start);
  const projection = Math.max(
    0,
    Math.min(
      1,
      ((point.x - start.x) * deltaX + (point.y - start.y) * deltaY) /
        lengthSquared,
    ),
  );
  return Math.hypot(
    point.x - (start.x + projection * deltaX),
    point.y - (start.y + projection * deltaY),
  );
}

function annotationHitTest(
  annotation: PersistentAnnotation,
  point: AnnotationPoint,
) {
  const radius = 0.018;
  if (annotation.kind === "text") {
    return pointDistance(annotation.position, point) <= 0.045;
  }
  if (annotation.kind === "freehand") {
    return annotation.points.some((candidate, index) => {
      if (index === 0) return pointDistance(candidate, point) <= radius;
      return segmentDistance(point, annotation.points[index - 1], candidate) <= radius;
    });
  }
  if (annotation.kind === "eraser") return false;
  if (
    annotation.tool === "rounded-rectangle" ||
    annotation.tool === "ellipse"
  ) {
    const left = Math.min(annotation.start.x, annotation.end.x) - radius;
    const right = Math.max(annotation.start.x, annotation.end.x) + radius;
    const top = Math.min(annotation.start.y, annotation.end.y) - radius;
    const bottom = Math.max(annotation.start.y, annotation.end.y) + radius;
    return point.x >= left && point.x <= right && point.y >= top && point.y <= bottom;
  }
  return segmentDistance(point, annotation.start, annotation.end) <= radius;
}

export function createToolControllers(options: ControllerOptions) {
  const controllers = new Map<AnnotationTool, AnnotationToolController>();
  controllers.set("cursor", new CursorController(options));
  const freehandTools: FreehandTool[] = [
    "pen",
    "pencil",
    "marker",
    "highlighter",
  ];
  const shapeTools: ShapeTool[] = [
    "line",
    "arrow",
    "rounded-rectangle",
    "ellipse",
  ];

  for (const tool of freehandTools) {
    controllers.set(tool, new FreehandController(tool, options));
  }
  for (const tool of shapeTools) {
    controllers.set(tool, new ShapeController(tool, options));
  }
  controllers.set("text", new TextController(options));
  controllers.set("eraser", new EraserController(options));
  controllers.set("laser", new LaserController(options));
  return controllers;
}

export class AnnotationToolEngine {
  private readonly controllers: Map<AnnotationTool, AnnotationToolController>;
  private readonly workspace: AnnotationWorkspace;
  readonly history: AnnotationHistory;
  private activeTool: AnnotationTool;

  constructor(
    options: ControllerOptions,
    initialTool: AnnotationTool = "pen",
    history = new AnnotationHistory(),
  ) {
    this.controllers = createToolControllers(options);
    this.workspace = options.workspace;
    this.history = history;
    this.activeTool = initialTool;
  }

  get tool() {
    return this.activeTool;
  }

  get controller() {
    return this.controllers.get(this.activeTool)!;
  }

  select(tool: AnnotationTool) {
    if (tool === this.activeTool) return;
    this.controller.cancel();
    this.history.cancel();
    this.activeTool = tool;
  }

  begin(point: AnnotationPoint) {
    this.history.begin(this.workspace.document);
    this.controller.begin(point);
  }

  update(point: AnnotationPoint) {
    this.controller.update(point);
  }

  finish(point: AnnotationPoint) {
    this.controller.finish(point);
    this.history.commit(this.workspace.document);
  }

  cancel() {
    this.controller.cancel();
    this.history.cancel();
  }

  clear() {
    this.cancel();
    if (this.workspace.document.annotations.length === 0) return;
    this.history.begin(this.workspace.document);
    this.workspace.document.annotations.length = 0;
    this.history.commit(this.workspace.document);
  }

  undo() {
    const cancelledDraft = this.history.transactionActive;
    this.controller.cancel();
    if (cancelledDraft) {
      this.history.cancel();
      return true;
    }
    return this.history.undo(this.workspace.document);
  }

  redo() {
    this.controller.cancel();
    return this.history.redo(this.workspace.document);
  }
}
