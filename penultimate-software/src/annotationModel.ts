export type PenInputEvent = {
  x: number;
  y: number;
  penDown: boolean;
  pressure?: number;
  timestamp: number;
};

export type AnnotationPoint = {
  x: number;
  y: number;
  pressure: number;
  timestamp: number;
};

export type AnnotationStyle = {
  color: string;
  width: number;
  opacity: number;
  fillColor?: string;
  fillOpacity?: number;
};

type AnnotationBase = {
  id: string;
  createdAt: number;
  style: AnnotationStyle;
};

export type FreehandTool = "pen" | "pencil" | "marker" | "highlighter";
export type ShapeTool = "line" | "arrow" | "rounded-rectangle" | "ellipse";
export type AnnotationTool =
  | "cursor"
  | FreehandTool
  | ShapeTool
  | "text"
  | "eraser"
  | "laser";

export type EraserMode = "stroke" | "pixel";

export type FreehandAnnotation = AnnotationBase & {
  kind: "freehand";
  tool: FreehandTool;
  points: AnnotationPoint[];
};

export type ShapeAnnotation = AnnotationBase & {
  kind: "shape";
  tool: ShapeTool;
  start: AnnotationPoint;
  end: AnnotationPoint;
  cornerRadius?: number;
  constrainProportions?: boolean;
};

export type TextAnnotation = AnnotationBase & {
  kind: "text";
  position: AnnotationPoint;
  text: string;
  fontSize: number;
};

export type EraserAnnotation = AnnotationBase & {
  kind: "eraser";
  points: AnnotationPoint[];
};

export type PersistentAnnotation =
  | FreehandAnnotation
  | ShapeAnnotation
  | TextAnnotation
  | EraserAnnotation;

export type LaserAnnotation = AnnotationBase & {
  kind: "laser";
  points: AnnotationPoint[];
  expiresAt: number;
};

export type AnnotationDocument = {
  version: 1;
  annotations: PersistentAnnotation[];
};

export type AnnotationWorkspace = {
  document: AnnotationDocument;
  transientAnnotations: LaserAnnotation[];
};

export const DEFAULT_PEN_STYLE: AnnotationStyle = {
  color: "#1769aa",
  width: 3.5,
  opacity: 1,
};

const DEFAULT_TOOL_STYLES: Record<AnnotationTool, AnnotationStyle> = {
  cursor: DEFAULT_PEN_STYLE,
  pen: DEFAULT_PEN_STYLE,
  pencil: { color: "#30343a", width: 1.7, opacity: 0.72 },
  marker: { color: "#1769aa", width: 8, opacity: 0.92 },
  highlighter: { color: "#ffd84d", width: 18, opacity: 0.32 },
  line: DEFAULT_PEN_STYLE,
  arrow: DEFAULT_PEN_STYLE,
  "rounded-rectangle": {
    ...DEFAULT_PEN_STYLE,
    fillColor: "#1769aa",
    fillOpacity: 0.08,
  },
  ellipse: {
    ...DEFAULT_PEN_STYLE,
    fillColor: "#1769aa",
    fillOpacity: 0.08,
  },
  text: { color: "#20252b", width: 1, opacity: 1 },
  eraser: { color: "#000000", width: 18, opacity: 1 },
  laser: { color: "#e23b3b", width: 5, opacity: 0.9 },
};

export function defaultStyleForTool(tool: AnnotationTool): AnnotationStyle {
  return { ...DEFAULT_TOOL_STYLES[tool] };
}

export function createAnnotationDocument(): AnnotationDocument {
  return { version: 1, annotations: [] };
}

export function createAnnotationWorkspace(): AnnotationWorkspace {
  return {
    document: createAnnotationDocument(),
    transientAnnotations: [],
  };
}

export function pointFromPenEvent(event: PenInputEvent): AnnotationPoint {
  return {
    x: Math.max(0, Math.min(1, event.x)),
    y: Math.max(0, Math.min(1, event.y)),
    pressure: Math.max(0, Math.min(1, event.pressure ?? 0.55)),
    timestamp: event.timestamp,
  };
}

export function createFreehandAnnotation(
  tool: FreehandTool,
  point: AnnotationPoint,
  style: AnnotationStyle,
): FreehandAnnotation {
  return {
    id: globalThis.crypto?.randomUUID?.() ??
      `annotation-${point.timestamp}-${Math.random().toString(16).slice(2)}`,
    kind: "freehand",
    tool,
    points: [point],
    style: { ...style },
    createdAt: point.timestamp,
  };
}
