import type {
  AnnotationDocument,
  AnnotationPoint,
  FreehandAnnotation,
  LaserAnnotation,
  ShapeAnnotation,
} from "./annotationModel";

type CanvasSize = { width: number; height: number };

function screenPoint(point: AnnotationPoint, size: CanvasSize) {
  return { x: point.x * size.width, y: point.y * size.height };
}

function constrainedEnd(annotation: ShapeAnnotation, size: CanvasSize) {
  const start = screenPoint(annotation.start, size);
  const end = screenPoint(annotation.end, size);
  if (!annotation.constrainProportions) return end;

  const deltaX = end.x - start.x;
  const deltaY = end.y - start.y;
  if (annotation.tool === "line" || annotation.tool === "arrow") {
    const distance = Math.hypot(deltaX, deltaY);
    const angle = Math.atan2(deltaY, deltaX);
    const snappedAngle = Math.round(angle / (Math.PI / 4)) * (Math.PI / 4);
    return {
      x: start.x + Math.cos(snappedAngle) * distance,
      y: start.y + Math.sin(snappedAngle) * distance,
    };
  }

  const side = Math.max(Math.abs(deltaX), Math.abs(deltaY));
  return {
    x: start.x + Math.sign(deltaX || 1) * side,
    y: start.y + Math.sign(deltaY || 1) * side,
  };
}

function roundedRectanglePath(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const clampedRadius = Math.min(radius, Math.abs(width) / 2, Math.abs(height) / 2);
  const right = x + width;
  const bottom = y + height;
  const left = Math.min(x, right);
  const top = Math.min(y, bottom);
  const normalizedRight = Math.max(x, right);
  const normalizedBottom = Math.max(y, bottom);

  context.moveTo(left + clampedRadius, top);
  context.lineTo(normalizedRight - clampedRadius, top);
  context.quadraticCurveTo(normalizedRight, top, normalizedRight, top + clampedRadius);
  context.lineTo(normalizedRight, normalizedBottom - clampedRadius);
  context.quadraticCurveTo(
    normalizedRight,
    normalizedBottom,
    normalizedRight - clampedRadius,
    normalizedBottom,
  );
  context.lineTo(left + clampedRadius, normalizedBottom);
  context.quadraticCurveTo(left, normalizedBottom, left, normalizedBottom - clampedRadius);
  context.lineTo(left, top + clampedRadius);
  context.quadraticCurveTo(left, top, left + clampedRadius, top);
  context.closePath();
}

function applyShapePaint(
  context: CanvasRenderingContext2D,
  annotation: ShapeAnnotation,
) {
  if (annotation.style.fillColor && (annotation.style.fillOpacity ?? 0) > 0) {
    context.save();
    context.fillStyle = annotation.style.fillColor;
    context.globalAlpha = annotation.style.fillOpacity ?? 0;
    context.fill();
    context.restore();
  }
  context.strokeStyle = annotation.style.color;
  context.globalAlpha = annotation.style.opacity;
  context.lineWidth = annotation.style.width;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.stroke();
}

export function drawShapeAnnotation(
  context: CanvasRenderingContext2D,
  annotation: ShapeAnnotation,
  size: CanvasSize,
) {
  const start = screenPoint(annotation.start, size);
  const end = constrainedEnd(annotation, size);
  const width = end.x - start.x;
  const height = end.y - start.y;

  context.save();
  context.beginPath();
  if (annotation.tool === "line" || annotation.tool === "arrow") {
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
  } else if (annotation.tool === "rounded-rectangle") {
    roundedRectanglePath(
      context,
      start.x,
      start.y,
      width,
      height,
      annotation.cornerRadius ?? 12,
    );
  } else {
    context.ellipse(
      start.x + width / 2,
      start.y + height / 2,
      Math.abs(width / 2),
      Math.abs(height / 2),
      0,
      0,
      Math.PI * 2,
    );
  }
  applyShapePaint(context, annotation);

  if (annotation.tool === "arrow") {
    const angle = Math.atan2(end.y - start.y, end.x - start.x);
    const headLength = Math.min(18, Math.max(8, Math.hypot(width, height) * 0.22));
    context.beginPath();
    context.moveTo(end.x, end.y);
    context.lineTo(
      end.x - Math.cos(angle - Math.PI / 6) * headLength,
      end.y - Math.sin(angle - Math.PI / 6) * headLength,
    );
    context.moveTo(end.x, end.y);
    context.lineTo(
      end.x - Math.cos(angle + Math.PI / 6) * headLength,
      end.y - Math.sin(angle + Math.PI / 6) * headLength,
    );
    applyShapePaint(context, annotation);
  }
  context.restore();
}

function pressureScale(annotation: FreehandAnnotation, pressure: number) {
  if (annotation.tool === "highlighter") return 1;
  if (annotation.tool === "marker") return 0.82 + pressure * 0.3;
  if (annotation.tool === "pencil") return 0.42 + pressure * 0.72;
  return 0.45 + pressure;
}

export function drawFreehandSegment(
  context: CanvasRenderingContext2D,
  annotation: FreehandAnnotation,
  from: AnnotationPoint,
  to: AnnotationPoint,
  size: CanvasSize,
  previous?: AnnotationPoint,
  finishAtPoint = false,
) {
  const fromScreen = screenPoint(from, size);
  const toScreen = screenPoint(to, size);
  const previousScreen = previous ? screenPoint(previous, size) : null;
  const start = previousScreen
    ? {
        x: (previousScreen.x + fromScreen.x) / 2,
        y: (previousScreen.y + fromScreen.y) / 2,
      }
    : fromScreen;
  const end = finishAtPoint
    ? toScreen
    : {
        x: (fromScreen.x + toScreen.x) / 2,
        y: (fromScreen.y + toScreen.y) / 2,
      };
  const pressure = Math.max(0.1, Math.min(1, to.pressure));

  context.save();
  context.beginPath();
  context.moveTo(start.x, start.y);
  context.quadraticCurveTo(
    fromScreen.x,
    fromScreen.y,
    end.x,
    end.y,
  );
  context.strokeStyle = annotation.style.color;
  context.globalAlpha = annotation.style.opacity;
  context.lineWidth = Math.max(
    0.5,
    annotation.style.width * pressureScale(annotation, pressure),
  );
  context.lineCap = annotation.tool === "highlighter" ? "square" : "round";
  context.lineJoin = "round";
  context.stroke();

  if (annotation.tool === "pencil") {
    const grainOffset = ((to.timestamp % 7) - 3) * 0.08;
    context.beginPath();
    context.moveTo(start.x, start.y + grainOffset);
    context.lineTo(end.x, end.y + grainOffset);
    context.globalAlpha = annotation.style.opacity * 0.26;
    context.lineWidth = Math.max(0.35, context.lineWidth * 0.38);
    context.stroke();
  }
  context.restore();
}

export function renderAnnotationDocument(
  context: CanvasRenderingContext2D,
  document: AnnotationDocument,
  size: CanvasSize,
  options: { renderText?: boolean } = {},
) {
  context.clearRect(0, 0, size.width, size.height);
  for (const annotation of document.annotations) {
    if (annotation.kind === "eraser") {
      context.save();
      context.globalCompositeOperation = "destination-out";
      context.lineWidth = annotation.style.width;
      context.lineCap = "round";
      context.lineJoin = "round";
      context.strokeStyle = "#000000";
      context.fillStyle = "#000000";
      if (annotation.points.length === 1) {
        const point = screenPoint(annotation.points[0], size);
        context.beginPath();
        context.arc(point.x, point.y, annotation.style.width / 2, 0, Math.PI * 2);
        context.fill();
      } else {
        context.beginPath();
        annotation.points.forEach((point, index) => {
          const screen = screenPoint(point, size);
          if (index === 0) context.moveTo(screen.x, screen.y);
          else context.lineTo(screen.x, screen.y);
        });
        context.stroke();
      }
      context.restore();
    } else if (annotation.kind === "freehand") {
      for (let index = 1; index < annotation.points.length; index += 1) {
        drawFreehandSegment(
          context,
          annotation,
          annotation.points[index - 1],
          annotation.points[index],
          size,
          annotation.points[index - 2],
          index === annotation.points.length - 1,
        );
      }
    } else if (annotation.kind === "shape") {
      drawShapeAnnotation(context, annotation, size);
    } else if (annotation.kind === "text" && options.renderText !== false) {
      const position = screenPoint(annotation.position, size);
      context.save();
      context.fillStyle = annotation.style.color;
      context.globalAlpha = annotation.style.opacity;
      context.font = `${annotation.fontSize}px "SF Pro Text", sans-serif`;
      context.textBaseline = "top";
      annotation.text.split("\n").forEach((line, index) => {
        context.fillText(line, position.x, position.y + index * annotation.fontSize * 1.3);
      });
      context.restore();
    }
  }
}

export function drawLaserAnnotations(
  context: CanvasRenderingContext2D,
  annotations: LaserAnnotation[],
  size: CanvasSize,
  now: number,
) {
  for (const annotation of annotations) {
    const remaining = Math.max(0, annotation.expiresAt - now);
    const fade = Math.min(1, remaining / 500);
    for (let index = 1; index < annotation.points.length; index += 1) {
      context.save();
      context.globalAlpha = annotation.style.opacity * fade;
      context.strokeStyle = annotation.style.color;
      context.lineWidth = annotation.style.width;
      context.lineCap = "round";
      context.beginPath();
      const start = screenPoint(annotation.points[index - 1], size);
      const end = screenPoint(annotation.points[index], size);
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
      context.restore();
    }
  }
}
