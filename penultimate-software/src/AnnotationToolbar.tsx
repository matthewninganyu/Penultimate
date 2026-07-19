import { useEffect, useState } from "react";
import { emitTo, listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  ArrowUpRight,
  Brush,
  Circle,
  Eraser,
  Highlighter,
  Minus,
  MousePointer2,
  Pencil,
  PenTool,
  RectangleHorizontal,
  Redo2,
  Trash2,
  Type,
  Undo2,
  X,
} from "lucide-react";
import {
  defaultStyleForTool,
  type AnnotationStyle,
  type AnnotationTool,
} from "./annotationModel";

const OVERLAY_LABEL = "overlay";
const TOOLS: Array<{
  tool: AnnotationTool;
  label: string;
  icon: typeof PenTool;
}> = [
  { tool: "cursor", label: "Cursor", icon: MousePointer2 },
  { tool: "pen", label: "Pen", icon: PenTool },
  { tool: "pencil", label: "Pencil", icon: Pencil },
  { tool: "marker", label: "Marker", icon: Brush },
  { tool: "highlighter", label: "Highlighter", icon: Highlighter },
  { tool: "line", label: "Line", icon: Minus },
  { tool: "arrow", label: "Arrow", icon: ArrowUpRight },
  {
    tool: "rounded-rectangle",
    label: "Rounded rectangle",
    icon: RectangleHorizontal,
  },
  { tool: "ellipse", label: "Ellipse", icon: Circle },
  { tool: "text", label: "Text", icon: Type },
  { tool: "eraser", label: "Eraser", icon: Eraser },
];
const COLORS = ["#1769aa", "#20252b", "#d83b3b", "#16845b", "#ffd84d"];

export default function AnnotationToolbar() {
  const [tool, setTool] = useState<AnnotationTool>("pen");
  const [style, setStyle] = useState(() => defaultStyleForTool("pen"));
  const [text, setText] = useState("Note");
  const [fill, setFill] = useState(true);

  const emitToolSettings = (
    nextTool: AnnotationTool,
    nextStyle: AnnotationStyle,
    nextText = text,
    nextFill = fill,
  ) => {
    void emitTo(OVERLAY_LABEL, "penultimate:annotation-settings", {
      tool: nextTool,
      style: {
        ...nextStyle,
        fillColor: nextFill ? nextStyle.color : undefined,
        fillOpacity: nextFill ? (nextStyle.fillOpacity ?? 0.08) : undefined,
      },
      text: nextText,
    });
  };

  const selectTool = (nextTool: AnnotationTool) => {
    const nextStyle = defaultStyleForTool(nextTool);
    setTool(nextTool);
    setStyle(nextStyle);
    setFill(Boolean(nextStyle.fillColor));
    void emitTo(OVERLAY_LABEL, "penultimate:select-annotation-tool", nextTool);
    emitToolSettings(nextTool, nextStyle, text, Boolean(nextStyle.fillColor));
  };

  const updateStyle = (patch: Partial<AnnotationStyle>) => {
    const nextStyle = { ...style, ...patch };
    setStyle(nextStyle);
    emitToolSettings(tool, nextStyle);
  };

  const closeAnnotation = async () => {
    await invoke("set_runtime_flags", {
      payload: {
        overlayEnabled: false,
        trackpadDrawingEnabled: false,
      },
    });
  };

  useEffect(() => {
    const unlisteners: Array<() => void> = [];
    void listen<boolean>("penultimate:set-toolbar-visible", async (event) => {
      const toolbar = getCurrentWindow();
      if (event.payload) {
        await toolbar.show();
      } else {
        await toolbar.hide();
      }
    }).then((unlisten) => unlisteners.push(unlisten));

    emitToolSettings(tool, style);

    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing = target?.tagName === "INPUT";
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        void emitTo(
          OVERLAY_LABEL,
          event.shiftKey
            ? "penultimate:redo-annotation"
            : "penultimate:undo-annotation",
        );
        return;
      }
      if (typing) return;
      const shortcuts: Partial<Record<string, AnnotationTool>> = {
        v: "cursor",
        p: "pen",
        q: "pencil",
        m: "marker",
        h: "highlighter",
        l: "line",
        a: "arrow",
        r: "rounded-rectangle",
        o: "ellipse",
        t: "text",
        e: "eraser",
      };
      const nextTool = shortcuts[event.key.toLowerCase()];
      if (nextTool) selectTool(nextTool);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      for (const unlisten of unlisteners) unlisten();
    };
  }, []);

  return (
    <main className="annotation-toolbar" data-tauri-drag-region>
      <div className="tool-strip" aria-label="Annotation tools">
        {TOOLS.map(({ tool: option, label, icon: Icon }) => (
          <button
            key={option}
            className={tool === option ? "tool-button selected" : "tool-button"}
            aria-label={label}
            title={label}
            onClick={() => selectTool(option)}
          >
            <Icon size={18} strokeWidth={1.8} />
          </button>
        ))}
        <span className="toolbar-divider" />
        <button
          className="tool-button"
          aria-label="Undo"
          title="Undo"
          onClick={() => void emitTo(OVERLAY_LABEL, "penultimate:undo-annotation")}
        >
          <Undo2 size={18} />
        </button>
        <button
          className="tool-button"
          aria-label="Redo"
          title="Redo"
          onClick={() => void emitTo(OVERLAY_LABEL, "penultimate:redo-annotation")}
        >
          <Redo2 size={18} />
        </button>
        <button
          className="tool-button danger"
          aria-label="Clear annotation"
          title="Clear annotation"
          onClick={() => void emitTo(OVERLAY_LABEL, "penultimate:clear-annotation")}
        >
          <Trash2 size={18} />
        </button>
        <button
          className="tool-button close-annotation-button"
          aria-label="Close annotation"
          title="Close annotation"
          onClick={() => void closeAnnotation()}
        >
          <X size={18} />
        </button>
      </div>

      <div className="tool-options">
        <div className="color-options" aria-label="Ink color">
          {COLORS.map((color) => (
            <button
              key={color}
              className={style.color === color ? "color-swatch selected" : "color-swatch"}
              style={{ backgroundColor: color }}
              aria-label={`Color ${color}`}
              onClick={() => updateStyle({ color })}
            />
          ))}
        </div>
        <label className="toolbar-slider">
          <span>Size</span>
          <input
            type="range"
            min="1"
            max={tool === "highlighter" ? "36" : "20"}
            step="0.5"
            value={style.width}
            onChange={(event) => updateStyle({ width: Number(event.currentTarget.value) })}
          />
        </label>
        <label className="toolbar-slider">
          <span>Opacity</span>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={style.opacity}
            onChange={(event) => updateStyle({ opacity: Number(event.currentTarget.value) })}
          />
        </label>
        {(tool === "rounded-rectangle" || tool === "ellipse") && (
          <label className="fill-toggle">
            <input
              type="checkbox"
              checked={fill}
              onChange={(event) => {
                const next = event.currentTarget.checked;
                setFill(next);
                emitToolSettings(tool, style, text, next);
              }}
            />
            <span>Fill</span>
          </label>
        )}
        {tool === "text" && (
          <input
            className="annotation-text-input"
            value={text}
            aria-label="Text to place"
            onChange={(event) => {
              const next = event.currentTarget.value;
              setText(next);
              emitToolSettings(tool, style, next);
            }}
          />
        )}
      </div>
    </main>
  );
}
