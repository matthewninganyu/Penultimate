import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";

type RuntimeSnapshot = {
  connected: boolean;
  cursorActive: boolean;
};

type HudState = "connected" | "disconnected";

const POLL_MS = 50;
const STABLE_MS = 400;

export default function StatusHud() {
  const [hudState, setHudState] = useState<HudState | null>(null);
  const [leaving, setLeaving] = useState(false);
  const candidateRef = useRef<string | null>(null);
  const candidateSinceRef = useRef(0);
  const stableRef = useRef<string | null>(null);
  const hideTimerRef = useRef<number | null>(null);
  const exitTimerRef = useRef<number | null>(null);

  useEffect(() => {
    const hudWindow = getCurrentWindow();
    let cancelled = false;

    void hudWindow.setIgnoreCursorEvents(true);
    void hudWindow.setFocusable(false);

    const hideLater = (delay: number) => {
      if (hideTimerRef.current !== null) window.clearTimeout(hideTimerRef.current);
      hideTimerRef.current = window.setTimeout(() => {
        setLeaving(true);
        exitTimerRef.current = window.setTimeout(() => {
          setHudState(null);
          setLeaving(false);
          void hudWindow.hide();
        }, 170);
      }, delay);
    };

    const present = (next: HudState, duration?: number) => {
      if (hideTimerRef.current !== null) {
        window.clearTimeout(hideTimerRef.current);
        hideTimerRef.current = null;
      }
      if (exitTimerRef.current !== null) {
        window.clearTimeout(exitTimerRef.current);
        exitTimerRef.current = null;
      }
      setLeaving(false);
      setHudState(next);
      void hudWindow.show();
      if (duration) hideLater(duration);
    };

    const poll = async () => {
      const snapshot = await invoke<RuntimeSnapshot>("get_runtime_snapshot");
      if (cancelled) return;
      const next = snapshot.connected ? "connected" : "disconnected";
      const now = Date.now();

      if (next === "connected") {
        candidateRef.current = next;
        candidateSinceRef.current = now;
        if (stableRef.current !== next) {
          stableRef.current = next;
          present("connected", 1_500);
        }
        return;
      }

      if (candidateRef.current !== next) {
        candidateRef.current = next;
        candidateSinceRef.current = now;
        return;
      }
      if (now - candidateSinceRef.current < STABLE_MS || stableRef.current === next) return;

      stableRef.current = next;
      present("disconnected");
    };

    void poll();
    const pollTimer = window.setInterval(() => void poll(), POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(pollTimer);
      if (hideTimerRef.current !== null) window.clearTimeout(hideTimerRef.current);
      if (exitTimerRef.current !== null) window.clearTimeout(exitTimerRef.current);
    };
  }, []);

  if (!hudState) return null;

  const labels: Record<HudState, string> = {
    connected: "Pen connected",
    disconnected: "Pen disconnected",
  };

  return (
    <main
      key={hudState}
      className={`status-hud status-${hudState} ${leaving ? "is-leaving" : ""}`}
      role="status"
      aria-live="polite"
    >
      <span className="status-hud-dot" aria-hidden="true" />
      <span>{labels[hudState]}</span>
    </main>
  );
}
