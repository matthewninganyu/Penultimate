use enigo::{Coordinate, Enigo, Mouse, Settings};
use serde::{Deserialize, Serialize};
use std::{
    net::{SocketAddr, UdpSocket},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconBuilder,
    AppHandle, Emitter, Manager, State,
};

const TRACKPAD_OVERRIDE_DISTANCE_PX: i32 = 12;
const TRACKPAD_OVERRIDE_MS: u64 = 1_200;
const UDP_BIND_ADDRESS: &str = "0.0.0.0:4242";
const UDP_DISCONNECT_MS: u64 = 500;
const MAX_UDP_PACKET_BYTES: usize = 2_048;
// A backward sequence jump larger than this is treated as the sender restarting
// its counter (Pi reboot/reconnect) rather than a late/duplicate packet, so a
// reset without a 500 ms silence gap does not lock out the feed permanently.
const UDP_SEQUENCE_RESET_GAP: u64 = 1_000;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeSnapshot {
    overlay_enabled: bool,
    cursor_active: bool,
    trackpad_override_enabled: bool,
    trackpad_drawing_enabled: bool,
    simulated: bool,
    rate: f64,
    applied_rate: f64,
    move_latency_ms: String,
    connected: bool,
    latency_ms: String,
    source: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PenInputEvent {
    x: f64,
    y: f64,
    pen_down: bool,
    pressure: Option<f64>,
    timestamp: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UdpPenPacket {
    sequence: Option<u64>,
    #[serde(alias = "normalized_x")]
    x: f64,
    #[serde(alias = "normalized_y")]
    y: f64,
    #[serde(alias = "pen_down", alias = "touching")]
    pen_down: bool,
    #[serde(alias = "contact_confidence")]
    pressure: Option<f64>,
    timestamp: Option<f64>,
    valid: Option<bool>,
    #[serde(alias = "tracking_confidence")]
    tracking_confidence: Option<f64>,
}

#[derive(Debug)]
struct RuntimeState {
    overlay_enabled: bool,
    cursor_active: bool,
    trackpad_override_enabled: bool,
    trackpad_drawing_enabled: bool,
    simulated: bool,
    rate: f64,
    applied_rate: f64,
    move_latency_ms: String,
    connected: bool,
    latency_ms: String,
    source: String,
    last_udp_received: Option<Instant>,
    last_udp_sequence: Option<u64>,
    udp_peer: Option<SocketAddr>,
}

struct RateMeter {
    count: usize,
    window_started_at: Instant,
    last_rate: f64,
}

impl RateMeter {
    fn new() -> Self {
        Self {
            count: 0,
            window_started_at: Instant::now(),
            last_rate: 0.0,
        }
    }

    fn tick(&mut self) -> f64 {
        self.count += 1;
        let elapsed = self.window_started_at.elapsed();
        if elapsed >= Duration::from_millis(250) {
            self.last_rate = self.count as f64 / elapsed.as_secs_f64();
            self.count = 0;
            self.window_started_at = Instant::now();
        }
        self.last_rate
    }

    fn reset(&mut self) {
        self.count = 0;
        self.window_started_at = Instant::now();
        self.last_rate = 0.0;
    }
}

impl RuntimeState {
    fn snapshot(&self) -> RuntimeSnapshot {
        RuntimeSnapshot {
            overlay_enabled: self.overlay_enabled,
            cursor_active: self.cursor_active,
            trackpad_override_enabled: self.trackpad_override_enabled,
            trackpad_drawing_enabled: self.trackpad_drawing_enabled,
            simulated: self.simulated,
            rate: self.rate,
            applied_rate: self.applied_rate,
            move_latency_ms: self.move_latency_ms.clone(),
            connected: self.connected,
            latency_ms: self.latency_ms.clone(),
            source: self.source.clone(),
        }
    }
}

type SharedState = Arc<Mutex<RuntimeState>>;

fn initial_state() -> RuntimeState {
    RuntimeState {
        overlay_enabled: false,
        cursor_active: true,
        trackpad_override_enabled: true,
        trackpad_drawing_enabled: false,
        simulated: true,
        rate: 0.0,
        applied_rate: 0.0,
        move_latency_ms: "—".into(),
        connected: false,
        latency_ms: "—".into(),
        source: "simulated".into(),
        last_udp_received: None,
        last_udp_sequence: None,
        udp_peer: None,
    }
}

fn show_notebook(app: &AppHandle) -> Result<(), String> {
    let notebook = app
        .get_webview_window("notebook")
        .ok_or_else(|| "Notebook window is not configured".to_string())?;

    notebook
        .unminimize()
        .map_err(|error| format!("Could not restore notebook: {error}"))?;
    notebook
        .show()
        .map_err(|error| format!("Could not show notebook: {error}"))?;
    notebook
        .set_focus()
        .map_err(|error| format!("Could not focus notebook: {error}"))?;

    Ok(())
}

fn simulate_position(elapsed: Duration) -> (f64, f64) {
    let orbit = elapsed.as_secs_f64();
    let x = 0.5 + (orbit * 1.1).sin() * 0.32;
    let y = 0.5 + (orbit * 0.9).cos() * 0.22;
    (x.clamp(0.0, 1.0), y.clamp(0.0, 1.0))
}

fn emit_pen_input_for_active_surface(
    app: &AppHandle,
    overlay_active: bool,
    event: PenInputEvent,
) {
    if overlay_active {
        let _ = app.emit_to("overlay", "penultimate:pen-input", event);
        return;
    }

    if let Some(notebook) = app.get_webview_window("notebook") {
        if notebook.is_visible().unwrap_or(false) {
            let _ = app.emit_to("notebook", "penultimate:pen-input", event);
        }
    }
}

fn spawn_udp_receiver(state: SharedState, app: AppHandle) {
    thread::spawn(move || {
        let socket = match UdpSocket::bind(UDP_BIND_ADDRESS) {
            Ok(socket) => socket,
            Err(error) => {
                eprintln!("Could not bind Penultimate UDP receiver to {UDP_BIND_ADDRESS}: {error}");
                let mut guard = state.lock().expect("runtime state lock");
                guard.connected = false;
                guard.latency_ms = "UDP bind failed".into();
                return;
            }
        };
        let _ = socket.set_read_timeout(Some(Duration::from_millis(25)));
        let mut buffer = [0_u8; MAX_UDP_PACKET_BYTES];
        let mut rate_meter = RateMeter::new();
        let mut last_event: Option<PenInputEvent> = None;
        let mut last_invalid_packet_log: Option<Instant> = None;

        loop {
            let expired_overlay_active = {
                let guard = state.lock().expect("runtime state lock");
                guard.last_udp_received.is_some_and(|last_received| {
                    last_received.elapsed() >= Duration::from_millis(UDP_DISCONNECT_MS)
                }) && guard.overlay_enabled
                    && guard.cursor_active
                    && !guard.trackpad_drawing_enabled
            };
            let session_expired = {
                let guard = state.lock().expect("runtime state lock");
                guard.last_udp_received.is_some_and(|last_received| {
                    last_received.elapsed() >= Duration::from_millis(UDP_DISCONNECT_MS)
                })
            };
            if session_expired {
                if expired_overlay_active {
                    if let Some(previous) = last_event.as_ref().filter(|event| event.pen_down) {
                        let _ = app.emit_to(
                            "overlay",
                            "penultimate:pen-input",
                            PenInputEvent {
                                pen_down: false,
                                pressure: None,
                                timestamp: unix_time_ms(),
                                ..previous.clone()
                            },
                        );
                    }
                }
                let mut guard = state.lock().expect("runtime state lock");
                guard.last_udp_received = None;
                guard.last_udp_sequence = None;
                guard.udp_peer = None;
                last_event = None;
                rate_meter.reset();
            }

            let (packet_size, peer) = match socket.recv_from(&mut buffer) {
                Ok(received) => received,
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    continue;
                }
                Err(error) => {
                    eprintln!("Penultimate UDP receive error: {error}");
                    continue;
                }
            };

            let packet = match serde_json::from_slice::<UdpPenPacket>(&buffer[..packet_size]) {
                Ok(packet) => packet,
                Err(error) => {
                    if last_invalid_packet_log
                        .is_none_or(|last_log| last_log.elapsed() >= Duration::from_secs(1))
                    {
                        eprintln!("Ignoring invalid Penultimate UDP packet: {error}");
                        last_invalid_packet_log = Some(Instant::now());
                    }
                    continue;
                }
            };
            if packet.valid == Some(false)
                || !packet.x.is_finite()
                || !packet.y.is_finite()
                || packet
                    .pressure
                    .is_some_and(|pressure| !pressure.is_finite())
                || packet
                    .tracking_confidence
                    .is_some_and(|confidence| !confidence.is_finite())
            {
                continue;
            }

            let received_at = Instant::now();
            let timestamp = packet_timestamp_ms(packet.timestamp);
            let event = PenInputEvent {
                x: packet.x.clamp(0.0, 1.0),
                y: packet.y.clamp(0.0, 1.0),
                pen_down: packet.pen_down,
                pressure: packet.pressure.map(|pressure| pressure.clamp(0.0, 1.0)),
                timestamp,
            };
            let overlay_active = {
                let mut guard = state.lock().expect("runtime state lock");
                if guard
                    .udp_peer
                    .is_some_and(|active_peer| active_peer != peer)
                {
                    continue;
                }
                guard.udp_peer = Some(peer);
                let ordered = packet.sequence.is_none_or(|sequence| {
                    guard.last_udp_sequence.is_none_or(|previous| {
                        sequence > previous
                            || previous.saturating_sub(sequence) > UDP_SEQUENCE_RESET_GAP
                    })
                });
                if !ordered {
                    continue;
                } else {
                    guard.last_udp_received = Some(received_at);
                    if let Some(sequence) = packet.sequence {
                        guard.last_udp_sequence = Some(sequence);
                    }
                    guard.rate = rate_meter.tick();
                    guard.applied_rate = 0.0;
                    guard.connected = true;
                    guard.source = "pi".into();
                    guard.move_latency_ms = "—".into();
                    guard.latency_ms = packet_latency(packet.timestamp.map(|value| {
                        if value >= 1_000_000_000_000.0 { value as u64 } else { (value * 1_000.0) as u64 }
                    }));
                    guard.overlay_enabled && guard.cursor_active && !guard.trackpad_drawing_enabled
                }
            };

            last_event = Some(event.clone());
            emit_pen_input_for_active_surface(&app, overlay_active, event);
        }
    });
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

fn packet_timestamp_ms(timestamp: Option<f64>) -> u64 {
    match timestamp {
        Some(value) if value.is_finite() && value >= 1_000_000_000_000.0 => value as u64,
        Some(value) if value.is_finite() && value >= 1_000_000_000.0 => (value * 1_000.0) as u64,
        _ => unix_time_ms(),
    }
}

fn packet_latency(timestamp: Option<u64>) -> String {
    let now_ms = unix_time_ms();
    if timestamp.is_some_and(|timestamp| timestamp > 1_000_000_000_000 && timestamp <= now_ms) {
        let latency = now_ms - timestamp.unwrap_or(now_ms);
        if latency <= 60_000 {
            format!("{latency} ms")
        } else {
            "clock unsynced".into()
        }
    } else {
        "arrival".into()
    }
}

fn spawn_simulation_loop(state: SharedState, app: AppHandle) {
    thread::spawn(move || {
        let mut cursor_driver = Enigo::new(&Settings::default()).ok();
        let display_size = cursor_driver
            .as_mut()
            .and_then(|enigo| enigo.main_display().ok());

        let started_at = Instant::now();
        let mut loop_meter = RateMeter::new();
        let mut applied_meter = RateMeter::new();
        let mut move_latency_total_ms = 0.0f64;
        let mut move_latency_samples = 0usize;
        let mut last_injected_position: Option<(i32, i32)> = None;
        let mut trackpad_override_until: Option<Instant> = None;
        let mut override_cursor_position: Option<(i32, i32)> = None;

        loop {
            thread::sleep(Duration::from_millis(8));
            let loop_started_at = Instant::now();

            let current_state = {
                let guard = state.lock().expect("runtime state lock");
                let udp_active = guard.last_udp_received.is_some();
                (
                    guard.overlay_enabled,
                    guard.cursor_active,
                    guard.simulated,
                    guard.trackpad_override_enabled,
                    guard.trackpad_drawing_enabled,
                    udp_active,
                )
            };

            if current_state.5 {
                loop_meter.reset();
                applied_meter.reset();
                last_injected_position = None;
                trackpad_override_until = None;
                override_cursor_position = None;
                continue;
            }

            if !current_state.2 {
                let mut guard = state.lock().expect("runtime state lock");
                guard.rate = 0.0;
                guard.applied_rate = 0.0;
                guard.move_latency_ms = "—".into();
                guard.connected = false;
                guard.latency_ms = "—".into();
                loop_meter.reset();
                applied_meter.reset();
                move_latency_total_ms = 0.0;
                move_latency_samples = 0;
                last_injected_position = None;
                trackpad_override_until = None;
                override_cursor_position = None;
                continue;
            }

            let elapsed = started_at.elapsed();
            let (x, y) = simulate_position(elapsed);
            let cycle = elapsed.as_secs_f64() % 3.2;
            let pen_down = cycle < 2.25;
            let pressure = if pen_down {
                Some(0.52 + (elapsed.as_secs_f64() * 2.4).sin() * 0.18)
            } else {
                None
            };
            let loop_rate = loop_meter.tick();
            let mut applied_rate = applied_meter.last_rate;
            let mut trackpad_override_active = false;

            if current_state.1 && !current_state.4 {
                if let (Some(enigo), Some((width, height))) = (cursor_driver.as_mut(), display_size)
                {
                    let px = (x * f64::from(width.saturating_sub(1))).round() as i32;
                    let py = (y * f64::from(height.saturating_sub(1))).round() as i32;

                    if !current_state.3 {
                        trackpad_override_until = None;
                        override_cursor_position = None;
                    } else if let Some(mut until) = trackpad_override_until {
                        if let Ok(current_position) = enigo.location() {
                            if override_cursor_position
                                .is_some_and(|previous| previous != current_position)
                            {
                                until =
                                    loop_started_at + Duration::from_millis(TRACKPAD_OVERRIDE_MS);
                                trackpad_override_until = Some(until);
                            }
                            override_cursor_position = Some(current_position);
                        }

                        if loop_started_at < until {
                            trackpad_override_active = true;
                        } else {
                            trackpad_override_until = None;
                            override_cursor_position = None;
                            last_injected_position = None;
                        }
                    }

                    if current_state.3 && !trackpad_override_active {
                        if let Some((last_x, last_y)) = last_injected_position {
                            if let Ok((current_x, current_y)) = enigo.location() {
                                let moved_by_user = (current_x - last_x).abs()
                                    > TRACKPAD_OVERRIDE_DISTANCE_PX
                                    || (current_y - last_y).abs() > TRACKPAD_OVERRIDE_DISTANCE_PX;

                                if moved_by_user {
                                    trackpad_override_until = Some(
                                        loop_started_at
                                            + Duration::from_millis(TRACKPAD_OVERRIDE_MS),
                                    );
                                    override_cursor_position = Some((current_x, current_y));
                                    trackpad_override_active = true;
                                    last_injected_position = None;
                                }
                            }
                        }
                    }

                    if trackpad_override_active {
                        applied_meter.reset();
                        applied_rate = 0.0;
                    } else {
                        let move_started_at = Instant::now();
                        if enigo.move_mouse(px, py, Coordinate::Abs).is_ok() {
                            last_injected_position = Some((px, py));
                            applied_rate = applied_meter.tick();
                            move_latency_total_ms +=
                                move_started_at.elapsed().as_secs_f64() * 1000.0;
                            move_latency_samples += 1;
                        }
                    }
                } else {
                    applied_meter.reset();
                    applied_rate = 0.0;
                    last_injected_position = None;
                    trackpad_override_until = None;
                    override_cursor_position = None;
                }
            } else {
                applied_meter.reset();
                applied_rate = 0.0;
                last_injected_position = None;
                trackpad_override_until = None;
                override_cursor_position = None;
            }

            let mut guard = state.lock().expect("runtime state lock");
            guard.rate = loop_rate;
            guard.applied_rate = applied_rate;
            if trackpad_override_active {
                guard.move_latency_ms = "trackpad override".into();
            } else if current_state.1
                && cursor_driver.is_some()
                && display_size.is_some()
            {
                if move_latency_samples > 0 {
                    guard.move_latency_ms = format!(
                        "{:.1} ms",
                        move_latency_total_ms / move_latency_samples as f64
                    );
                    move_latency_total_ms = 0.0;
                    move_latency_samples = 0;
                }
            } else if current_state.1 {
                guard.move_latency_ms = "cursor unavailable".into();
            } else {
                guard.move_latency_ms = "—".into();
            }
            guard.connected = true;
            guard.latency_ms = "0 ms".into();
            guard.source = "simulated".into();
            drop(guard);

            if current_state.1 && !trackpad_override_active && !current_state.4 {
                emit_pen_input_for_active_surface(
                    &app,
                    current_state.0,
                    PenInputEvent {
                        x,
                        y,
                        pen_down,
                        pressure,
                        timestamp: elapsed.as_millis() as u64,
                    },
                );
            }
        }
    });
}

#[tauri::command]
fn get_runtime_snapshot(state: State<SharedState>) -> RuntimeSnapshot {
    let guard = state.lock().expect("runtime state lock");
    guard.snapshot()
}

fn apply_annotation_visibility(app: &AppHandle, enabled: bool) {
    if let Some(overlay) = app.get_webview_window("overlay") {
        if enabled {
            let _ = overlay.show();
        } else {
            let _ = overlay.hide();
        }
    }
    if let Some(toolbar) = app.get_webview_window("toolbar") {
        if enabled {
            position_annotation_toolbar(&toolbar);
            let _ = toolbar.show();
        } else {
            let _ = toolbar.hide();
        }
    }
    // Keep the overlay/toolbar webviews in sync with native toggles so their
    // internal listeners run identically whether the toggle came from the
    // keyboard shortcut or the tray menu.
    let _ = app.emit_to("overlay", "penultimate:set-overlay-enabled", enabled);
    let _ = app.emit_to("toolbar", "penultimate:set-toolbar-visible", enabled);
}

fn position_annotation_toolbar(toolbar: &tauri::WebviewWindow) {
    if let (Ok(Some(monitor)), Ok(size)) = (toolbar.current_monitor(), toolbar.outer_size()) {
        let x = monitor.position().x
            + (monitor.size().width.saturating_sub(size.width) / 2) as i32;
        let y = monitor.position().y + (40.0 * monitor.scale_factor()).round() as i32;
        let _ = toolbar.set_position(tauri::PhysicalPosition::new(x, y));
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SetRuntimeFlagsPayload {
    overlay_enabled: Option<bool>,
    cursor_active: Option<bool>,
    trackpad_override_enabled: Option<bool>,
    trackpad_drawing_enabled: Option<bool>,
    simulated: Option<bool>,
}

#[tauri::command]
fn set_runtime_flags(payload: SetRuntimeFlagsPayload, state: State<SharedState>, app: AppHandle) {
    let mut guard = state.lock().expect("runtime state lock");
    let overlay_visibility = payload.overlay_enabled;
    let trackpad_drawing = payload.trackpad_drawing_enabled;
    if let Some(value) = payload.overlay_enabled {
        guard.overlay_enabled = value;
    }
    if let Some(value) = payload.cursor_active {
        guard.cursor_active = value;
    }
    if let Some(value) = payload.trackpad_override_enabled {
        guard.trackpad_override_enabled = value;
    }
    if let Some(value) = payload.trackpad_drawing_enabled {
        guard.trackpad_drawing_enabled = value;
    }
    if let Some(value) = payload.simulated {
        guard.simulated = value;
    }
    drop(guard);

    if let Some(visible) = overlay_visibility {
        if let Some(overlay) = app.get_webview_window("overlay") {
            if visible {
                let _ = overlay.show();
            } else {
                let _ = overlay.hide();
            }
        }
        if let Some(toolbar) = app.get_webview_window("toolbar") {
            if visible {
                position_annotation_toolbar(&toolbar);
                let _ = toolbar.show();
            } else {
                let _ = toolbar.hide();
            }
        }
    }

    if let Some(enabled) = trackpad_drawing {
        if let Some(overlay) = app.get_webview_window("overlay") {
            let _ = overlay.set_ignore_cursor_events(!enabled);
            let _ = overlay.set_focusable(enabled);
            if enabled {
                let _ = overlay.set_focus();
            }
            let _ = app.emit_to("overlay", "penultimate:set-trackpad-drawing", enabled);
        }
    }
}

#[tauri::command]
fn open_calibration(app: AppHandle) -> Result<(), String> {
    let calibration = app
        .get_webview_window("calibration")
        .ok_or_else(|| "Calibration window is not configured".to_string())?;

    calibration
        .set_decorations(false)
        .map_err(|error| format!("Could not hide calibration window chrome: {error}"))?;
    let monitor = calibration
        .current_monitor()
        .map_err(|error| format!("Could not find the calibration monitor: {error}"))?
        .or(calibration
            .primary_monitor()
            .map_err(|error| format!("Could not find the primary monitor: {error}"))?)
        .ok_or_else(|| "No monitor is available for calibration".to_string())?;
    calibration
        .set_position(*monitor.position())
        .map_err(|error| format!("Could not position calibration on the monitor: {error}"))?;
    calibration
        .show()
        .map_err(|error| format!("Could not show calibration: {error}"))?;
    let actual_position = calibration
        .outer_position()
        .map_err(|error| format!("Could not measure the calibration position: {error}"))?;
    let top_inset = actual_position
        .y
        .saturating_sub(monitor.position().y)
        .max(0) as u32;
    let mut calibration_size = *monitor.size();
    calibration_size.height = calibration_size
        .height
        .saturating_sub(top_inset)
        .saturating_add(2);
    calibration
        .set_size(calibration_size)
        .map_err(|error| format!("Could not size calibration to the visible screen: {error}"))?;
    calibration
        .set_focus()
        .map_err(|error| format!("Could not focus calibration: {error}"))?;

    if let Some(notebook) = app.get_webview_window("notebook") {
        notebook
            .hide()
            .map_err(|error| format!("Could not hide notebook: {error}"))?;
    }

    Ok(())
}

#[tauri::command]
fn close_calibration(app: AppHandle) -> Result<(), String> {
    let calibration = app
        .get_webview_window("calibration")
        .ok_or_else(|| "Calibration window is not configured".to_string())?;

    calibration
        .hide()
        .map_err(|error| format!("Could not hide calibration: {error}"))?;

    show_notebook(&app)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let shared_state = Arc::new(Mutex::new(initial_state()));

    tauri::Builder::default()
        .manage(shared_state.clone())
        .setup(move |app| {
            spawn_simulation_loop(shared_state.clone(), app.handle().clone());
            spawn_udp_receiver(shared_state.clone(), app.handle().clone());

            let open_notebook = MenuItem::with_id(app, "open-notebook", "Open Notebook", true, None::<&str>)?;
            let toggle_annotation = MenuItem::with_id(
                app,
                "toggle-annotation",
                "Start Annotation",
                true,
                Some("CmdOrCtrl+Alt+A"),
            )?;
            let calibrate = MenuItem::with_id(app, "calibrate", "Calibrate Screen", true, None::<&str>)?;
            let connection = MenuItem::with_id(app, "connection", "Pen connecting…", false, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit Penultimate", true, None::<&str>)?;
            let menu = Menu::with_items(
                app,
                &[
                    &open_notebook,
                    &toggle_annotation,
                    &PredefinedMenuItem::separator(app)?,
                    &calibrate,
                    &connection,
                    &PredefinedMenuItem::separator(app)?,
                    &quit,
                ],
            )?;
            let toggle_annotation_item = toggle_annotation.clone();
            let annotation_status_item = toggle_annotation.clone();
            let tray_builder = TrayIconBuilder::new()
                .menu(&menu)
                .tooltip("Penultimate")
                .icon_as_template(true)
                .on_menu_event(move |app, event| match event.id().as_ref() {
                    "open-notebook" => {
                        let _ = show_notebook(app);
                    }
                    "toggle-annotation" => {
                        let state = app.state::<SharedState>();
                        let enabled = {
                            let mut guard = state.lock().expect("runtime state lock");
                            guard.overlay_enabled = !guard.overlay_enabled;
                            guard.overlay_enabled
                        };
                        apply_annotation_visibility(app, enabled);
                        let _ = toggle_annotation_item.set_text(
                            if enabled { "Stop Annotation" } else { "Start Annotation" },
                        );
                    }
                    "calibrate" => {
                        let _ = open_calibration(app.clone());
                    }
                    "quit" => app.exit(0),
                    _ => {}
                });
            let tray_builder = if let Some(icon) = app.default_window_icon() {
                tray_builder.icon(icon.clone())
            } else {
                tray_builder
            };
            let _tray = tray_builder.build(app)?;

            let connection_item = connection.clone();
            let connection_state = shared_state.clone();
            thread::spawn(move || loop {
                thread::sleep(Duration::from_millis(500));
                let (connected, annotation_enabled) = {
                    let guard = connection_state.lock().expect("runtime state lock");
                    (guard.connected, guard.overlay_enabled)
                };
                let _ = connection_item.set_text(
                    if connected { "Pen connected" } else { "Pen disconnected" },
                );
                let _ = annotation_status_item.set_text(
                    if annotation_enabled { "Stop Annotation" } else { "Start Annotation" },
                );
            });
            if let Some(overlay) = app.get_webview_window("overlay") {
                let _ = overlay.hide();
                let _ = overlay.set_ignore_cursor_events(true);
                let _ = overlay.set_focusable(false);
            }

            if let Some(notebook) = app.get_webview_window("notebook") {
                let _ = notebook.hide();
            }

            if let Some(calibration) = app.get_webview_window("calibration") {
                let _ = calibration.hide();
            }

            if let Some(toolbar) = app.get_webview_window("toolbar") {
                let _ = toolbar.hide();
                #[cfg(target_os = "macos")]
                let _ = window_vibrancy::apply_vibrancy(
                    &toolbar,
                    window_vibrancy::NSVisualEffectMaterial::Popover,
                    Some(window_vibrancy::NSVisualEffectState::Active),
                    Some(12.0),
                );
            }

            if let Some(status) = app.get_webview_window("status") {
                let _ = status.set_ignore_cursor_events(true);
                let _ = status.set_focusable(false);
                if let Ok(Some(monitor)) = status.primary_monitor() {
                    if let Ok(size) = status.outer_size() {
                        let scale = monitor.scale_factor();
                        let x = monitor.position().x
                            + (monitor.size().width.saturating_sub(size.width) / 2) as i32;
                        let y = monitor.position().y + (48.0 * scale).round() as i32;
                        let _ = status.set_position(tauri::PhysicalPosition::new(x, y));
                    }
                }
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == "notebook" {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            get_runtime_snapshot,
            set_runtime_flags,
            open_calibration,
            close_calibration
        ])
        .build(tauri::generate_context!())
        .expect("error while building Penultimate")
        .run(|app, event| {
            #[cfg(target_os = "macos")]
            if matches!(event, tauri::RunEvent::Reopen { .. }) {
                let _ = show_notebook(app);
            }
        });
}
