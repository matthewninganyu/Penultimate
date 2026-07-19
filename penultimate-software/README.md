# Tauri + React + Typescript

## UDP pen input

Penultimate listens for UDP pen packets on `0.0.0.0:4242`. A valid UDP feed
takes priority over simulation; simulation resumes after 500 ms without a UDP
packet.

Send one JSON object per datagram:

```json
{
  "type": "pen",
  "sequence": 42,
  "x": 0.5,
  "y": 0.4,
  "penDown": true,
  "pressure": 0.65,
  "timestamp": 1784390123456
}
```

- `x` and `y` are normalized screen coordinates from 0 to 1.
- `sequence` is optional but recommended. It describes packet order and is
  used to reject duplicates or UDP packets that arrive late.
- `pressure` is optional and clamped from 0 to 1.
- `timestamp` is optional. It can be Unix time in milliseconds; other
  monotonic timestamps are accepted, but network latency will be reported as
  `arrival`. When omitted, Penultimate uses laptop arrival time.
- The first valid sender owns the UDP session. Packets from other addresses are
  ignored until the active sender has been silent for 500 ms.
- Packets larger than 2048 bytes are not accepted.

For a Raspberry Pi or another computer, send to the laptop's LAN IP on port
4242. Local test software can send to `127.0.0.1:4242`.

This template should help get you started developing with Tauri, React and Typescript in Vite.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
