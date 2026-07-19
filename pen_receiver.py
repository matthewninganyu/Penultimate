from __future__ import annotations

import argparse
import csv
import json
import socket
import time
from pathlib import Path
from typing import Any


RAW_PACKET_FIELDS = ("type", "sequence", "timestamp", "valid", "camera_0", "camera_1")
PEN_PACKET_FIELDS = (
    "sequence",
    "timestamp",
    "valid",
    "normalized_x",
    "normalized_y",
    "pixel_x",
    "pixel_y",
    "x_mm",
    "y_mm",
    "distance_mm",
    "touching",
    "contact_confidence",
    "tracking_confidence",
    "frame_skew_ms",
)


class SequenceGate:
    def __init__(self) -> None:
        self.last_sequence: int | None = None

    def accept(self, packet: dict[str, Any]) -> bool:
        sequence = int(packet["sequence"])
        if self.last_sequence is not None and sequence <= self.last_sequence:
            return False
        self.last_sequence = sequence
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive finalized Penultimate UDP pen states.")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Accept raw camera-coordinate packets from realtime.py.",
    )
    return parser.parse_args()


def run_receiver(args: argparse.Namespace) -> None:
    if args.raw and args.csv is not None:
        raise ValueError("--csv currently supports finalized PenState packets, not --raw packets.")

    gate = SequenceGate()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.settimeout(args.timeout)
    writer = None
    csv_file = None
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.csv.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=list(PEN_PACKET_FIELDS))
        writer.writeheader()

    last_packet_time = time.time()
    try:
        print(f"Listening on {args.bind}:{args.port}")
        while True:
            try:
                payload, _ = sock.recvfrom(65535)
            except socket.timeout:
                if time.time() - last_packet_time >= args.timeout:
                    print("Tracking lost: no packets; touching=false")
                continue
            packet = decode_raw_packet(payload) if args.raw else decode_pen_packet(payload)
            if packet is None or not gate.accept(packet):
                continue
            last_packet_time = time.time()
            print(format_raw_packet(packet) if args.raw else format_packet(packet))
            if writer is not None:
                writer.writerow(packet)
                csv_file.flush()
    except KeyboardInterrupt:
        print("Receiver interrupted.")
    finally:
        sock.close()
        if csv_file is not None:
            csv_file.close()


def format_packet(packet: dict[str, Any]) -> str:
    if not packet["valid"]:
        return f"seq={packet['sequence']} invalid touching=false"
    return (
        f"seq={packet['sequence']} px=({packet['pixel_x']}, {packet['pixel_y']}) "
        f"norm=({packet['normalized_x']:.3f}, {packet['normalized_y']:.3f}) "
        f"z={packet['distance_mm']:.1f}mm touching={packet['touching']} "
        f"track={packet['tracking_confidence']:.2f} contact={packet['contact_confidence']:.2f}"
    )


def decode_pen_packet(payload: bytes) -> dict[str, Any] | None:
    try:
        packet = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not validate_pen_packet(packet):
        return None
    return packet


def validate_pen_packet(packet: Any) -> bool:
    if not isinstance(packet, dict):
        return False
    if any(field not in packet for field in PEN_PACKET_FIELDS):
        return False
    if not isinstance(packet["sequence"], int):
        return False
    if not isinstance(packet["valid"], bool):
        return False
    if not isinstance(packet["touching"], bool):
        return False
    return True


def decode_raw_packet(payload: bytes) -> dict[str, Any] | None:
    try:
        packet = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not validate_raw_packet(packet):
        return None
    return packet


def validate_raw_packet(packet: Any) -> bool:
    if not isinstance(packet, dict):
        return False
    if any(field not in packet for field in RAW_PACKET_FIELDS):
        return False
    if packet["type"] != "raw_coordinates":
        return False
    if not isinstance(packet["sequence"], int):
        return False
    if not isinstance(packet["valid"], bool):
        return False
    return True


def format_raw_packet(packet: dict[str, Any]) -> str:
    camera_0 = packet["camera_0"]
    camera_1 = packet["camera_1"]
    if not packet["valid"] or camera_0 is None or camera_1 is None:
        return f"seq={packet['sequence']} raw invalid camera_0={camera_0} camera_1={camera_1}"
    return (
        f"seq={packet['sequence']} raw "
        f"cam0=({camera_0['x']:.1f}, {camera_0['y']:.1f}) "
        f"cam1=({camera_1['x']:.1f}, {camera_1['y']:.1f}) "
        f"area=({camera_0['area']:.1f}, {camera_1['area']:.1f})"
    )


def main() -> None:
    run_receiver(parse_args())


if __name__ == "__main__":
    main()

