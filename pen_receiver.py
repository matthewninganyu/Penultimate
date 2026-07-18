from __future__ import annotations

import argparse
import csv
import socket
import time
from pathlib import Path
from typing import Any

from network_sender import REQUIRED_PACKET_FIELDS, SequenceGate, decode_pen_packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive finalized Penultimate UDP pen states.")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def run_receiver(args: argparse.Namespace) -> None:
    gate = SequenceGate()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    sock.settimeout(args.timeout)
    writer = None
    csv_file = None
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.csv.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=list(REQUIRED_PACKET_FIELDS))
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
            packet = decode_pen_packet(payload)
            if packet is None or not gate.accept(packet):
                continue
            last_packet_time = time.time()
            print(format_packet(packet))
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


def main() -> None:
    run_receiver(parse_args())


if __name__ == "__main__":
    main()

