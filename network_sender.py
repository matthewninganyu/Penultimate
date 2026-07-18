from __future__ import annotations

import json
import logging
import socket
from typing import Any

from models import PenState


LOGGER = logging.getLogger(__name__)
REQUIRED_PACKET_FIELDS = (
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


class UdpPenSender:
    def __init__(self, host: str, port: int, raise_on_error: bool = False) -> None:
        self.address = (host, port)
        self.raise_on_error = raise_on_error
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, state: PenState) -> None:
        payload = serialize_pen_state(state)
        try:
            self.socket.sendto(payload, self.address)
        except OSError as error:
            LOGGER.warning("UDP send failed: %s", error)
            if self.raise_on_error:
                raise

    def close(self) -> None:
        self.socket.close()


def serialize_pen_state(state: PenState) -> bytes:
    return json.dumps(state.to_packet(), separators=(",", ":"), allow_nan=False).encode("utf-8")


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
    if any(field not in packet for field in REQUIRED_PACKET_FIELDS):
        return False
    if not isinstance(packet["sequence"], int):
        return False
    if not isinstance(packet["valid"], bool):
        return False
    if not isinstance(packet["touching"], bool):
        return False
    return True


class SequenceGate:
    def __init__(self) -> None:
        self.last_sequence: int | None = None

    def accept(self, packet: dict[str, Any]) -> bool:
        sequence = int(packet["sequence"])
        if self.last_sequence is not None and sequence <= self.last_sequence:
            return False
        self.last_sequence = sequence
        return True

