import unittest

from models import PenState
from network_sender import SequenceGate, decode_pen_packet, serialize_pen_state, validate_pen_packet


class PacketFormatTests(unittest.TestCase):
    def test_serializes_and_validates_pen_state(self) -> None:
        state = PenState(
            sequence=1524,
            timestamp=1784389912.184,
            valid=True,
            normalized_x=0.437,
            normalized_y=0.681,
            pixel_x=839,
            pixel_y=735,
            x_mm=150.3,
            y_mm=132.1,
            distance_mm=2.8,
            touching=True,
            contact_confidence=0.91,
            tracking_confidence=0.94,
            frame_skew_ms=2.4,
        )
        packet = decode_pen_packet(serialize_pen_state(state))
        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertTrue(validate_pen_packet(packet))
        self.assertEqual(packet["sequence"], 1524)
        self.assertTrue(packet["touching"])

    def test_rejects_missing_required_field(self) -> None:
        self.assertFalse(validate_pen_packet({"sequence": 1}))

    def test_rejects_out_of_order_sequences(self) -> None:
        gate = SequenceGate()
        self.assertTrue(gate.accept({"sequence": 10}))
        self.assertFalse(gate.accept({"sequence": 9}))
        self.assertFalse(gate.accept({"sequence": 10}))
        self.assertTrue(gate.accept({"sequence": 11}))


if __name__ == "__main__":
    unittest.main()

