import socket
import time
import unittest

from fbg_lan_transport import FbgLanTransport, LAN_HELLO
from fbg_remote_protocol import (
    MODE_STRESS,
    MetadataFrame,
    TelemetryFrame,
    decode_mode_command,
    encode_metadata,
    encode_telemetry,
)


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    return predicate()


class LanTransportTests(unittest.TestCase):
    def test_network_port_must_be_an_exact_integer(self):
        with self.assertRaises(ValueError):
            FbgLanTransport(board_port=45670.5)
        with self.assertRaises(ValueError):
            FbgLanTransport(board_port=True)

    def setUp(self):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.listener.settimeout(2.0)
        self.metadata = []
        self.connections = []
        self.transport = FbgLanTransport(
            board_host="127.0.0.1",
            board_port=self.listener.getsockname()[1],
            heartbeat_s=0.1,
            discovery_interval_s=0.05,
            on_metadata=self.metadata.append,
            on_connection=lambda connected, detail: self.connections.append(
                (connected, detail)
            ),
        )
        self.transport.start()
        self.board, _ = self.listener.accept()
        self.board.settimeout(2.0)

    def tearDown(self):
        self.transport.stop()
        self.board.close()
        self.listener.close()

    def _receive_until(self, marker, timeout=2.0):
        data = bytearray()
        deadline = time.monotonic() + timeout
        while marker not in data and time.monotonic() < deadline:
            data.extend(self.board.recv(256))
        return bytes(data)

    def test_stream_raw_frame_metadata_and_mode_command(self):
        self.assertIn(LAN_HELLO, self._receive_until(LAN_HELLO))

        metadata = MetadataFrame(
            mode=MODE_STRESS,
            table_crc32=0x12345678,
            wavelength_pm=(1_532_000, 1_532_100),
        )
        frame = TelemetryFrame(
            mode=MODE_STRESS,
            flags=0x40,
            boot_id=7,
            seq=11,
            uptime_ms=1234,
            temperature_mC=25_000,
            table_crc32=metadata.table_crc32,
            gainmask0=0,
            gainmask1=0,
            channel_mask=0x0F,
            samples=((100, 200, 300, 400), (101, 201, 301, 401)),
        )
        # Deliberately coalesce packets and split the resulting TCP stream.
        stream = encode_metadata(metadata) + encode_telemetry(frame)
        self.board.sendall(stream[:17])
        self.board.sendall(stream[17:])

        self.assertTrue(wait_until(lambda: len(self.metadata) == 1))
        self.assertTrue(wait_until(lambda: self.transport.connected))
        self.assertEqual(self.metadata[0], metadata)
        self.assertEqual(self.transport.take_latest_telemetry(), frame)
        self.assertEqual(self.transport.board_ip, "127.0.0.1")

        self.transport.publish_mode(MODE_STRESS, 0xAABBCCDD)
        received = self._receive_until(b"FCMD1|AABBCCDD|MODE|STRESS")
        offset = received.index(b"FCMD1|")
        command = decode_mode_command(received[offset : offset + 26])
        self.assertEqual(command.command_id, 0xAABBCCDD)
        self.assertEqual(command.mode, MODE_STRESS)

    def test_invalid_stream_prefix_is_resynchronized(self):
        metadata = MetadataFrame(
            mode=MODE_STRESS,
            table_crc32=0x12345678,
            wavelength_pm=(1_532_000, 1_532_100),
        )
        self.board.sendall(b"noise" + encode_metadata(metadata))
        self.assertTrue(wait_until(lambda: len(self.metadata) == 1))
        self.assertEqual(self.metadata[0], metadata)

    def test_malformed_ack_prefix_does_not_block_following_packet(self):
        metadata = MetadataFrame(
            mode=MODE_STRESS,
            table_crc32=0x87654321,
            wavelength_pm=(1_540_000, 1_540_100),
        )
        self.board.sendall(b"FACK1|BROKEN|" + encode_metadata(metadata))
        self.assertTrue(wait_until(lambda: len(self.metadata) == 1))
        self.assertEqual(self.metadata[0], metadata)

    def test_protocol_maximum_metadata_frame_is_accepted(self):
        metadata = MetadataFrame(
            mode=MODE_STRESS,
            table_crc32=0x10203040,
            wavelength_pm=tuple(range(1_500_000, 1_502_500)),
        )
        encoded = encode_metadata(metadata)
        self.assertGreater(len(encoded), 4096)
        self.board.sendall(encoded)
        self.assertTrue(wait_until(lambda: len(self.metadata) == 1))
        self.assertEqual(self.metadata[0], metadata)

    def test_protocol_maximum_telemetry_frame_is_accepted(self):
        frame = TelemetryFrame(
            mode=MODE_STRESS,
            flags=0,
            boot_id=3,
            seq=9,
            uptime_ms=500,
            temperature_mC=25_000,
            table_crc32=0x11223344,
            gainmask0=0,
            gainmask1=0,
            channel_mask=0x0F,
            samples=tuple((100, 200, 300, 400) for _ in range(2500)),
        )
        encoded = encode_telemetry(frame)
        self.assertGreater(len(encoded), 4096)
        self.board.sendall(encoded)
        self.assertTrue(
            wait_until(lambda: self.transport.take_latest_telemetry() == frame)
        )


if __name__ == "__main__":
    unittest.main()
