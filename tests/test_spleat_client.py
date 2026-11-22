import socket
import struct
import threading

import numpy as np
import pytest

from reflex.runtime.spleat_client import (ProtocolError, cbor_decode, cbor_encode, parse_message,
                                          read_message)


def test_cbor_roundtrip_and_known_bytes():
    msg = {"id": 42, "payload": b"\x00\x01", "latency": 0, "error": "", "f": 1.5, "neg": -3, "ok": True}
    assert cbor_decode(cbor_encode(msg)) == msg
    # From docs/fpga-pipeline.md: map of 4 pairs, text "id", number 42.
    assert cbor_encode({"id": 42}) == bytes([0xA1, 0x62, 0x69, 0x64, 0x18, 0x2A])
    assert cbor_decode(bytes([0xFA]) + struct.pack(">f", 0.25)) == 0.25       # float32


def test_payload_is_little_endian_float32_reshaped():
    values = np.arange(2 * 3 * 4, dtype="<f4")
    body = cbor_encode({"id": 1, "payload": values.tobytes(), "latency": 0, "error": ""})
    m = parse_message(body, (2, 3, 4))
    assert m["shape_ok"] and m["values"].shape == (2, 3, 4) and m["values"][1, 2, 3] == 23
    assert not parse_message(body, (9, 80, 80))["shape_ok"]
    with pytest.raises(ProtocolError):
        parse_message(cbor_encode({"id": 1}), (2, 3, 4))


def test_reads_length_prefixed_messages_from_a_server():
    """A stand-in for Durance's qualia_sim: 4-byte big-endian length, then the CBOR map."""
    bodies = [cbor_encode({"id": i, "payload": np.full(6, i, "<f4").tobytes(), "latency": 0, "error": ""})
              for i in range(3)]
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        with conn:
            for b in bodies:
                conn.sendall(struct.pack(">I", len(b)) + b)

    threading.Thread(target=serve, daemon=True).start()
    with socket.create_connection(srv.getsockname()) as c:
        got = [parse_message(read_message(c), (1, 2, 3)) for _ in bodies]
    srv.close()
    assert [m["id"] for m in got] == [0, 1, 2] and got[2]["values"].sum() == 12
