"""Wire protocol between the SIRIUS application and the compute worker.

The C++ side is ``app/core/rpc.hpp``; both ends implement exactly this:

    frame := u32 header_len | header (UTF-8 JSON) | u64 payload_len | payload

Both integers are little endian. The header is an object with

    id       request id (echoed by every reply to it)
    type     "request" | "progress" | "result" | "error"
    method   "hello" | "model_info" | "run" | "cancel" | "ping" | "shutdown"   (requests)
    params   method arguments (object)
    tensors  [{"name", "dtype", "shape", "offset", "nbytes"}]  -- arrays in the payload
    message  progress / error text
    fraction progress 0..1

Tensors are raw little-endian C-order arrays concatenated in the payload at
the given byte offsets. Nothing is pickled: every value crossing the wire is
JSON or a plain array.

Both lengths and every tensor descriptor come from the peer, so both ends cap
them before they size an allocation or index a buffer: ``MAX_HEADER`` is
``kMaxHeaderBytes`` in ``app/core/rpc.cpp``; ``MAX_PAYLOAD`` is what this
worker accepts (a run's input), while the application holds the replies it
reads to its own, lower cap (``rpc::maxPayloadBytes``, 8 GiB unless
``$SIRIUS_RPC_MAX_PAYLOAD_GIB`` says otherwise); at most ``MAX_TENSORS``
descriptors, which must not overlap. A connection that has not completed its
handshake is held to ``MAX_PREAUTH_FRAME`` (see server.py).

``PROTOCOL_VERSION`` is exchanged in ``hello`` (request params and reply
result) and must match on both ends; it is ``kProtocolVersion`` in
``app/core/rpc.hpp``. Bump it whenever the framing or the method set changes
in a way an older peer cannot understand.

The token never crosses the wire (version 2). Both sides prove they know it:

    client -> hello {protocol_version, client_nonce}
    worker -> {protocol_version, server_nonce, server_proof}
    client    checks server_proof; on a mismatch it stops here
    client -> auth  {client_proof}
    worker -> the capabilities

with ``*_proof = handshake_proof(token, role, client_nonce, server_nonce)``,
role ``"worker"`` or ``"client"``: hex HMAC-SHA256 of a fixed message. A
stand-in on the worker's port therefore never sees the token nor anything it
could replay, and the client sends it nothing past the nonce.
``client_handshake`` is the client's half, for tests and scripts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import select
import socket
import struct
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "DTYPES",
    "MAX_HEADER",
    "MAX_PAYLOAD",
    "MAX_PREAUTH_FRAME",
    "MAX_TENSORS",
    "PROTOCOL_VERSION",
    "FrameReader",
    "ProtocolError",
    "client_handshake",
    "decode_frame",
    "encode_frame",
    "handshake_proof",
    "read_frame",
    "recv_exactly",
    "valid_nonce",
    "write_frame",
]

# Wire protocol version, exchanged in `hello`; the peer must answer with the
# same number (app/core/rpc.hpp: kProtocolVersion).
PROTOCOL_VERSION = 2

HEADER_LEN = struct.Struct("<I")
PAYLOAD_LEN = struct.Struct("<Q")
MAX_HEADER = 64 << 20          # a header larger than this is a corrupt stream
MAX_PAYLOAD = 32 << 30         # 32 GiB of tensors in one request (the application reads replies up to its own, lower cap)
# Nothing legitimate is large before `hello`: a hello frame is a few hundred
# bytes. Holding an unauthenticated peer to this keeps it from making the
# worker allocate, or wait for, anything worth the trouble.
MAX_PREAUTH_FRAME = 16 << 10
# Most tensors one frame may describe (kMaxTensors in app/core/rpc.hpp): a
# request or reply carries a handful.
MAX_TENSORS = 64

# protocol dtype name -> numpy little-endian dtype
DTYPES: Dict[str, np.dtype] = {
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
    "uint8": np.dtype("u1"),
    "int8": np.dtype("i1"),
    "uint16": np.dtype("<u2"),
    "int16": np.dtype("<i2"),
    "uint32": np.dtype("<u4"),
    "int32": np.dtype("<i4"),
    "uint64": np.dtype("<u8"),
    "int64": np.dtype("<i8"),
}
_NAMES = {v: k for k, v in DTYPES.items()}

Tensors = Union[Dict[str, np.ndarray], Sequence[Tuple[str, np.ndarray]], None]


class ProtocolError(RuntimeError):
    pass


def dtype_name(a: np.ndarray) -> str:
    """Protocol name of an array's dtype (bool becomes uint8)."""
    d = a.dtype
    if d == np.bool_:
        return "uint8"
    d = d.newbyteorder("<") if d.byteorder == ">" else d
    key = np.dtype(d.str.replace(">", "<").replace("|", ""))
    for name, nd in DTYPES.items():
        if nd == key or nd.kind == d.kind and nd.itemsize == d.itemsize:
            return name
    raise ProtocolError(f"unsupported tensor dtype {a.dtype}")


def _items(tensors: Tensors) -> List[Tuple[str, np.ndarray]]:
    if not tensors:
        return []
    if isinstance(tensors, dict):
        return list(tensors.items())
    return list(tensors)


def encode_frame(header: Dict[str, Any], tensors: Tensors = None) -> bytes:
    """Serialize one frame. `tensors` are appended to the header's "tensors"
    list and their bytes concatenated into the payload."""
    header = dict(header)
    descriptors: List[Dict[str, Any]] = []
    chunks: List[bytes] = []
    offset = 0
    for name, array in _items(tensors):
        a = np.ascontiguousarray(array)
        if a.dtype == np.bool_:
            a = a.astype(np.uint8)
        name_ = dtype_name(a)
        a = a.astype(DTYPES[name_], copy=False)
        data = a.tobytes(order="C")
        descriptors.append({"name": str(name), "dtype": name_, "shape": [int(s) for s in a.shape],
                            "offset": offset, "nbytes": len(data)})
        chunks.append(data)
        offset += len(data)
    header["tensors"] = descriptors
    hb = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    parts = [HEADER_LEN.pack(len(hb)), hb, PAYLOAD_LEN.pack(offset)]
    parts.extend(chunks)
    return b"".join(parts)


def _numel(shape: Tuple[int, ...], name: str) -> int:
    """Element count of a shape that came off the wire. Python integers do not
    wrap, but an absurd product must still be refused before it is multiplied
    by an item size and used to size an allocation, so the running product is
    bounded as it goes (`checkedProduct` on the C++ side)."""
    n = 1
    for s in shape:
        if s < 0:
            raise ProtocolError(f"tensor '{name}': negative extent in shape {shape}")
        n *= s
        if n > MAX_PAYLOAD:
            raise ProtocolError(f"tensor '{name}': shape {shape} exceeds {MAX_PAYLOAD} elements")
    return n


def _reject_constant(name: str):
    # json.loads would happily make NaN and Infinity out of the wire; a
    # descriptor with an infinite extent then ended the process in int()
    raise ProtocolError(f"header contains the non-number {name}")


def _load_header(data: bytes) -> Dict[str, Any]:
    try:
        header = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"header is not JSON: {e}") from e
    if not isinstance(header, dict):
        raise ProtocolError("header is not a JSON object")
    return header


def _count(value: Any, what: str) -> int:
    """A descriptor number: a non-negative JSON integer, nothing else (a
    float such as 1e309 is not truncated, rounded or overflowed into one)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{what} is not an integer: {value!r}")
    if value < 0:
        raise ProtocolError(f"{what} is negative: {value}")
    return value


def _decode_tensors(header: Dict[str, Any], payload: memoryview) -> Dict[str, np.ndarray]:
    descriptors = header.get("tensors") or []
    if not isinstance(descriptors, list):
        raise ProtocolError("header 'tensors' is not an array")
    if len(descriptors) > MAX_TENSORS:
        raise ProtocolError(f"a frame describing {len(descriptors)} tensors (at most {MAX_TENSORS})")
    out: Dict[str, np.ndarray] = {}
    # The descriptors tile the payload in order: each starts at or after the
    # end of the one before, so no byte is decoded twice and together they
    # never claim more than the payload holds.
    previous_end = 0
    for d in descriptors:
        if not isinstance(d, dict):
            raise ProtocolError(f"malformed tensor descriptor {d!r}")
        try:
            name = str(d["name"])
            dtype = DTYPES[str(d["dtype"])]
            raw_shape = d["shape"]
            if not isinstance(raw_shape, list):
                raise ProtocolError(f"tensor '{name}': shape is not an array")
            shape = tuple(_count(s, f"tensor '{name}': shape extent") for s in raw_shape)
            offset = _count(d["offset"], f"tensor '{name}': offset")
            nbytes = _count(d["nbytes"], f"tensor '{name}': nbytes")
        except (KeyError, TypeError, ValueError) as e:
            raise ProtocolError(f"malformed tensor descriptor {d!r}: {e}") from e
        expected = _numel(shape, name) * dtype.itemsize if shape else dtype.itemsize
        # Written the way rpc.cpp writes it, so no sum of peer-supplied
        # numbers is formed before it is known to fit the payload.
        if offset < 0 or nbytes < 0 or nbytes > len(payload) or offset > len(payload) - nbytes:
            raise ProtocolError(f"tensor '{name}': {nbytes} bytes at {offset} do not fit the payload ({len(payload)} bytes)")
        if nbytes != expected:
            raise ProtocolError(f"tensor '{name}': {nbytes} bytes do not match shape {shape} of {dtype.name}")
        if offset < previous_end:
            raise ProtocolError(f"tensor '{name}' overlaps the one before it, or is out of order")
        if name in out:
            raise ProtocolError(f"tensor '{name}' is described twice")
        previous_end = offset + nbytes
        arr = np.frombuffer(payload[offset:offset + nbytes], dtype=dtype).reshape(shape)
        out[name] = arr.copy()  # own the memory: the buffer is reused by the reader
    return out


def decode_frame(buffer: bytearray, max_header: int = MAX_HEADER,
                 max_payload: int = MAX_PAYLOAD) -> Optional[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    """Consume one complete frame from the front of `buffer`; None when it holds
    less than a frame. Raises ProtocolError on a malformed frame or on one
    larger than the caps (which a caller lowers before authentication)."""
    if len(buffer) < HEADER_LEN.size:
        return None
    (hlen,) = HEADER_LEN.unpack_from(buffer, 0)
    if hlen > max_header:
        raise ProtocolError(f"header length {hlen} exceeds {max_header}")
    pos = HEADER_LEN.size + hlen
    if len(buffer) < pos + PAYLOAD_LEN.size:
        return None
    (plen,) = PAYLOAD_LEN.unpack_from(buffer, pos)
    if plen > max_payload:
        raise ProtocolError(f"payload length {plen} exceeds {max_payload}")
    end = pos + PAYLOAD_LEN.size + plen
    if len(buffer) < end:
        return None
    header = _load_header(bytes(buffer[HEADER_LEN.size:pos]))
    if not isinstance(header, dict):
        raise ProtocolError("header is not a JSON object")
    payload = memoryview(buffer)[pos + PAYLOAD_LEN.size:end]
    tensors = _decode_tensors(header, payload)
    del payload
    del buffer[:end]
    return header, tensors


class FrameReader:
    """Incremental decoder: feed() bytes as they arrive, take complete frames."""

    def __init__(self, max_header: int = MAX_HEADER, max_payload: int = MAX_PAYLOAD) -> None:
        self._buf = bytearray()
        self.max_header = max_header
        self.max_payload = max_payload

    def feed(self, data: bytes) -> List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
        self._buf.extend(data)
        frames = []
        while True:
            frame = decode_frame(self._buf, self.max_header, self.max_payload)
            if frame is None:
                break
            frames.append(frame)
        return frames

    @property
    def pending(self) -> int:
        return len(self._buf)


def recv_exactly(sock: socket.socket, n: int, keep_waiting: Optional[Callable[[], bool]] = None,
                 poll: float = 0.5) -> bytes:
    """Read n bytes or raise ConnectionError when the peer closes.

    With `keep_waiting`, each chunk is waited for `poll` seconds at a time and
    TimeoutError is raised as soon as keep_waiting() is false -- checked while
    waiting and before every chunk, so neither a peer that stops halfway
    through a frame nor one that drips it a byte at a time holds the reader
    past the caller's deadline or stop flag. The socket itself stays blocking:
    a large transfer, in either direction, is never cut short by a timeout."""
    chunks = []
    remaining = n
    while remaining > 0:
        if keep_waiting is not None:
            while True:
                if not keep_waiting():
                    raise TimeoutError(f"gave up on an incomplete frame ({n - remaining} of {n} bytes arrived)")
                ready, _, _ = select.select([sock], [], [], poll)
                if ready:
                    break
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket, max_header: int = MAX_HEADER, max_payload: int = MAX_PAYLOAD,
               keep_waiting: Optional[Callable[[], bool]] = None) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Blocking read of one frame from a socket.

    Each length is checked against its cap *before* the bytes it announces are
    read, so an oversize frame costs the peer one refused read and this side no
    allocation at all; `recv_exactly` then accumulates in 1 MiB pieces.
    `keep_waiting` (see recv_exactly) bounds the wait for the whole frame."""
    (hlen,) = HEADER_LEN.unpack(recv_exactly(sock, HEADER_LEN.size, keep_waiting))
    if hlen > max_header:
        raise ProtocolError(f"header length {hlen} exceeds {max_header}")
    hb = recv_exactly(sock, hlen, keep_waiting)
    (plen,) = PAYLOAD_LEN.unpack(recv_exactly(sock, PAYLOAD_LEN.size, keep_waiting))
    if plen > max_payload:
        raise ProtocolError(f"payload length {plen} exceeds {max_payload}")
    payload = recv_exactly(sock, plen, keep_waiting) if plen else b""
    header = _load_header(hb)
    return header, _decode_tensors(header, memoryview(payload))


def write_frame(sock: socket.socket, header: Dict[str, Any], tensors: Tensors = None) -> None:
    sock.sendall(encode_frame(header, tensors))


# --- the handshake ----------------------------------------------------------------------


def handshake_proof(token: str, role: str, client_nonce: str, server_nonce: str) -> str:
    """Hex HMAC-SHA256(token, "sirius-worker-auth/2|<role>|<client>|<server>"):
    what each side sends to show that it holds the token without sending it
    (rpc::handshakeProof in app/core/rpc.cpp)."""
    message = f"sirius-worker-auth/2|{role}|{client_nonce}|{server_nonce}".encode()
    return hmac.new((token or "").encode("utf-8"), message, hashlib.sha256).hexdigest()


def valid_nonce(value: Any) -> bool:
    """32 to 128 lower-case hex digits."""
    return (isinstance(value, str) and 32 <= len(value) <= 128
            and all(c in "0123456789abcdef" for c in value))


def client_handshake(sock: socket.socket, token: str, protocol_version: int = PROTOCOL_VERSION,
                     first_id: int = 1) -> Dict[str, Any]:
    """The client's half of the handshake on a connected socket. Returns the
    worker's last reply header: an error when it refused the hello or the
    proof, else the auth reply, whose result holds the capabilities. Raises
    ProtocolError -- having sent nothing but its nonce -- when the worker
    cannot prove that it holds `token`. Uses request ids first_id and
    first_id + 1."""
    client_nonce = secrets.token_hex(16)
    write_frame(sock, {"id": first_id, "type": "request", "method": "hello",
                       "params": {"protocol_version": protocol_version, "client_nonce": client_nonce}})
    header, _ = read_frame(sock)
    if header.get("type") != "result":
        return header
    result = header.get("result") or {}
    server_nonce = result.get("server_nonce")
    proof = result.get("server_proof")
    if (not valid_nonce(server_nonce) or server_nonce == client_nonce or not isinstance(proof, str)
            or not hmac.compare_digest(proof.encode("utf-8"),
                                       handshake_proof(token, "worker", client_nonce, server_nonce).encode("utf-8"))):
        raise ProtocolError("the worker could not prove that it holds the token (a wrong token, or something else "
                            "answers on that port); nothing else was sent to it")
    write_frame(sock, {"id": first_id + 1, "type": "request", "method": "auth",
                       "params": {"client_proof": handshake_proof(token, "client", client_nonce, server_nonce)}})
    header, _ = read_frame(sock)
    return header
