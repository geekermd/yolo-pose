"""A minimal X11 image viewer speaking the wire protocol directly.

This gives live camera preview with no dependency on OpenCV, Pillow, Tk or
python-xlib: a TCP/Unix connection to the X server plus raw X protocol requests.
Only TrueColor visuals at depth 16/24/32 are supported, which covers every
modern X server.
"""

from __future__ import annotations

import os
import socket
import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class X11Error(RuntimeError):
    """Raised when the X server cannot be reached or refuses a request."""


# Request opcodes
_CREATE_WINDOW = 1
_MAP_WINDOW = 8
_PUT_IMAGE = 72
_CREATE_GC = 55
_FREE_GC = 60
_DESTROY_WINDOW = 4
_IMAGE_TEXT_8 = 0
_IMAGE_Z_PIXMAP = 2
_EVENT_EXPOSURE = 12
_EVENT_KEY_PRESS = 2
_EVENT_BUTTON_PRESS = 4
_EVENT_CLIENT_MESSAGE = 33

_FLAG_BACKGROUND_PIXEL = 0x00000002
_FLAG_EVENT_MASK = 0x00000800
_FLAG_EXPOSURE = 0x00008000
_FLAG_KEY_PRESS = 0x00000001
_FLAG_BUTTON_PRESS = 0x00000004

# X11 keysyms we care about
_KEYSYM_Q = 0x0071
_KEYSYM_ESCAPE = 0xFF1B


@dataclass
class ScreenInfo:
    root: int
    root_visual: int
    root_depth: int
    width: int
    height: int
    white: int
    black: int
    resource_base: int


def _display_socket() -> Tuple[socket.socket, bool]:
    """Connect to the X server, honouring ``DISPLAY`` and XAUTHORITY.

    Modern Xwayland/Xorg listen on an *abstract* unix socket as well as the
    filesystem path.  Some Xwayland builds only answer on the abstract one, so
    try that first and fall back to /tmp/.X11-unix/X<n>.
    """
    display = os.environ.get("DISPLAY", "")
    if not display:
        raise X11Error("DISPLAY is not set; cannot open a window")
    host, _, number = display.rpartition(":")
    screen = 0
    if "." in number:
        number, _, screen = number.partition(".")

    if host in ("", "unix"):
        abstract = "\0/tmp/.X11-unix/X" + number
        path = "/tmp/.X11-unix/X" + number
        errors = []
        for address in (abstract, path):
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                connection.connect(address)
                return connection, True
            except OSError as exc:
                errors.append(f"{address!r}: {exc}")
                connection.close()
        raise X11Error("cannot reach the X server: " + "; ".join(errors))

    connection = socket.create_connection((host, 6000 + int(number)))
    return connection, False


def _authorisation() -> Tuple[bytes, bytes]:
    """Read the MIT-MAGIC-COOKIE for the current display from XAUTHORITY.

    Returns ``(protocol_name, cookie)``.  The authority file is a sequence of
    big-endian length-prefixed records; ``$XAUTHORITY`` matters because GNOME's
    Wayland session stores the cookie outside ``~/.Xauthority``.
    """
    authority = os.environ.get("XAUTHORITY") or os.path.expanduser("~/.Xauthority")
    if not os.path.exists(authority):
        return b"", b""
    try:
        with open(authority, "rb") as handle:
            data = handle.read()
    except OSError:
        return b"", b""

    display = os.environ.get("DISPLAY", "")
    wanted_number = display.rpartition(":")[2].split(".")[0]
    hostname = socket.gethostname().encode()

    records = []
    position = 0
    def take() -> bytes:
        nonlocal position
        if position + 2 > len(data):
            return b""
        length = struct.unpack_from(">H", data, position)[0]
        position += 2
        value = data[position:position + length]
        position += length
        return value

    while position + 2 <= len(data):
        family = struct.unpack_from(">H", data, position)[0]
        position += 2
        address = take()
        number = take()
        name = take()
        cookie = take()
        records.append((family, address, number, name, cookie))

    if not records:
        return b"", b""

    def rank(record) -> int:
        family, address, number, name, _cookie = record
        if name != b"MIT-MAGIC-COOKIE-1":
            return 99
        score = 0
        # Prefer the entry whose display number matches, then hostname matches.
        if number.decode("latin-1") in (wanted_number, ""):
            score -= 2
        if address in (hostname, b"", b"localhost"):
            score -= 1
        if family in (256, 65535):   # FamilyLocal / FamilyWild
            score -= 1
        return score

    best = min(records, key=rank)
    _family, _address, _number, name, cookie = best
    return name, cookie


def _pad4(value: bytes) -> bytes:
    """Pad an auth field to a multiple of four bytes (X11 spec)."""
    return value + b"\x00" * ((4 - len(value) % 4) % 4)


class X11Viewer:
    """A resizable X11 window that displays RGB frames via ``PutImage``."""

    def __init__(self, width: int, height: int, title: str = "yolo-pose"):
        self.width = int(width)
        self.height = int(height)
        self.title = title
        self.connection, _unix = _display_socket()
        self.screen: Optional[ScreenInfo] = None
        self.window = 0
        self.gc = 0
        self._closed = False
        self._next_id = 0
        self._depth = 24
        self._bytes_per_pixel = 4
        self._byte_order = "little"
        self._pending_expose = True
        self._connect()
        self._create_window()
        self._create_gc()

    # -- setup -------------------------------------------------------------- #
    def _connect(self) -> None:
        name, cookie = _authorisation()
        name_padded = _pad4(name)
        cookie_padded = _pad4(cookie)
        # The length field counts the *padded* auth data; getting this wrong
        # makes the server wait forever for bytes that never arrive.
        prefix = struct.pack("<BBHHH", 0x6C, 0, 11, 0,
                             len(name_padded) + len(cookie_padded))
        self.connection.sendall(prefix + name_padded + cookie_padded)
        reply = self._recv_exact(8)
        status = reply[0]
        if status != 1:
            reason = self._recv_exact(struct.unpack_from("<H", reply, 6)[0])
            raise X11Error(f"X server refused the connection: {reason.decode('latin-1')}")
        length = struct.unpack_from("<H", reply, 6)[0]
        body = self._recv_exact(length * 4)
        self._parse_setup(body, reply)

    def _parse_setup(self, body: bytes, header: bytes) -> None:
        self._byte_order = "little"
        vendor_len = struct.unpack_from("<H", body, 16)[0]
        position = 32 + vendor_len
        position += (4 - position % 4) % 4
        count = body[position]
        position += 8
        if count == 0:
            raise X11Error("X server reports no screens")
        root, _default_colormap, white, black = struct.unpack_from("<IIII", body, position)
        width, height = struct.unpack_from("<HH", body, position + 20)
        root_visual = struct.unpack_from("<I", body, position + 32)[0]
        depth = body[position + 38]
        self._depth = depth
        self._bytes_per_pixel = 4 if depth in (24, 32) else 2
        self.screen = ScreenInfo(root=root, root_visual=root_visual, root_depth=depth,
                                 width=width, height=height, white=white, black=black,
                                 resource_base=0)
        self.resource_base = struct.unpack_from("<I", header, 4)[0]
        self._next_id = self.resource_base

    def _recv_exact(self, count: int) -> bytes:
        chunks = []
        remaining = count
        while remaining > 0:
            chunk = self.connection.recv(remaining)
            if not chunk:
                raise X11Error("X connection closed unexpectedly")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _allocate_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # -- window ------------------------------------------------------------- #
    def _create_window(self) -> None:
        assert self.screen is not None
        self.window = self._allocate_id()
        values = struct.pack("<II", self.screen.black, _FLAG_EXPOSURE
                             | _FLAG_KEY_PRESS | _FLAG_BUTTON_PRESS)
        request = struct.pack(
            "<BBHIIHHHHIII",
            _CREATE_WINDOW, 0, 8 + len(values) // 4,
            self.window, self.screen.root,
            0, 0, self.width, self.height, 0, 1,
            self.screen.root_visual,
            _FLAG_BACKGROUND_PIXEL | _FLAG_EVENT_MASK,
        ) + values
        self._send_request(request)
        self._send_request(struct.pack("<BBHII", _MAP_WINDOW, 0, 2, self.window, 0))
        self.connection.sendall(b"")
        self._flush()

    def _create_gc(self) -> None:
        assert self.screen is not None
        self.gc = self._allocate_id()
        values = struct.pack("<II", self.screen.black, self.screen.white)
        request = struct.pack("<BBHIII", _CREATE_GC, 0, 3 + len(values) // 4,
                              self.gc, self.window, 0x00000004 | 0x00000008) + values
        self._send_request(request)

    def _send_request(self, payload: bytes) -> None:
        self.connection.sendall(payload)

    def _flush(self) -> None:
        try:
            self.connection.sendall(b"")
        except OSError:
            pass

    # -- drawing ------------------------------------------------------------ #
    def _pack_pixels(self, image: np.ndarray) -> bytes:
        """Convert an ``(H, W, 3)`` uint8 frame into X11 pixel data."""
        height, width = image.shape[:2]
        if self._depth in (24, 32):
            pixels = ((image[:, :, 0].astype(np.uint32) << 16)
                      | (image[:, :, 1].astype(np.uint32) << 8)
                      | image[:, :, 2].astype(np.uint32))
            data = pixels.astype("<u4").tobytes()
        else:
            r = (image[:, :, 0].astype(np.uint16) >> 3) << 11
            g = (image[:, :, 1].astype(np.uint16) >> 2) << 5
            b = (image[:, :, 2].astype(np.uint16) >> 3)
            data = (r | g | b).astype("<u2").tobytes()

        bytes_per_line = width * self._bytes_per_pixel
        padded = (bytes_per_line + 3) // 4 * 4
        if padded != bytes_per_line:
            rows = np.frombuffer(data, dtype=np.uint8).reshape(height, bytes_per_line)
            data = np.concatenate(
                [rows, np.zeros((height, padded - bytes_per_line), dtype=np.uint8)],
                axis=1).tobytes()
        return data

    def show(self, image: np.ndarray) -> None:
        """Display one RGB frame, resizing the window if the frame size changed."""
        if self._closed:
            raise X11Error("viewer is closed")
        frame = np.ascontiguousarray(image)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        height, width = frame.shape[:2]
        if (width, height) != (self.width, self.height):
            self.width, self.height = width, height
            self._configure_window(width, height)

        data = self._pack_pixels(frame)
        units = len(data) // 4
        request = struct.pack(
            "<BBHIIHHhhBBB",
            _PUT_IMAGE, _IMAGE_Z_PIXMAP, 6 + units,
            self.window, self.gc, width, height, 0, 0,
            self._depth, _IMAGE_TEXT_8, 0,
        ) + data
        self._send_request(request)

    def _configure_window(self, width: int, height: int) -> None:
        request = struct.pack("<BBHIIHHHHH", 12, 0, 7, self.window, 0, 0,
                              0, width, height, 0)
        self._send_request(request)

    # -- event loop --------------------------------------------------------- #
    def poll_events(self, block: bool = False) -> bool:
        """Process pending events; returns ``False`` when the user asked to quit."""
        self._flush()
        if block:
            self.connection.settimeout(0.2)
        else:
            self.connection.settimeout(0.0)
        try:
            while True:
                try:
                    header = self.connection.recv(32)
                except (socket.timeout, BlockingIOError):
                    break
                if not header:
                    return False
                if len(header) < 32:
                    header += self._recv_exact(32 - len(header))
                parsed = self._handle_event(header)
                if not parsed:
                    return False
        finally:
            self.connection.settimeout(None)
        return True

    def _handle_event(self, header: bytes) -> bool:
        code = header[0] & 0x7F
        if code == _EVENT_EXPOSURE:
            self._pending_expose = True
        elif code == _EVENT_KEY_PRESS:
            keysym = struct.unpack_from("<I", header, 8)[0]
            if keysym in (_KEYSYM_Q, _KEYSYM_ESCAPE):
                return False
        elif code == _EVENT_BUTTON_PRESS:
            pass
        return True

    @property
    def needs_redraw(self) -> bool:
        if self._pending_expose:
            self._pending_expose = False
            return True
        return False

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.gc:
                self._send_request(struct.pack("<BBHI", _FREE_GC, 0, 2, self.gc))
            if self.window:
                self._send_request(struct.pack("<BBHI", _DESTROY_WINDOW, 0, 2, self.window))
            self._flush()
        except OSError:
            pass
        try:
            self.connection.close()
        except OSError:
            pass

    def __enter__(self) -> "X11Viewer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<X11Viewer {self.width}x{self.height} depth={self._depth}>"


def viewer_available() -> Tuple[bool, str]:
    """Check whether a window can be opened, returning ``(ok, reason)``."""
    if not os.environ.get("DISPLAY"):
        return False, "DISPLAY is not set (no graphical session)"
    try:
        socket_path = "/tmp/.X11-unix/X" + os.environ["DISPLAY"].rpartition(":")[2].split(".")[0]
        if os.environ["DISPLAY"].startswith(":") and not os.path.exists(socket_path):
            return False, f"X socket {socket_path} is missing"
    except Exception:
        pass
    return True, ""
