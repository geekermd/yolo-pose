"""Live frame display backends.

Three strategies are attempted, in order:

1. :class:`XlibViewer` — calls ``libX11`` through :mod:`ctypes`.  Works on any
   distribution with libX11 (a dependency of essentially every desktop), and is
   architecture independent because the library is resolved at runtime.
2. :class:`X11Viewer` — a pure-Python implementation of the X11 wire protocol
   (:mod:`yolopose.render.x11`), for systems without libX11 bindings.
3. :class:`HeadlessWriter` — writes annotated frames to disk, for machines with
   no display at all.

``open_viewer`` picks the best available option and reports which one it chose.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time
from typing import Optional, Tuple

import numpy as np

from ..vision.imaging import save_image


class ViewerError(RuntimeError):
    """Raised when no display backend can be initialised."""


# --------------------------------------------------------------------------- #
# ctypes structures for libX11
# --------------------------------------------------------------------------- #

# XImage field offsets (from Xlib.h).  The struct has a tail of function
# pointers that we never touch, so the object is allocated as a raw zeroed
# buffer of XIMAGE_STRUCT_SIZE bytes and fields are written by offset.
_XIMAGE_STRUCT_SIZE = 136
_IMG_WIDTH = 0        # int
_IMG_HEIGHT = 4       # int
_IMG_XOFFSET = 8      # int
_IMG_FORMAT = 12      # int
_IMG_DATA = 16        # char*
_IMG_BYTE_ORDER = 24  # int
_IMG_BITMAP_UNIT = 28
_IMG_BITMAP_BIT_ORDER = 32
_IMG_BITMAP_PAD = 36
_IMG_DEPTH = 40
_IMG_BYTES_PER_LINE = 44
_IMG_BITS_PER_PIXEL = 48
_IMG_RED_MASK = 56    # unsigned long (8-byte aligned)
_IMG_GREEN_MASK = 64
_IMG_BLUE_MASK = 72

_LSB_FIRST = 0
_ZPIXMAP = 2


class _Visual(ctypes.Structure):
    _fields_ = [
        ("ext_data", ctypes.c_void_p),
        ("visualid", ctypes.c_ulong),
        ("class_", ctypes.c_int),
        ("red_mask", ctypes.c_ulong),
        ("green_mask", ctypes.c_ulong),
        ("blue_mask", ctypes.c_ulong),
        ("bits_per_rgb", ctypes.c_int),
        ("map_entries", ctypes.c_int),
    ]


class _XWindowAttributes(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int), ("y", ctypes.c_int),
        ("width", ctypes.c_int), ("height", ctypes.c_int),
        ("border_width", ctypes.c_int), ("depth", ctypes.c_int),
        ("visual", ctypes.POINTER(_Visual)),
        ("root", ctypes.c_ulong),
        ("class_", ctypes.c_int),
        ("bit_gravity", ctypes.c_int), ("win_gravity", ctypes.c_int),
        ("backing_store", ctypes.c_int),
        ("backing_planes", ctypes.c_ulong),
        ("backing_pixel", ctypes.c_ulong),
        ("save_under", ctypes.c_int),
        ("colormap", ctypes.c_ulong),
        ("map_installed", ctypes.c_int),
        ("map_state", ctypes.c_int),
        ("all_event_masks", ctypes.c_long),
        ("your_event_mask", ctypes.c_long),
        ("do_not_propagate_mask", ctypes.c_long),
        ("override_redirect", ctypes.c_int),
        ("screen", ctypes.c_void_p),
    ]


class _XEvent(ctypes.Union):
    _fields_ = [("type", ctypes.c_int), ("pad", ctypes.c_long * 24)]


class _XSizeHints(ctypes.Structure):
    """XSizeHints from Xlib.h (fields are only valid when flagged)."""

    _fields_ = [
        ("flags", ctypes.c_long),
        ("x", ctypes.c_int), ("y", ctypes.c_int),
        ("width", ctypes.c_int), ("height", ctypes.c_int),
        ("min_width", ctypes.c_int), ("min_height", ctypes.c_int),
        ("max_width", ctypes.c_int), ("max_height", ctypes.c_int),
        ("width_inc", ctypes.c_int), ("height_inc", ctypes.c_int),
        ("min_aspect_x", ctypes.c_int), ("min_aspect_y", ctypes.c_int),
        ("max_aspect_x", ctypes.c_int), ("max_aspect_y", ctypes.c_int),
        ("base_width", ctypes.c_int), ("base_height", ctypes.c_int),
        ("win_gravity", ctypes.c_int),
    ]


USPosition, USSize = 1 << 0, 1 << 1
PPosition, PSize = 1 << 2, 1 << 3
PMinSize, PMaxSize = 1 << 4, 1 << 5
PResizeInc, PAspect = 1 << 6, 1 << 7
PBaseSize, PWinGravity = 1 << 8, 1 << 9

_STRUCTURE_NOTIFY = 0x00020000


_LIBX11_CANDIDATES = ("libX11.so.6", "libX11.so", "libX11.dylib")


def _load_libx11():
    for name in _LIBX11_CANDIDATES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    found = ctypes.util.find_library("X11")
    if found:
        try:
            return ctypes.CDLL(found)
        except OSError:
            pass
    return None


def _bind(lib) -> None:
    """Declare argument/return types so ctypes marshals correctly."""
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XCloseDisplay.restype = ctypes.c_int
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    lib.XDefaultScreen.restype = ctypes.c_int
    lib.XDefaultScreen.argtypes = [ctypes.c_void_p]
    lib.XRootWindow.restype = ctypes.c_ulong
    lib.XRootWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.XDefaultVisual.restype = ctypes.POINTER(_Visual)
    lib.XDefaultVisual.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.XDefaultDepth.restype = ctypes.c_int
    lib.XDefaultDepth.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.XCreateSimpleWindow.restype = ctypes.c_ulong
    lib.XCreateSimpleWindow.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_ulong]
    lib.XStoreName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_char_p]
    lib.XMapWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    lib.XSelectInput.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_long]
    # The XImage is a raw buffer with a manual layout, so no struct type is
    # needed here; XPutImage takes the XImage pointer as an opaque void*.
    lib.XPutImage.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
    lib.XFlush.argtypes = [ctypes.c_void_p]
    lib.XFreeGC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.XPending.restype = ctypes.c_int
    lib.XPending.argtypes = [ctypes.c_void_p]
    lib.XNextEvent.argtypes = [ctypes.c_void_p, ctypes.POINTER(_XEvent)]
    lib.XDestroyWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    lib.XGetWindowAttributes.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(_XWindowAttributes)]
    lib.XResizeWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                  ctypes.c_uint, ctypes.c_uint]
    lib.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.XSetWMNormalHints.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                      ctypes.c_void_p]
    lib.XGetWMNormalHints.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                      ctypes.c_void_p, ctypes.c_void_p]


class XlibViewer:
    """Displays frames in an X11 window through libX11 (ctypes)."""

    def __init__(self, width: int, height: int, title: str = "yolo-pose",
                 display: Optional[str] = None, resizable: bool = True):
        self.lib = _load_libx11()
        if self.lib is None:
            raise ViewerError("libX11 could not be loaded")
        _bind(self.lib)

        display_name = (display or os.environ.get("DISPLAY") or "").encode()
        self.display = self.lib.XOpenDisplay(display_name or None)
        if not self.display:
            raise ViewerError(
                f"XOpenDisplay failed for DISPLAY={display_name.decode() or '(unset)'}"
            )

        self.screen = self.lib.XDefaultScreen(self.display)
        self.root = self.lib.XRootWindow(self.display, self.screen)
        self.visual = self.lib.XDefaultVisual(self.display, self.screen)
        self.depth = self.lib.XDefaultDepth(self.display, self.screen)

        self.window = self.lib.XCreateSimpleWindow(
            self.display, self.root, 0, 0, int(width), int(height), 0, 0, 0)
        if not self.window:
            self.lib.XCloseDisplay(self.display)
            raise ViewerError("XCreateSimpleWindow failed")
        self.lib.XStoreName(self.display, self.window, title.encode())
        # Expose | KeyPress | ButtonPress | StructureNotify (so resizes arrive)
        self.lib.XSelectInput(self.display, self.window,
                              0x00008000 | 0x00000001 | 0x00000004
                              | _STRUCTURE_NOTIFY)
        self.resizable = bool(resizable)
        self._apply_size_hints(int(width), int(height), self.resizable)
        self.lib.XMapWindow(self.display, self.window)
        self.lib.XFlush(self.display)

        self.width = int(width)
        self.height = int(height)
        self._image_width = int(width)
        self._image_height = int(height)
        self.pixels = None
        self.image = None
        self.gc = None
        self._closed = False
        self._needs_redraw = True
        self._size_dirty = True

        visual = self.visual.contents
        self._red_mask = int(visual.red_mask)
        self._green_mask = int(visual.green_mask)
        self._blue_mask = int(visual.blue_mask)
        self._shift = (self._mask_shift(self._red_mask),
                       self._mask_shift(self._green_mask),
                       self._mask_shift(self._blue_mask))
        self._allocate_image(self.width, self.height)

    # -- internals ---------------------------------------------------------- #
    @staticmethod
    def _mask_shift(mask: int) -> int:
        if not mask:
            return 0
        shift = 0
        while not (mask >> shift) & 1:
            shift += 1
        return shift

    def _allocate_image(self, width: int, height: int) -> None:
        """Allocate the pixel buffer and a zeroed XImage describing it."""
        bytes_per_line = width * 4
        self.pixels = ctypes.create_string_buffer(bytes_per_line * height)
        self.bytes_per_line = bytes_per_line

        self.image = ctypes.create_string_buffer(_XIMAGE_STRUCT_SIZE)
        base = ctypes.addressof(self.image)

        def put_int(offset: int, value: int) -> None:
            ctypes.memmove(base + offset, ctypes.byref(ctypes.c_int(value)), 4)

        def put_long(offset: int, value: int) -> None:
            ctypes.memmove(base + offset, ctypes.byref(ctypes.c_ulong(value)), 8)

        put_int(_IMG_WIDTH, width)
        put_int(_IMG_HEIGHT, height)
        put_int(_IMG_XOFFSET, 0)
        put_int(_IMG_FORMAT, _ZPIXMAP)
        put_long(_IMG_DATA, ctypes.addressof(self.pixels))
        put_int(_IMG_BYTE_ORDER, _LSB_FIRST)
        put_int(_IMG_BITMAP_UNIT, 32)
        put_int(_IMG_BITMAP_BIT_ORDER, _LSB_FIRST)
        put_int(_IMG_BITMAP_PAD, 32)
        put_int(_IMG_DEPTH, self.depth)
        put_int(_IMG_BYTES_PER_LINE, bytes_per_line)
        put_int(_IMG_BITS_PER_PIXEL, 32)
        put_long(_IMG_RED_MASK, self._red_mask)
        put_long(_IMG_GREEN_MASK, self._green_mask)
        put_long(_IMG_BLUE_MASK, self._blue_mask)

        self._image_width = int(width)
        self._image_height = int(height)

        if self.gc is None:
            self.lib.XCreateGC.restype = ctypes.c_void_p
            self.lib.XCreateGC.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
                                           ctypes.c_ulong, ctypes.c_void_p]
            self.gc = self.lib.XCreateGC(self.display, self.window, 0, None)

    def _image_pointer(self):
        return ctypes.cast(self.image, ctypes.c_void_p)

    def _pack(self, frame: np.ndarray) -> None:
        """Fill the XImage buffer with packed pixels for the current visual."""
        height, width = frame.shape[:2]
        pointer = ctypes.cast(self.pixels, ctypes.POINTER(ctypes.c_uint8))
        array = np.ctypeslib.as_array(pointer, shape=(width * height * 4,))
        array = array.reshape(height, width, 4)

        # Generic path: shift each channel into its mask and OR the results.
        values = (frame[:, :, 0].astype(np.uint32) << self._shift[2])
        values |= (frame[:, :, 1].astype(np.uint32) << self._shift[1])
        values |= (frame[:, :, 2].astype(np.uint32) << self._shift[0])
        array[:, :, 0] = values & 0xFF
        array[:, :, 1] = (values >> 8) & 0xFF
        array[:, :, 2] = (values >> 16) & 0xFF
        array[:, :, 3] = (values >> 24) & 0xFF

    # -- public API --------------------------------------------------------- #
    def show(self, frame: np.ndarray, window_size: Optional[Tuple[int, int]] = None) -> None:
        """Display a frame, optionally stretched to ``window_size``.

        When ``window_size`` is larger than the frame, X11 performs the scaling
        during ``XPutImage`` — that costs nothing in Python and avoids a
        multi-millisecond NumPy resize on every frame.
        """
        if self._closed:
            raise ViewerError("viewer is closed")
        image = np.ascontiguousarray(frame)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        height, width = image.shape[:2]

        # The XImage always holds the frame at its native resolution.
        if (width, height) != (self._image_width, self._image_height):
            self._allocate_image(width, height)

        if self.resizable:
            # Follow whatever geometry the X server reports so a user drag or a
            # window-manager resize is honoured immediately.
            live_w, live_h = self.query_window_size()
            target_w, target_h = ((live_w, live_h) if live_w > 0 and live_h > 0
                                  else (window_size or (width, height)))
            self.width, self.height = target_w, target_h
        # In locked mode the geometry never changes, so draw at the fixed size.


        self._pack(image)
        self.lib.XPutImage(self.display, self.window, self.gc,
                           self._image_pointer(), 0, 0, 0, 0, width, height)
        self.lib.XFlush(self.display)
        self._needs_redraw = False

    def _apply_size_hints(self, width: int, height: int,
                          resizable: bool = True) -> None:
        """Tell the window manager whether the window may be resized.

        ``resizable=False`` sets min == max, which is how a window asks to have
        no resize handles at all — the geometry then stays exactly as requested.
        """
        hints = _XSizeHints()
        ctypes.memset(ctypes.byref(hints), 0, ctypes.sizeof(hints))
        hints.x = 0
        hints.y = 0
        hints.win_gravity = 1          # NorthWestGravity
        if resizable:
            hints.flags = PMinSize | PBaseSize | PWinGravity
            hints.min_width = max(160, min(width, 640))
            hints.min_height = max(120, min(height, 480))
        else:
            hints.flags = (USPosition | USSize | PPosition | PSize
                           | PMinSize | PMaxSize | PBaseSize | PWinGravity)
            hints.min_width = hints.max_width = int(width)
            hints.min_height = hints.max_height = int(height)
        hints.base_width = int(width)
        hints.base_height = int(height)
        self.lib.XSetWMNormalHints(self.display, self.window,
                                   ctypes.byref(hints))

    def query_window_size(self) -> Tuple[int, int]:
        """Current window geometry as the X server sees it."""
        attributes = _XWindowAttributes()
        try:
            if self.lib.XGetWindowAttributes(self.display, self.window,
                                             ctypes.byref(attributes)):
                if attributes.width > 0 and attributes.height > 0:
                    return int(attributes.width), int(attributes.height)
        except Exception:
            pass
        return self.width, self.height

    def poll_events(self) -> bool:
        """Drain X events; returns ``False`` when the user closed the window."""
        if self._closed:
            return False
        event = _XEvent()
        while self.lib.XPending(self.display):
            self.lib.XNextEvent(self.display, ctypes.byref(event))
            kind = event.type
            if kind == 12:                       # Expose
                self._needs_redraw = True
            elif kind == 22:                     # ConfigureNotify (resize)
                self._needs_redraw = True
                self._size_dirty = True
            elif kind == 2:                      # KeyPress
                return False
            elif kind == 33:                     # ClientMessage (WM delete)
                return False
        return True

    @property
    def needs_redraw(self) -> bool:
        return self._needs_redraw

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.gc:
                self.lib.XFreeGC(self.display, self.gc)
            if self.window:
                self.lib.XDestroyWindow(self.display, self.window)
            self.lib.XCloseDisplay(self.display)
        except Exception:
            pass
        # The XImage was a plain buffer we own, so nothing to free in Xlib.


class HeadlessWriter:
    """Fallback "viewer" that writes frames to disk instead of a screen."""

    def __init__(self, width: int, height: int, directory: str = ".",
                 prefix: str = "frame", every: int = 10, **kwargs):
        self.directory = directory
        self.prefix = prefix
        self.every = max(1, every)
        self.frame_index = 0
        self.saved = 0
        os.makedirs(directory, exist_ok=True)
        self._needs_redraw = True

    def show(self, frame: np.ndarray) -> None:
        if self.frame_index % self.every == 0:
            path = os.path.join(self.directory, f"{self.prefix}_{self.saved:05d}.png")
            save_image(path, frame)
            self.saved += 1
        self.frame_index += 1
        self._needs_redraw = False

    def poll_events(self) -> bool:
        return True

    @property
    def needs_redraw(self) -> bool:
        return False

    def close(self) -> None:
        pass


def open_viewer(width: int, height: int, title: str = "yolo-pose",
                headless_dir: Optional[str] = None,
                display: Optional[str] = None,
                resizable: bool = True) -> Tuple[object, str]:
    """Return ``(viewer, backend_name)`` choosing the best backend available."""
    problems = []
    try:
        return XlibViewer(width, height, title, display=display,
                          resizable=resizable), "xlib"
    except Exception as exc:
        problems.append(f"xlib: {exc}")

    try:
        from .x11 import X11Viewer
        return X11Viewer(width, height, title), "x11-protocol"
    except Exception as exc:
        problems.append(f"x11-protocol: {exc}")

    if headless_dir:
        return HeadlessWriter(width, height, directory=headless_dir), "headless-files"
    raise ViewerError("no display backend available (" + "; ".join(problems) + ")")
