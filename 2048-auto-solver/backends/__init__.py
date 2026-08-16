"""Per-OS capture and input implementations, hidden behind OS-agnostic protocols.

Everything outside this package (``vision/``, ``app/``, ``ui/``) talks to
:class:`backends.capture.base.CaptureBackend` and :class:`backends.input.base.InputBackend`
only. No other package may import ``mss``, ``dxcam``, ``pynput``, ``pydirectinput``,
``pygetwindow``, or any macOS/Windows-specific module directly -- that keeps the universality
principle (screen pixels in, OS keystrokes out) enforced at the import boundary, not just by
convention.
"""
