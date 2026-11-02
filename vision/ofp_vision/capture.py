"""Camera acquisition.

Two rules learned the hard way on this machine:

1. **Select the device by name, never by index.** An "OBS Virtual Camera" is
   installed and enumerates *before* the real webcam. Opening it succeeds and
   yields a frozen frame, so an index-based grabber silently records a
   motionless image forever.

2. **Open by DirectShow index, not by OpenCV's own enumeration.** OpenCV's
   device numbering has not matched DirectShow's on this box, so we enumerate
   names through DirectShow (pygrabber) and then open that index with the
   DSHOW backend.

3. **A network camera is just a URL.** When `CaptureConfig.source_url` is
   set, both rules above are skipped: a phone app's MJPEG/RTSP stream needs
   no driver and cannot be a stale virtual camera.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import CaptureConfig

log = logging.getLogger(__name__)


def list_directshow_devices() -> list[str]:
    """Device names in DirectShow order. Empty list if enumeration fails."""
    try:
        from pygrabber.dshow_graph import FilterGraph

        graph = FilterGraph()
        return [str(device) for device in graph.get_input_devices()]
    except Exception as exc:  # pragma: no cover - depends on Windows COM
        log.warning("DirectShow enumeration failed: %s", exc)
        return []


@dataclass
class OpenedCamera:
    index: int
    name: str
    backend: str


class CameraSource:
    """Opens the configured camera and yields BGR frames."""

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self.cap = None
        self.opened: OpenedCamera | None = None

    # --- device resolution --------------------------------------------------

    def _looks_virtual(self, name: str) -> bool:
        lowered = name.lower()
        return any(keyword in lowered for keyword in self.config.reject_keywords)

    def resolve_device(self) -> OpenedCamera | None:
        """Pick the device index to open."""
        import cv2

        names = list_directshow_devices()

        if names:
            target = self.config.device_name.lower()
            for index, name in enumerate(names):
                if target in name.lower():
                    return OpenedCamera(index=index, name=name, backend="DSHOW")
            # No exact match: take the first non-virtual device.
            for index, name in enumerate(names):
                if not self._looks_virtual(name):
                    return OpenedCamera(index=index, name=name, backend="DSHOW")
            log.warning("All devices look virtual; falling back to index scan")

        for index in self.config.fallback_indices:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.release()
                return OpenedCamera(index=index, name=f"index {index}", backend="DSHOW")
        return None

    # --- lifecycle -----------------------------------------------------------

    def open(self) -> bool:
        import cv2

        if self.config.source_url:
            return self._open_url(cv2)

        target = self.resolve_device()
        if target is None:
            log.error(
                "No camera could be opened. Check that 'Let desktop apps access "
                "your camera' is ON in Settings > Privacy > Camera."
            )
            return False

        cap = cv2.VideoCapture(target.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            log.error("Could not open camera index %d (%s)", target.index, target.name)
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)

        self.cap = cap
        self.opened = target
        self._log_mode(cap, requested=(self.config.width, self.config.height))
        log.info("Camera opened: index=%d name=%r", target.index, target.name)
        return True

    def _open_url(self, cv2) -> bool:
        """Open a network camera: a phone app's MJPEG/RTSP URL.

        This is the route that needs no driver and no virtual camera, so the
        phone and the PC only have to be able to reach each other on the
        local network. A URL that opens in the PC's browser will open here.
        """
        url = self.config.source_url
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log.error(
                "Could not open camera URL %s. Open that URL in the PC's "
                "browser first: if the browser cannot play the video, the "
                "problem is the network (firewall, guest-Wi-Fi client "
                "isolation, or phone and PC on different networks), not this "
                "code.",
                url,
            )
            return False
        # Keep only the newest frame. A network stream buffers, and a backlog
        # of a second or two turns into lag: the release detector watches
        # pull *velocity*, so a late frame reads as a slower hand.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # not every backend exposes this
            pass
        self.cap = cap
        self.opened = OpenedCamera(index=-1, name=url, backend="FFMPEG")
        self._log_mode(cap)
        log.info("Camera opened: url=%r", url)
        return True

    @staticmethod
    def _log_mode(cap, requested: tuple[int, int] | None = None) -> None:
        """Log the mode the source actually delivers, not the one requested.

        A driver (or a phone app) may silently ignore the request and hand
        back a smaller frame. The lab then upscales it to fill the window and
        it is easy to blame the sensor for a resolution nobody is capturing.
        """
        import cv2

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        log.info("Camera mode: %dx%d @ %.0f fps", width, height, fps)
        if requested and width and height and (width, height) != requested:
            log.warning(
                "Camera negotiated %dx%d, not the requested %dx%d - the driver "
                "or app is resizing. Pick the resolution in the source itself.",
                width,
                height,
                *requested,
            )

    def read(self):
        """Return (ok, bgr_frame)."""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
