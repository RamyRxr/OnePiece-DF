"""Camera-driven stretch perception for the One Piece Devil Fruit project.

The package is deliberately split so that the risky part (math) is pure:

    indices.py   landmark index constants, no dependencies
    signals.py   pure functions: landmarks -> geometric features
    filters.py   OneEuroFilter + derivative
    channels.py  per-channel filter/activate/release state machine
    grab.py      pinch-grab-pull, the main interaction (pure)
    pipeline.py  features -> calibration -> channels -> message (pure)
    landmarks.py MediaPipe wrappers (the only module that imports mediapipe)
    capture.py   camera acquisition, device selected by NAME
    baseline.py  calibration capture + persistence
    protocol.py  wire format shared with the Unity client
    warp.py      the stretch effect: elastic-skin physics + rendering (cosmetic)
    skin.py      what counts as skin for the stretch effect
    server.py    camera -> pipeline -> websocket + lab overlay
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
