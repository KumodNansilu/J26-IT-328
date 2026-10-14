"""Camera diagnostic — run with: python scripts/check_camera.py

Tries indexes 0..3 with DSHOW / MSMF backends, reads a few frames,
reports resolution + brightness, and saves one snapshot per working
camera so you can SEE what Windows gives OpenCV.

This tells us whether index 0 is wrong, busy, blocked by privacy
settings, or covered by a physical shutter.
"""
from __future__ import annotations

import os
import tempfile

import cv2


def probe(index: int, backend: int, backend_name: str) -> bool:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"[index {index} {backend_name}] cannot open")
        cap.release()
        return False
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        print(f"[index {index} {backend_name}] opened but read FAILED "
              f"(busy / no permission / no frames)")
        cap.release()
        return False
    h, w = frame.shape[:2]
    mean = float(frame.mean())
    print(f"[index {index} {backend_name}] OK — {w}x{h}, mean brightness {mean:.1f}")
    if mean < 3.0:
        print(f"    -> frame is (almost) BLACK: shutter closed, kill-switch, "
              f"or lens covered?")
    out = os.path.join(tempfile.gettempdir(), f"mindsync_camtest_{index}_{backend_name}.jpg")
    cv2.imwrite(out, frame)
    print(f"    snapshot saved: {out}  (open it to see what the camera returns)")
    cap.release()
    return True


def main() -> None:
    backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]
    found = False
    for backend, name in backends:
        print(f"--- backend {name} ---")
        for i in range(4):
            try:
                if probe(i, backend, name):
                    found = True
            except Exception as e:  # noqa: BLE001 — diagnostic only
                print(f"[index {i} {name}] error: {e}")
    if not found:
        print("\nNo camera could be opened. Check:")
        print(" 1. Windows Settings > Privacy & security > Camera >")
        print("    'Let desktop apps access your camera' = ON")
        print(" 2. Close Teams / Zoom / browser that may hold the camera")
        print(" 3. Physical shutter slider / Fn-key kill switch on the laptop")
        print(" 4. Open the built-in Windows Camera app — if IT shows the same")
        print("    blocked icon, the problem is Windows/hardware, not MindSync.")


if __name__ == "__main__":
    main()
