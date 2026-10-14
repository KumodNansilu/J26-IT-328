"""Quick env check — run with: python scripts/check_env.py"""
import importlib.util
import sys

print("python:", sys.version)
for pkg, mod in [("PySide6", "PySide6"), ("opencv", "cv2"),
                 ("numpy", "numpy"), ("pytest", "pytest"),
                 ("mediapipe", "mediapipe"), ("torch", "torch"),
                 ("sqlalchemy", "sqlalchemy")]:
    print(f"{pkg}:", "OK" if importlib.util.find_spec(mod) else "MISSING")
