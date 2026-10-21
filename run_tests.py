import os
import subprocess
import sys

# run_tests.py sits at c:\Users\User\Desktop\J26-IT-328
REPO = os.path.abspath(os.path.dirname(__file__))
# Package root: ai_services (and MindSync-Enterprise/tests) live under
# MindSync-Enterprise/.
PACKAGE_ROOT = os.path.join(REPO, "MindSync-Enterprise")

os.environ["PYTHONPATH"] = PACKAGE_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

r = subprocess.run(
    [sys.executable, "-m", "pytest",
     "MindSync-Enterprise/tests/unit/test_cnn_emotion.py", "-v"],
    cwd=REPO, env=os.environ, capture_output=True, text=True, timeout=180)
print("===== STDOUT =====")
print(r.stdout)
print("===== STDERR =====")
print(r.stderr)
print("===== EXIT CODE =====")
print(r.returncode)
