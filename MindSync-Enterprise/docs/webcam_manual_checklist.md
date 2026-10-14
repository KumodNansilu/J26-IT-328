# VFEBA Video Input — manual webcam checklist (Windows)

Automated tests cover validation (`pytest tests/unit -v`) but NOT the
physical webcam. Work through this list with the app running:

1. Launch: `python -m app.main` from `MindSync-Enterprise/`
2. Upload valid ~30 s MP4 → expect "Passed", Continue enabled.
3. Upload wrong extension (.txt) → expect "Unsupported format".
4. Upload 2 s clip → expect "too short".
5. Start Recording → live preview + countdown from 30.
6. Stop early → file still saved + validated (likely "too short" — correct).
7. Start → Cancel → no temp file left in %TEMP%\mindsync_vfeba.
8. Retake → old temp recording deleted, new one validated.
9. Continue to Analysis → confirm dialog (CNN/VBS = next phase).
10. Close window mid-recording → camera released (OS camera light off),
    thread stopped, temp file removed.
11. No webcam machine → clear "Webcam unavailable" error, no crash.
