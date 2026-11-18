# SignBridge — Real-Time Sign Language Prototype

## Project Structure

- `backend/` — Python MediaPipe `GestureRecognizer` + `HolisticLandmarker` + FastAPI WebSocket & HTTP server
  - `signbridge/recognizer.py` — Parallel MediaPipe `GestureRecognizer` + `HolisticLandmarker` (553 landmarks: 33 pose, 478 face + 52 blendshapes, 2×21 hands) wrapper supporting `IMAGE`, `VIDEO`, and `LIVE_STREAM` modes
  - `signbridge/spatial_grammar.py` — Holistic Spatial Grammar & Non-Manual Marker (NMM) engine: torso-anchored Left/Center/Right & Head/Chest/Low signing space, brow question mood, head nod/shake/tilt, iris/blendshape eye gaze, body role shift, and directional verb trajectories (`A → B`)
  - `signbridge/config.py` — Runtime options (`SIGNBRIDGE_MODEL_PATH`, `SIGNBRIDGE_HOLISTIC_MODEL_PATH`, `SIGNBRIDGE_ENABLE_HOLISTIC`) and stock/custom/directional phrase mappings
  - `signbridge/stabilizer.py` — Temporal majority-vote stabilizer and cooldown filter
  - `signbridge/model_manager.py` — Downloads & caches `gesture_recognizer.task` and `holistic_landmarker.task` in `backend/models/`
  - `signbridge/server.py` — FastAPI `/api/health`, `/api/signs`, `/api/recognize-image` (`IMAGE` mode), and `/ws/recognize` (`VIDEO` mode) endpoints
  - `signbridge/webcam_demo.py` — Standalone OpenCV `LIVE_STREAM` MediaPipe Holistic + Gesture webcam tester
- `frontend/` — React + Vite UI designed for clear camera gesture capture
  - Camera selector, HD/Fast resolution toggle, Holistic Spatial toggle, mirror toggle, brightness & contrast clarity sliders, plus static image upload (`IMAGE` mode)
  - Framing guide box + body-anchored Left/Center/Right spatial grammar lanes + directional trajectory arcs
  - Pixel-aligned 21-landmark hand skeletons + holistic upper-body pose & facial contour overlay + lighting & edge-clipping diagnostics
  - `Sign → Text & Speech` (live Sign → Text transcript builder with NMM grammar inflection, Undo/Copy/Speak + optional gesture shortcuts, and Web Speech `SpeechSynthesis` in English/Hindi/Kannada) & `Speech → Sign` (Web Speech `SpeechRecognition` + phrase chips + video/cue display)

## Quick Start

- **Recommended (both servers at once):** Double-click `start_all.bat` (keeps **both** the Backend and Frontend windows open; wait ~15–20 seconds for the backend window to print `Uvicorn running on http://127.0.0.1:8000`, then open `http://localhost:5173`).

Or start them manually in **two separate terminal windows** (keep both open):

1. Start the Python MediaPipe backend (`http://127.0.0.1:8000`):
   ```bat
   start_backend.bat
   ```
2. Start the React UI (`http://localhost:5173`):
   ```bat
   start_frontend.bat
   ```
3. Or test Python MediaPipe standalone with OpenCV:
   ```bat
   start_webcam_demo.bat
   ```

## Running Tests

```bat
cd C:\Hackathon\backend
.venv\Scripts\pytest.exe -v
```
