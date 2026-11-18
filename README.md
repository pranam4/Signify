# SignBridge — Real-Time Sign Language Prototype

## Project Structure

- `backend/` — Python MediaPipe `GestureRecognizer` + FastAPI WebSocket & HTTP server
  - `signbridge/recognizer.py` — MediaPipe `GestureRecognizer` wrapper supporting `IMAGE` (`recognize`), `VIDEO` (`recognize_for_video`), and `LIVE_STREAM` (`recognize_async`) modes, `canned_gestures_classifier_options` / `custom_gestures_classifier_options`, `hand_landmarks`, `hand_world_landmarks`, `handedness`, top-K `gestures`, and spatial zone
  - `signbridge/config.py` — Runtime options (including optional `SIGNBRIDGE_MODEL_PATH` custom `.task` model override) and stock/custom label translation helpers
  - `signbridge/stabilizer.py` — Temporal majority-vote stabilizer and cooldown filter
  - `signbridge/model_manager.py` — Downloads & caches `gesture_recognizer.task` in `backend/models/`
  - `signbridge/server.py` — FastAPI `/api/health`, `/api/signs`, `/api/recognize-image` (`IMAGE` mode), and `/ws/recognize` (`VIDEO` mode) endpoints
  - `signbridge/webcam_demo.py` — Standalone OpenCV `LIVE_STREAM` MediaPipe webcam tester
- `frontend/` — React + Vite UI designed for clear camera gesture capture
  - Camera selector, HD/Fast resolution toggle, mirror toggle, brightness & contrast clarity sliders, plus static image upload (`IMAGE` mode)
  - Framing guide box + Left/Center/Right spatial grammar lanes
  - Pixel-aligned 21-landmark hand skeleton overlay + lighting & edge-clipping diagnostics
  - `Sign → Text & Speech` (live Sign → Text transcript builder with Undo/Copy/Speak + optional gesture shortcuts, and Web Speech `SpeechSynthesis` in English/Hindi/Kannada) & `Speech → Sign` (Web Speech `SpeechRecognition` + phrase chips + video/cue display)

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
