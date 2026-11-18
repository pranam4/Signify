# Signify — Real-Time Sign Language Prototype

## Project Structure

- `backend/` — Python MediaPipe `GestureRecognizer` + `HolisticLandmarker` + FastAPI WebSocket & HTTP server
  - `signbridge/recognizer.py` — Parallel MediaPipe `GestureRecognizer` + `HolisticLandmarker` (553 landmarks: 33 pose, 478 face + 52 blendshapes, 2×21 hands) wrapper supporting `IMAGE`, `VIDEO`, and `LIVE_STREAM` modes
  - `signbridge/spatial_grammar.py` — Holistic Spatial Grammar & Non-Manual Marker (NMM) engine: torso-anchored Left/Center/Right & Head/Chest/Low signing space, brow question mood, head nod/shake/tilt, iris/blendshape eye gaze, body role shift, and directional verb trajectories (`A → B`)
  - `signbridge/config.py` — Runtime options (`SIGNBRIDGE_MODEL_PATH`, `SIGNBRIDGE_HOLISTIC_MODEL_PATH`, `SIGNBRIDGE_ENABLE_HOLISTIC`) and stock/custom/directional phrase mappings
  - `signbridge/stabilizer.py` — Temporal majority-vote stabilizer and cooldown filter
  - `signbridge/model_manager.py` — Downloads & caches `gesture_recognizer.task` and `holistic_landmarker.task` in `backend/models/`
  - `signbridge/speech_to_text.py` & `signbridge/stt_router.py` — Local `openai/whisper-tiny` Speech-to-Text (`POST /api/stt/transcribe`, `WS /ws/stt`)
  - `signbridge/sign_planner.py` — Spoken text → Sign gloss reordering, NMM clause mood tagging, and A–Z fingerspelling planner (`POST /api/text-to-sign`)
  - `signbridge/server.py` — FastAPI `/api/health`, `/api/signs`, `/api/recognize-image`, `/api/text-to-sign`, `/api/stt/transcribe`, `/ws/recognize`, and `/ws/stt` endpoints
  - `signbridge/webcam_demo.py` — Standalone OpenCV `LIVE_STREAM` MediaPipe Holistic + Gesture webcam tester
- `frontend/` — React + Vite UI designed for clear camera gesture capture
  - Camera selector, HD/Fast resolution toggle, Holistic Spatial toggle, mirror toggle, brightness & contrast clarity sliders, plus static image upload (`IMAGE` mode)
  - Framing guide box + body-anchored Left/Center/Right spatial grammar lanes + directional trajectory arcs
  - Pixel-aligned 21-landmark hand skeletons + holistic upper-body pose & facial contour overlay + lighting & edge-clipping diagnostics
  - `Sign → Text & Speech` (live Sign → Text transcript builder with NMM grammar inflection, Undo/Copy/Speak + optional gesture shortcuts, and Web Speech `SpeechSynthesis` in English/Hindi/Kannada) & `Voice → Text & Sign` (Whisper AI mic recording, audio upload, live browser mic, gloss planner, and interactive Three.js **3D Signing Avatar** with articulated 15-joint hands, 2-bone arm IK, NMM facial expressions, and A–Z fingerspelling)

## Running on a New Device (After Cloning from GitHub)

### Prerequisites
- **Python 3.10–3.12** (on Windows, check *"Add Python to PATH"* during installation)
- **Node.js 18+** (includes `npm`)
- **Internet connection on first launch** so the backend can auto-download `gesture_recognizer.task` (~8.4 MB), `holistic_landmarker.task` (~13.7 MB), and `openai/whisper-tiny` from Hugging Face on first transcription. After first launch, everything runs locally.

### Windows (One-Click Setup & Launch)
```bat
git clone https://github.com/pranam4/Signify.git
cd Signify
start_all.bat
```
- On the first run, `start_backend.bat` and `start_frontend.bat` automatically create `backend/.venv`, install Python & npm packages, and start both servers.
- Keep **both** terminal windows open and open `http://localhost:5173` in Chrome or Edge.

### macOS / Linux
Terminal 1 (Backend):
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn signbridge.server:app --host 127.0.0.1 --port 8000
```
Terminal 2 (Frontend):
```bash
cd frontend
npm install
npm run dev
```
Then open `http://localhost:5173`.

### Accessing from a Phone / Tablet on the Same Wi-Fi
- Browsers block **camera and microphone** on plain `http://192.168.x.x:5173` (non-HTTPS LAN URLs).
- To use the app from a phone or another device over the network, expose `http://localhost:5173` over an **HTTPS** tunnel (for example, `ngrok http 5173` or `cloudflared tunnel --url http://localhost:5173`) and open the `https://...` URL on the phone.

## Quick Start (Already Set Up)

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
