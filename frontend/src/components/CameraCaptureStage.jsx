import { useCallback, useEffect, useRef, useState } from "react";
import {
  Camera,
  Eye,
  FlipHorizontal2,
  Grid,
  RefreshCw,
  SlidersHorizontal,
  Sun,
  Upload,
  Video,
  Wifi,
  WifiOff,
} from "lucide-react";
import { drawOverlay, evaluateCaptureQuality } from "../utils/drawing";

const RESOLUTIONS = {
  fast: { label: "640×480 (Fast)", width: 640, height: 480 },
  crisp: { label: "1280×720 (HD)", width: 1280, height: 720 },
};

function getWebSocketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/recognize`;
}

function flipZone(zone) {
  if (zone === "left") return "right";
  if (zone === "right") return "left";
  return zone;
}

function mirrorPredictionForDisplay(payload) {
  return {
    ...payload,
    spatial_zone: flipZone(payload.spatial_zone),
    hands: (payload.hands || []).map((hand) => ({
      ...hand,
      spatial_zone: flipZone(hand.spatial_zone),
      wrist_xy: hand.wrist_xy
        ? [1 - hand.wrist_xy[0], hand.wrist_xy[1]]
        : hand.wrist_xy,
      landmarks: (hand.landmarks || []).map((lm) => ({
        ...lm,
        x: 1 - lm.x,
      })),
    })),
  };
}

export default function CameraCaptureStage({
  activeLanguage,
  activeDialect,
  onRecognizedEvent,
  activeLoci,
}) {
  const videoRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const sampleCanvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const inFlightRef = useRef(false);
  const lastFrameSentAtRef = useRef(0);
  const latestHandsRef = useRef([]);
  const fpsCounterRef = useRef({ frames: 0, lastTick: performance.now() });
  const onRecognizedEventRef = useRef(onRecognizedEvent);

  useEffect(() => {
    onRecognizedEventRef.current = onRecognizedEvent;
  }, [onRecognizedEvent]);

  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [resolutionKey, setResolutionKey] = useState("crisp");
  const [mirrored, setMirrored] = useState(true);
  const mirroredRef = useRef(mirrored);
  useEffect(() => {
    mirroredRef.current = mirrored;
  }, [mirrored]);
  const [showGuides, setShowGuides] = useState(true);
  const [showSpatialLanes, setShowSpatialLanes] = useState(true);
  const [brightness, setBrightness] = useState(105);
  const [contrast, setContrast] = useState(105);
  const [showTuning, setShowTuning] = useState(false);

  const [cameraStatus, setCameraStatus] = useState("starting");
  const [cameraError, setCameraError] = useState("");
  const [videoDims, setVideoDims] = useState({ width: 1280, height: 720 });
  const [wsStatus, setWsStatus] = useState("connecting");
  const [fps, setFps] = useState(0);
  const [latencyMs, setLatencyMs] = useState(null);
  const [avgLuminance, setAvgLuminance] = useState(null);
  const [imageAnalysis, setImageAnalysis] = useState(null);
  const [imageBusy, setImageBusy] = useState(false);
  const [imageError, setImageError] = useState("");
  const [prediction, setPrediction] = useState({
    hands: [],
    candidate_label: null,
    stability_progress: 0,
    locked_label: null,
    phrase: null,
    spatial_zone: "center",
  });

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = all.filter((d) => d.kind === "videoinput");
      setDevices(videoInputs);
      if (!selectedDeviceId && videoInputs.length > 0) {
        setSelectedDeviceId(videoInputs[0].deviceId);
      }
    } catch {
      // Ignore enumeration errors prior to camera permission
    }
  }, [selectedDeviceId]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
  }, []);

  const startCamera = useCallback(async () => {
    stopCamera();
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraStatus("error");
      setCameraError(
        "Camera access requires a secure context (localhost or HTTPS)."
      );
      return;
    }

    setCameraStatus("starting");
    setCameraError("");
    const preset = RESOLUTIONS[resolutionKey] || RESOLUTIONS.crisp;

    try {
      const constraints = {
        audio: false,
        video: {
          width: { ideal: preset.width },
          height: { ideal: preset.height },
          frameRate: { ideal: 30 },
          ...(selectedDeviceId
            ? { deviceId: { exact: selectedDeviceId } }
            : { facingMode: "user" }),
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraStatus("ready");
      await refreshDevices();
    } catch (err) {
      setCameraStatus("error");
      setCameraError(
        err?.message || "Could not open camera. Check browser permissions."
      );
    }
  }, [selectedDeviceId, resolutionKey, stopCamera, refreshDevices]);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, [startCamera, stopCamera]);

  useEffect(() => {
    return () => {
      if (imageAnalysis?.url) {
        URL.revokeObjectURL(imageAnalysis.url);
      }
    };
  }, [imageAnalysis]);

  const clearImageAnalysis = useCallback(() => {
    setImageAnalysis(null);
    setImageError("");
    latestHandsRef.current = [];
  }, []);

  const handleImageUpload = useCallback(async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setImageError("");
    setImageBusy(true);
    const objectUrl = URL.createObjectURL(file);

    try {
      const imgDims = await new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () =>
          resolve({
            width: img.naturalWidth || 640,
            height: img.naturalHeight || 480,
          });
        img.onerror = () => reject(new Error("Could not read selected image."));
        img.src = objectUrl;
      });

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/recognize-image", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload?.detail || "Image recognition failed.");
      }

      latestHandsRef.current = payload.hands || [];
      setPrediction(payload);
      setImageAnalysis({
        url: objectUrl,
        name: file.name,
        width: imgDims.width,
        height: imgDims.height,
        hands: payload.hands || [],
      });

      if (payload.locked_label && onRecognizedEventRef.current) {
        onRecognizedEventRef.current(payload);
      }
    } catch (err) {
      URL.revokeObjectURL(objectUrl);
      setImageError(err?.message || "Could not analyze uploaded image.");
    } finally {
      setImageBusy(false);
    }
  }, []);

  // Connect WebSocket to Python MediaPipe backend with clean auto-reconnect.
  useEffect(() => {
    let closedByEffect = false;
    let reconnectTimer = null;

    function connect() {
      if (closedByEffect) return;
      setWsStatus("connecting");
      const ws = new WebSocket(getWebSocketUrl());
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        if (closedByEffect) return;
        setWsStatus("connected");
        inFlightRef.current = false;
      };

      ws.onmessage = (event) => {
        inFlightRef.current = false;
        const roundTrip = Math.round(
          performance.now() - lastFrameSentAtRef.current
        );
        setLatencyMs(roundTrip);

        try {
          const rawPayload = JSON.parse(event.data);
          // MediaPipe expects unmirrored frames for accurate Left/Right handedness;
          // mirror overlay coordinates and L/R lanes only for display when mirrored.
          const payload = mirroredRef.current
            ? mirrorPredictionForDisplay(rawPayload)
            : rawPayload;
          latestHandsRef.current = payload.hands || [];
          setPrediction(payload);

          fpsCounterRef.current.frames += 1;
          const now = performance.now();
          if (now - fpsCounterRef.current.lastTick >= 1000) {
            setFps(fpsCounterRef.current.frames);
            fpsCounterRef.current.frames = 0;
            fpsCounterRef.current.lastTick = now;
          }

          if (payload.locked_label && onRecognizedEventRef.current) {
            onRecognizedEventRef.current(payload);
          }
        } catch {
          // Ignore malformed message
        }
      };

      ws.onerror = () => {
        inFlightRef.current = false;
      };

      ws.onclose = () => {
        inFlightRef.current = false;
        if (closedByEffect) return;
        setWsStatus("disconnected");
        reconnectTimer = setTimeout(connect, 1800);
      };
    }

    connect();

    return () => {
      closedByEffect = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  // Render static image landmarks cleanly while in IMAGE analysis mode.
  useEffect(() => {
    if (!imageAnalysis) return;
    const overlay = overlayCanvasRef.current;
    if (!overlay) return;
    overlay.width = imageAnalysis.width;
    overlay.height = imageAnalysis.height;
    const ctx = overlay.getContext("2d");
    drawOverlay(ctx, imageAnalysis.width, imageAnalysis.height, {
      hands: imageAnalysis.hands,
      showGuides,
      showSpatialLanes,
      activeLoci,
    });
  }, [imageAnalysis, showGuides, showSpatialLanes, activeLoci]);

  // Overlay drawing + backpressure-safe frame streaming loop for live camera VIDEO mode.
  useEffect(() => {
    if (imageAnalysis) return undefined;

    let rafId = 0;
    let lumTick = 0;

    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement("canvas");
    }
    if (!sampleCanvasRef.current) {
      sampleCanvasRef.current = document.createElement("canvas");
      sampleCanvasRef.current.width = 32;
      sampleCanvasRef.current.height = 24;
    }

    const renderLoop = () => {
      const video = videoRef.current;
      const overlay = overlayCanvasRef.current;

      if (video && overlay && video.readyState >= 2) {
        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;
        if (overlay.width !== vw || overlay.height !== vh) {
          overlay.width = vw;
          overlay.height = vh;
          setVideoDims({ width: vw, height: vh });
        }

        const overlayCtx = overlay.getContext("2d");
        drawOverlay(overlayCtx, vw, vh, {
          hands: latestHandsRef.current,
          showGuides,
          showSpatialLanes,
          activeLoci,
        });

        // Estimate frame luminance periodically (~every 15 animation frames)
        lumTick += 1;
        if (lumTick % 15 === 0) {
          const sCanvas = sampleCanvasRef.current;
          const sCtx = sCanvas.getContext("2d", { willReadFrequently: true });
          sCtx.filter = `brightness(${brightness}%) contrast(${contrast}%)`;
          sCtx.drawImage(video, 0, 0, sCanvas.width, sCanvas.height);
          const { data } = sCtx.getImageData(
            0,
            0,
            sCanvas.width,
            sCanvas.height
          );
          let totalLum = 0;
          for (let i = 0; i < data.length; i += 4) {
            totalLum +=
              0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
          }
          setAvgLuminance(Math.round(totalLum / (data.length / 4)));
        }

        // Stream frame to Python backend only when previous frame finished
        const ws = wsRef.current;
        const now = performance.now();
        const timedOut = now - lastFrameSentAtRef.current > 1200;

        if (
          ws &&
          ws.readyState === WebSocket.OPEN &&
          (!inFlightRef.current || timedOut)
        ) {
          inFlightRef.current = true;
          lastFrameSentAtRef.current = now;

          const sendW = Math.min(640, vw);
          const sendH = Math.round((sendW / vw) * vh);
          const cap = captureCanvasRef.current;
          if (cap.width !== sendW || cap.height !== sendH) {
            cap.width = sendW;
            cap.height = sendH;
          }

          const capCtx = cap.getContext("2d");
          capCtx.save();
          capCtx.filter = `brightness(${brightness}%) contrast(${contrast}%)`;
          // Always send the unmirrored camera frame so MediaPipe reports true Left/Right handedness.
          capCtx.drawImage(video, 0, 0, sendW, sendH);
          capCtx.restore();

          cap.toBlob(
            (blob) => {
              if (
                blob &&
                wsRef.current &&
                wsRef.current.readyState === WebSocket.OPEN
              ) {
                wsRef.current.send(blob);
              } else {
                inFlightRef.current = false;
              }
            },
            "image/jpeg",
            0.78
          );
        }
      }

      rafId = window.requestAnimationFrame(renderLoop);
    };

    rafId = window.requestAnimationFrame(renderLoop);
    return () => window.cancelAnimationFrame(rafId);
  }, [
    brightness,
    contrast,
    showGuides,
    showSpatialLanes,
    activeLoci,
    imageAnalysis,
  ]);

  const quality = imageAnalysis
    ? {
        status: imageAnalysis.hands.length > 0 ? "ok" : "idle",
        message:
          imageAnalysis.hands.length > 0
            ? `Image mode (${imageAnalysis.name}): detected ${imageAnalysis.hands.length} hand(s)`
            : `Image mode (${imageAnalysis.name}): no hand detected in uploaded photo`,
      }
    : evaluateCaptureQuality(prediction.hands, avgLuminance);

  const translatedCandidate =
    prediction.candidate_label &&
    prediction.phrases?.[activeLanguage]
      ? prediction.phrases[activeLanguage]
      : prediction.candidate_label;

  const activeDims = imageAnalysis
    ? { width: imageAnalysis.width, height: imageAnalysis.height }
    : videoDims;

  return (
    <section className="capture-card">
      <div className="capture-toolbar">
        <div className="toolbar-group">
          <label className="select-pill" title="Select Camera">
            <Camera size={15} />
            <select
              value={selectedDeviceId}
              onChange={(e) => setSelectedDeviceId(e.target.value)}
            >
              {devices.length === 0 ? (
                <option value="">Default Camera</option>
              ) : (
                devices.map((dev, idx) => (
                  <option key={dev.deviceId} value={dev.deviceId}>
                    {dev.label || `Camera ${idx + 1}`}
                  </option>
                ))
              )}
            </select>
          </label>

          <label className="select-pill" title="Capture Resolution">
            <Eye size={15} />
            <select
              value={resolutionKey}
              onChange={(e) => setResolutionKey(e.target.value)}
            >
              {Object.entries(RESOLUTIONS).map(([key, item]) => (
                <option key={key} value={key}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="toolbar-group">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={handleImageUpload}
          />

          <button
            type="button"
            className={`icon-btn ${imageAnalysis ? "active" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            disabled={imageBusy}
            title="Test static sign image using MediaPipe IMAGE mode"
          >
            <Upload size={15} />
            <span>{imageBusy ? "Analyzing..." : "Analyze Image"}</span>
          </button>

          {imageAnalysis && (
            <button
              type="button"
              className="icon-btn active"
              onClick={clearImageAnalysis}
              title="Return to live camera stream"
            >
              <Video size={15} />
              <span>Live Camera</span>
            </button>
          )}

          <button
            type="button"
            className={`icon-btn ${mirrored ? "active" : ""}`}
            onClick={() => setMirrored((m) => !m)}
            title="Mirror selfie view"
          >
            <FlipHorizontal2 size={15} />
            <span>Mirror</span>
          </button>

          <button
            type="button"
            className={`icon-btn ${showGuides ? "active" : ""}`}
            onClick={() => setShowGuides((g) => !g)}
            title="Toggle optimal framing box"
          >
            <Grid size={15} />
            <span>Guide</span>
          </button>

          <button
            type="button"
            className={`icon-btn ${showSpatialLanes ? "active" : ""}`}
            onClick={() => setShowSpatialLanes((s) => !s)}
            title="Toggle L/C/R spatial grammar lanes"
          >
            <span>Spatial Lanes</span>
          </button>

          <button
            type="button"
            className={`icon-btn ${showTuning ? "active" : ""}`}
            onClick={() => setShowTuning((t) => !t)}
            title="Adjust brightness & contrast for low light"
          >
            <SlidersHorizontal size={15} />
            <span>Clarity</span>
          </button>
        </div>
      </div>

      {imageError && <div className="inline-warn">{imageError}</div>}

      {showTuning && (
        <div className="tuning-bar">
          <label className="slider-control">
            <Sun size={14} />
            <span>Brightness ({brightness}%)</span>
            <input
              type="range"
              min="70"
              max="170"
              value={brightness}
              onChange={(e) => setBrightness(Number(e.target.value))}
            />
          </label>
          <label className="slider-control">
            <span>Contrast ({contrast}%)</span>
            <input
              type="range"
              min="80"
              max="160"
              value={contrast}
              onChange={(e) => setContrast(Number(e.target.value))}
            />
          </label>
          <button
            type="button"
            className="text-btn"
            onClick={() => {
              setBrightness(105);
              setContrast(105);
            }}
          >
            Reset
          </button>
        </div>
      )}

      <div className="stage-shell">
        <div
          className="video-aspect-frame"
          style={{
            aspectRatio: `${activeDims.width} / ${activeDims.height}`,
            maxWidth: `calc(66vh * ${(activeDims.width / activeDims.height).toFixed(4)})`,
            marginInline: "auto",
          }}
        >
          <video
            ref={videoRef}
            playsInline
            muted
            className="live-video"
            style={{
              display: imageAnalysis ? "none" : "block",
              transform: mirrored ? "scaleX(-1)" : "none",
              filter: `brightness(${brightness}%) contrast(${contrast}%)`,
            }}
          />
          {imageAnalysis && (
            <img
              src={imageAnalysis.url}
              alt={imageAnalysis.name}
              className="live-video uploaded-preview"
            />
          )}
          <canvas ref={overlayCanvasRef} className="overlay-canvas" />

          {!imageAnalysis && cameraStatus !== "ready" && (
            <div className="camera-fallback">
              <Camera size={36} />
              <p>
                {cameraStatus === "starting"
                  ? "Starting camera..."
                  : cameraError}
              </p>
              {cameraStatus === "error" && (
                <button
                  type="button"
                  className="primary-btn"
                  onClick={startCamera}
                >
                  <RefreshCw size={15} /> Retry Camera
                </button>
              )}
            </div>
          )}

          <div className="stage-top-badges">
            {imageAnalysis ? (
              <span className="status-pill ws-connected">
                <Upload size={13} />
                MediaPipe IMAGE Mode · {imageAnalysis.name}
              </span>
            ) : (
              <span className={`status-pill ws-${wsStatus}`}>
                {wsStatus === "connected" ? (
                  <Wifi size={13} />
                ) : (
                  <WifiOff size={13} />
                )}
                {wsStatus === "connected"
                  ? `Python MediaPipe · ${fps} FPS${
                      latencyMs !== null ? ` · ${latencyMs}ms` : ""
                    }`
                  : wsStatus === "connecting"
                  ? "Connecting to Python MediaPipe..."
                  : "Backend offline — run start_backend.bat"}
              </span>
            )}

            <span className="status-pill neutral">
              Dialect: {activeDialect}
            </span>
          </div>
        </div>
      </div>

      <div className="capture-footer">
        <div className={`clarity-banner clarity-${quality.status}`}>
          <span className="clarity-dot" />
          <span>{quality.message}</span>
          {!imageAnalysis && avgLuminance !== null && (
            <span className="lum-tag">Light: {avgLuminance}/255</span>
          )}
        </div>

        <div className="stability-strip">
          <div className="stability-label">
            <span>
              {translatedCandidate
                ? `Holding: ${translatedCandidate}`
                : "Waiting for clear sign..."}
            </span>
            <span>{Math.round((prediction.stability_progress || 0) * 100)}%</span>
          </div>
          <div className="stability-track">
            <div
              className="stability-fill"
              style={{
                width: `${Math.round((prediction.stability_progress || 0) * 100)}%`,
              }}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
