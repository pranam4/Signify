import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Camera,
  CheckCircle2,
  Cpu,
  FolderUp,
  Images,
  Plus,
  RefreshCw,
  Trash2,
  Upload,
} from "lucide-react";
import PosterReferencesBrowser from "./PosterReferencesBrowser";

const SUGGESTED_LABELS = [
  "Bathroom",
  "Book",
  "Change",
  "Correct",
  "Dad",
  "Drink",
  "Eat",
  "Finish",
  "Goodbye",
  "Hello",
  "Help",
  "Home",
  "Hug",
  "Meet",
  "Milk",
  "Mom",
  "More",
  "Nice",
  "No",
  "Pain",
  "Please",
  "Sorry",
  "Stop",
  "Thank You",
  "Touch",
  "Want",
  "Yes",
  "You",
];

const SOURCE_FILTERS = [
  { id: "all", label: "All Sources" },
  { id: "voice_to_sign_avatar", label: "Voice → Sign 3D Poses" },
  { id: "poster_photo", label: "First Signs (Real Photos)" },
  { id: "poster_illustrated", label: "Essentials Poster (Hands)" },
  { id: "image2_labeled", label: "Image 2 (15 Words)" },
  { id: "image1_inferred", label: "Image 1 (Inferred · Review)" },
  { id: "webcam", label: "Webcam Captured" },
  { id: "imported", label: "Imported Folder" },
];

export default function SignDatasetPanel({ onModelTrained }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);

  const [overview, setOverview] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedLabel, setSelectedLabel] = useState("Help");
  const [customLabelInput, setCustomLabelInput] = useState("");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [cameraActive, setCameraActive] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [captureProgress, setCaptureProgress] = useState("");
  const [recordingSeq, setRecordingSeq] = useState(false);
  const [training, setTraining] = useState(false);
  const [syncingVoiceSigns, setSyncingVoiceSigns] = useState(false);
  const [folderPath, setFolderPath] = useState("");
  const [importing, setImporting] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [editingSampleId, setEditingSampleId] = useState(null);
  const [editingLabelText, setEditingLabelText] = useState("");
  const [datasetTab, setDatasetTab] = useState("trainer");

  const loadOverview = useCallback(async () => {
    try {
      const res = await fetch("/api/dataset");
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Failed to load dataset.");
      setOverview(data);
    } catch (err) {
      setErrorMsg(err?.message || "Could not reach dataset API.");
    }
  }, []);

  const loadSamples = useCallback(async (label, src) => {
    try {
      const params = new URLSearchParams();
      if (label) params.set("label", label);
      if (src && src !== "all") params.set("source", src);
      const res = await fetch(`/api/dataset/samples?${params.toString()}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Failed to load samples.");
      setSamples(data.samples || []);
    } catch (err) {
      setErrorMsg(err?.message || "Could not load samples.");
    }
  }, []);

  useEffect(() => {
    void loadOverview();
  }, [loadOverview]);

  useEffect(() => {
    void loadSamples(selectedLabel, sourceFilter);
  }, [loadSamples, selectedLabel, sourceFilter]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
    setCameraActive(false);
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const startCamera = async () => {
    setErrorMsg("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMsg("Camera capture requires localhost or HTTPS.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
    } catch (err) {
      setErrorMsg(err?.message || "Could not open webcam.");
    }
  };

  const activeTargetLabel = (customLabelInput.trim() || selectedLabel || "Help").trim();

  const uploadFrameBlob = async (blob, label, source = "webcam") => {
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");
    formData.append("label", label);
    formData.append("source", source);
    const res = await fetch("/api/dataset/samples", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data?.detail || "Sample extraction failed.");
    }
    return data.sample;
  };

  const captureCanvasBlob = () =>
    new Promise((resolve, reject) => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) {
        reject(new Error("Webcam stream is not ready yet."));
        return;
      }
      if (!canvasRef.current) {
        canvasRef.current = document.createElement("canvas");
      }
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error("Frame encoding failed."))),
        "image/jpeg",
        0.9
      );
    });

  const handleCaptureFrames = async (count) => {
    setErrorMsg("");
    setStatusMsg("");
    setCapturing(true);
    let saved = 0;
    let missed = 0;

    try {
      for (let i = 0; i < count; i += 1) {
        setCaptureProgress(`Capturing ${i + 1}/${count} for "${activeTargetLabel}"...`);
        const blob = await captureCanvasBlob();
        try {
          await uploadFrameBlob(blob, activeTargetLabel, "webcam");
          saved += 1;
        } catch {
          missed += 1;
        }
        if (i < count - 1) {
          await new Promise((r) => setTimeout(r, 160));
        }
      }
      setSelectedLabel(activeTargetLabel);
      setCustomLabelInput("");
      setStatusMsg(
        `Saved ${saved} webcam sample(s) for "${activeTargetLabel}"${
          missed > 0 ? ` (${missed} skipped without hands)` : ""
        }. Click Train Model to update live recognition.`
      );
      await loadOverview();
      await loadSamples(activeTargetLabel, sourceFilter);
    } catch (err) {
      setErrorMsg(err?.message || "Failed to capture webcam frames.");
    } finally {
      setCapturing(false);
      setCaptureProgress("");
    }
  };

  const handleRecordSequence = async () => {
    setErrorMsg("");
    setStatusMsg("");
    setRecordingSeq(true);
    const frameBlobs = [];
    const timestamps = [];
    const startMs = performance.now();

    try {
      for (let i = 0; i < 10; i += 1) {
        setCaptureProgress(`Recording motion ${i + 1}/10 for "${activeTargetLabel}"...`);
        const blob = await captureCanvasBlob();
        frameBlobs.push(blob);
        timestamps.push(Math.round(performance.now() - startMs));
        if (i < 9) {
          await new Promise((r) => setTimeout(r, 125));
        }
      }
      const formData = new FormData();
      frameBlobs.forEach((b, idx) => formData.append("files", b, `seq_${idx}.jpg`));
      formData.append("label", activeTargetLabel);
      formData.append("timestamps_ms", timestamps.join(","));

      const res = await fetch("/api/dataset/sequences", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Sequence recording failed.");

      setSelectedLabel(activeTargetLabel);
      setCustomLabelInput("");
      setStatusMsg(
        `Recorded moving sign clip (${data.sequence?.num_frames || 0} frames, ${
          data.sequence?.duration_ms || 0
        }ms) for "${activeTargetLabel}". Click Train recognition model to update live motion tracking!`
      );
      await loadOverview();
    } catch (err) {
      setErrorMsg(err?.message || "Failed to record moving sign clip.");
    } finally {
      setRecordingSeq(false);
      setCaptureProgress("");
    }
  };

  const handleUploadSampleImage = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setErrorMsg("");
    setStatusMsg("");
    try {
      await uploadFrameBlob(file, activeTargetLabel, "imported");
      setSelectedLabel(activeTargetLabel);
      setStatusMsg(`Added "${file.name}" to class "${activeTargetLabel}".`);
      await loadOverview();
      await loadSamples(activeTargetLabel, sourceFilter);
    } catch (err) {
      setErrorMsg(err?.message || "Could not extract landmarks from uploaded image.");
    }
  };

  const handleImportFolder = async (e) => {
    e.preventDefault();
    if (!folderPath.trim()) return;
    setErrorMsg("");
    setStatusMsg("");
    setImporting(true);
    try {
      const res = await fetch("/api/dataset/import-folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_path: folderPath.trim(), max_per_class: 40 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Folder import failed.");
      setStatusMsg(
        `Imported ${data.imported} sample(s) across ${
          Object.keys(data.classes || {}).length
        } class(es) (${data.skipped} skipped).`
      );
      await loadOverview();
      await loadSamples(selectedLabel, sourceFilter);
    } catch (err) {
      setErrorMsg(err?.message || "Folder import failed.");
    } finally {
      setImporting(false);
    }
  };

  const handleTrainModel = async () => {
    setErrorMsg("");
    setStatusMsg("");
    setTraining(true);
    try {
      const res = await fetch("/api/dataset/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ epochs: 45 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Training failed.");
      setStatusMsg(
        `Trained PyTorch classifier across ${data.num_classes} sign(s) (${data.total_original_samples} samples, ${(
          data.train_accuracy * 100
        ).toFixed(0)}% train accuracy). Live camera now recognizes all trained signs!`
      );
      await loadOverview();
      if (onModelTrained) onModelTrained(data);
    } catch (err) {
      setErrorMsg(err?.message || "Model training failed.");
    } finally {
      setTraining(false);
    }
  };

  const handleSyncVoiceSigns = async () => {
    setErrorMsg("");
    setStatusMsg("");
    setSyncingVoiceSigns(true);
    try {
      const res = await fetch("/api/dataset/sync-voice-signs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ include_existing: false, epochs: 55 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Sync failed.");
      setStatusMsg(
        `Synced ${data.added_classes?.length || 0} Voice → Sign class(es) (${
          data.added_samples || 0
        } samples). Classifier now supports ${
          data.classifier?.num_classes || data.summary?.total_classes || 0
        } signs!`
      );
      await loadOverview();
      await loadSamples(selectedLabel, sourceFilter);
      if (onModelTrained && data.train) onModelTrained(data.train);
    } catch (err) {
      setErrorMsg(err?.message || "Could not sync Voice → Sign classes.");
    } finally {
      setSyncingVoiceSigns(false);
    }
  };

  const handleRelabelSample = async (sampleId, newLabel, markVerified = true) => {
    setErrorMsg("");
    try {
      const res = await fetch(`/api/dataset/samples/${sampleId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...(newLabel ? { label: newLabel } : {}),
          ...(markVerified ? { needs_review: false } : {}),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Update failed.");
      setEditingSampleId(null);
      setEditingLabelText("");
      await loadOverview();
      await loadSamples(selectedLabel, sourceFilter);
    } catch (err) {
      setErrorMsg(err?.message || "Could not update sample.");
    }
  };

  const handleDeleteSample = async (sampleId) => {
    setErrorMsg("");
    try {
      const res = await fetch(`/api/dataset/samples/${sampleId}`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error("Failed to delete sample.");
      await loadOverview();
      await loadSamples(selectedLabel, sourceFilter);
    } catch (err) {
      setErrorMsg(err?.message || "Could not delete sample.");
    }
  };

  const filteredClasses = useMemo(() => {
    const all = overview?.classes || [];
    if (sourceFilter === "all") return all;
    return all.filter((c) => (c.sources?.[sourceFilter] || 0) > 0);
  }, [overview, sourceFilter]);

  const labelOptions = useMemo(() => {
    const set = new Set(SUGGESTED_LABELS);
    for (const cls of overview?.classes || []) set.add(cls.label);
    return Array.from(set);
  }, [overview]);

  const classifier = overview?.classifier;

  return (
    <section className="dataset-card">
      <div className="dataset-header">
        <div>
          <h3>Your sign collection</h3>
          <p>
            Capture a sign, add your examples, and help your personal library grow.
          </p>
        </div>
        <div className="toolbar-group">
          <button
            type="button"
            className="icon-btn"
            onClick={handleSyncVoiceSigns}
            disabled={syncingVoiceSigns || training}
            title="Ensure all Voice -> Sign 3D avatar signs are in the Sign -> Text/Voice classifier"
          >
            <RefreshCw size={15} />
            <span>{syncingVoiceSigns ? "Syncing..." : "Sync Voice → Sign"}</span>
          </button>
          <button
            type="button"
            className="mic-btn"
            onClick={handleTrainModel}
            disabled={training || !overview?.total_samples}
          >
            <Cpu size={16} />
            <span>{training ? "Training model…" : "Train recognition model"}</span>
          </button>
        </div>
      </div>

      <div className="dataset-stats-bar">
        <span className="status-pill neutral">
          Classes: {overview?.total_classes ?? 0}
        </span>
        <span className="status-pill neutral">
          Samples: {overview?.total_samples ?? 0}
        </span>
        <span className="status-pill neutral">
          Needs Review (Img 1): {overview?.needs_review_total ?? 0}
        </span>
        {overview?.database?.islrtc_entries > 0 && (
          <span className="status-pill neutral">
            ISLRTC DB: {overview.database.islrtc_entries.toLocaleString()} signs loaded
          </span>
        )}
        <span className={`status-pill ${classifier?.loaded ? "ws-connected" : "ws-disconnected"}`}>
          {classifier?.loaded
            ? `Static Model · ${classifier.num_classes} signs · ${Math.round(
                (classifier.train_accuracy || 0) * 100
              )}% acc`
            : "Static Model Not Trained"}
        </span>
        <span
          className={`status-pill ${
            overview?.dynamic_classifier?.loaded ? "ws-connected" : "ws-disconnected"
          }`}
        >
          {overview?.dynamic_classifier?.loaded
            ? `Motion Model · ${overview.dynamic_classifier.num_classes} moving signs · ${Math.round(
                (overview.dynamic_classifier.train_accuracy || 0) * 100
              )}% acc`
            : "Motion Model Not Trained"}
        </span>
      </div>

      {statusMsg && <div className="inline-ok">{statusMsg}</div>}
      {errorMsg && <div className="inline-warn">{errorMsg}</div>}

      <div className="dataset-subtabs">
        <button
          type="button"
          className={`icon-btn ${datasetTab === "trainer" ? "active" : ""}`}
          onClick={() => setDatasetTab("trainer")}
        >
          <Cpu size={14} />
          <span>My collection ({overview?.total_classes ?? 0} signs)</span>
        </button>
        <button
          type="button"
          className={`icon-btn ${datasetTab === "posters" ? "active" : ""}`}
          onClick={() => setDatasetTab("posters")}
        >
          <Images size={14} />
          <span>Reference posters</span>
        </button>
      </div>

      {datasetTab === "posters" && (
        <PosterReferencesBrowser
          onSelectForTraining={(lbl) => {
            if (labelOptions.includes(lbl)) {
              setSelectedLabel(lbl);
              setCustomLabelInput("");
            } else {
              setCustomLabelInput(lbl);
            }
            setDatasetTab("trainer");
          }}
        />
      )}

      {datasetTab === "trainer" && (
      <div className="dataset-studio-grid">
        <div className="dataset-recorder-box">
          <div className="card-header">
            <span>
              <Camera size={14} /> Record Webcam / Import Samples
            </span>
            {!cameraActive ? (
              <button type="button" className="text-btn" onClick={startCamera}>
                Open Webcam
              </button>
            ) : (
              <button type="button" className="text-btn" onClick={stopCamera}>
                Stop Webcam
              </button>
            )}
          </div>

          <div className="dataset-video-box">
            <video
              ref={videoRef}
              playsInline
              muted
              className="dataset-video-preview"
              style={{ display: cameraActive ? "block" : "none" }}
            />
            {!cameraActive && (
              <div className="dataset-video-placeholder">
                <Camera size={28} />
                <span>Open webcam to capture 15–30 real frames per sign</span>
              </div>
            )}
          </div>

          <div className="dataset-form-row">
            <label className="select-pill">
              <span>Sign</span>
              <select
                value={selectedLabel}
                onChange={(e) => setSelectedLabel(e.target.value)}
              >
                {labelOptions.map((lbl) => (
                  <option key={lbl} value={lbl}>
                    {lbl}
                  </option>
                ))}
              </select>
            </label>

            <input
              type="text"
              className="dataset-text-input"
              value={customLabelInput}
              onChange={(e) => setCustomLabelInput(e.target.value)}
              placeholder="Or type new label (e.g. Water)..."
            />
          </div>

          <div className="dataset-btn-row">
            <button
              type="button"
              className="icon-btn active"
              disabled={!cameraActive || capturing || recordingSeq}
              onClick={() => handleCaptureFrames(15)}
            >
              <Plus size={14} />
              <span>{capturing ? captureProgress : "Capture 15-Frame Burst"}</span>
            </button>

            <button
              type="button"
              className="icon-btn active"
              disabled={!cameraActive || capturing || recordingSeq}
              onClick={handleRecordSequence}
              title="Record a ~1.2s continuous moving sign clip for real-time dynamic recognition"
            >
              <RefreshCw size={14} />
              <span>
                {recordingSeq ? captureProgress : "Record Moving Sign (1.2s)"}
              </span>
            </button>

            <button
              type="button"
              className="icon-btn"
              disabled={!cameraActive || capturing || recordingSeq}
              onClick={() => handleCaptureFrames(1)}
            >
              <Camera size={14} />
              <span>Single Frame</span>
            </button>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={handleUploadSampleImage}
            />
            <button
              type="button"
              className="icon-btn"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload size={14} />
              <span>Upload Image</span>
            </button>
          </div>

          <form className="dataset-import-form" onSubmit={handleImportFolder}>
            <input
              type="text"
              className="dataset-text-input"
              value={folderPath}
              onChange={(e) => setFolderPath(e.target.value)}
              placeholder="Local dataset folder (e.g. C:\data\isl_images)..."
            />
            <button
              type="submit"
              className="icon-btn"
              disabled={importing || !folderPath.trim()}
            >
              <FolderUp size={14} />
              <span>{importing ? "Importing..." : "Import Folder"}</span>
            </button>
          </form>
        </div>

        <div className="dataset-browser-box">
          <div className="card-header">
            <span>Classes &amp; Label Review</span>
            <div className="transcript-actions">
              <select
                className="dataset-filter-select"
                value={sourceFilter}
                onChange={(e) => setSourceFilter(e.target.value)}
              >
                {SOURCE_FILTERS.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.label}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="text-btn"
                onClick={() => {
                  void loadOverview();
                  void loadSamples(selectedLabel, sourceFilter);
                }}
              >
                <RefreshCw size={13} /> Refresh
              </button>
            </div>
          </div>

          <div className="dataset-class-chips">
            {filteredClasses.map((cls) => (
              <button
                key={cls.label}
                type="button"
                className={`chip-btn ${
                  selectedLabel === cls.label ? "selected" : ""
                }`}
                onClick={() => setSelectedLabel(cls.label)}
              >
                {cls.label} ({cls.count})
                {cls.needs_review_count > 0 ? " *" : ""}
              </button>
            ))}
          </div>

          <div className="dataset-sample-grid">
            {samples.length === 0 ? (
              <p className="empty-log">
                No samples match &ldquo;{selectedLabel}&rdquo; under this filter.
              </p>
            ) : (
              samples.map((s) => (
                <div key={s.sample_id} className="dataset-sample-card">
                  <img
                    src={s.image_url}
                    alt={`${s.label} sample`}
                    className="dataset-sample-thumb"
                  />
                  <div className="dataset-sample-meta">
                    <strong>{s.label}</strong>
                    <small>
                      {s.source} · {s.num_hands} hand(s)
                      {s.has_pose ? " · pose" : ""}
                    </small>
                    {s.needs_review && (
                      <span className="review-badge">Needs Review</span>
                    )}
                  </div>

                  {editingSampleId === s.sample_id ? (
                    <div className="dataset-relabel-row">
                      <input
                        type="text"
                        className="dataset-text-input"
                        value={editingLabelText}
                        onChange={(e) => setEditingLabelText(e.target.value)}
                        placeholder="New label..."
                      />
                      <button
                        type="button"
                        className="text-btn"
                        onClick={() =>
                          handleRelabelSample(s.sample_id, editingLabelText.trim(), true)
                        }
                      >
                        Save
                      </button>
                    </div>
                  ) : (
                    <div className="dataset-sample-actions">
                      <button
                        type="button"
                        className="text-btn"
                        onClick={() => {
                          setEditingSampleId(s.sample_id);
                          setEditingLabelText(s.label);
                        }}
                      >
                        Relabel
                      </button>
                      {s.needs_review && (
                        <button
                          type="button"
                          className="text-btn"
                          onClick={() => handleRelabelSample(s.sample_id, null, true)}
                        >
                          <CheckCircle2 size={12} /> Verify
                        </button>
                      )}
                      <button
                        type="button"
                        className="text-btn"
                        onClick={() => handleDeleteSample(s.sample_id)}
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
      )}
    </section>
  );
}
