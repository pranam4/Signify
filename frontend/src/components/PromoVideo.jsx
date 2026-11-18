import { useEffect, useRef, useState } from "react";
import { Play, X } from "lucide-react";

export default function PromoVideo() {
  const dialogRef = useRef(null);
  const videoRef = useRef(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (open) {
      dialogRef.current?.showModal();
    } else {
      videoRef.current?.pause();
      dialogRef.current?.close();
    }
  }, [open]);

  return (
    <>
      <button
        type="button"
        className="promo-top-btn"
        onClick={() => setOpen(true)}
        title="View 30-second Signify showcase video"
      >
        <Play size={14} />
        <span>View 30-sec video</span>
      </button>

      <dialog
        ref={dialogRef}
        className="promo-dialog"
        aria-label="Signify 30-second feature video"
        onCancel={() => setOpen(false)}
        onClick={(event) => {
          if (event.target === dialogRef.current) setOpen(false);
        }}
      >
        <div className="promo-dialog-header">
          <div>
            <strong>Meet Signify · 30-Second Feature Tour</strong>
            <p>Two-way sign translation, 553-pt spatial grammar, Whisper STT &amp; 3D Ray.</p>
          </div>
          <button
            type="button"
            className="icon-btn"
            aria-label="Close video"
            onClick={() => setOpen(false)}
          >
            <X size={20} />
          </button>
        </div>
        {open && (
          <video
            ref={videoRef}
            src="/media/signify-30s.mp4"
            poster="/media/signify-poster.jpg"
            autoPlay
            controls
            playsInline
            preload="metadata"
          />
        )}
        <p className="promo-caption">
          30-second HD showcase · Music included; all features and metrics are highlighted on screen.
        </p>
      </dialog>
    </>
  );
}
