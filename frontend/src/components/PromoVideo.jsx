import { useEffect, useRef, useState } from "react";
import { Play, X } from "lucide-react";

export default function PromoVideo() {
  const dialogRef = useRef(null);
  const videoRef = useRef(null);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (open) dialogRef.current?.showModal();
    else {
      videoRef.current?.pause();
      dialogRef.current?.close();
    }
  }, [open]);
  return <>
    <button type="button" className="guide-button" onClick={() => setOpen(true)}><Play size={15} /> Watch the film <span>30s</span></button>
    <dialog ref={dialogRef} className="promo-dialog" aria-label="Signify in 30 seconds" onCancel={() => setOpen(false)} onClick={event => { if (event.target === dialogRef.current) setOpen(false); }}>
      <div className="promo-dialog-header"><div><strong>Meet Signify.</strong><p>A bridge between signs and words.</p></div><button type="button" className="icon-btn" aria-label="Close video" onClick={() => setOpen(false)}><X size={20} /></button></div>
      {open && <video ref={videoRef} src="/media/signify-30s.mp4" poster="/media/signify-poster.jpg" controls playsInline preload="metadata" />}
      <p className="promo-caption">An illustrated feature tour · Music only; all feature descriptions appear on screen.</p>
    </dialog>
  </>;
}
