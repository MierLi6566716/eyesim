import { useEffect, useRef, useState } from "react";
import styles from "./CameraCapture.module.css";

const READY_TIMEOUT_MS = 7000;

/**
 * The honest version of "crop the pupil region": no face/eye detection
 * model runs here (see proposal/ — Diagnose mode's preprocessing is
 * documented as a known simplification). This overlays a static alignment
 * guide over the live camera feed so the user frames their own eye, then
 * captures a single centered square frame.
 *
 * The <video> element is ALWAYS mounted (never conditionally rendered) so
 * `videoRef.current` is available the instant we get a MediaStream — an
 * earlier version set `srcObject` from a requestAnimationFrame callback to
 * dodge a "ref not mounted yet" race, which was itself unreliable on
 * Safari and could leave the preview stuck on "starting camera…"
 * indefinitely. The Capture/Retake buttons are deliberately siblings of
 * the aspect-ratio-clipped video frame, not children of it — nesting them
 * inside a box with `overflow: hidden` + `aspect-ratio` risked the button
 * being squeezed out of the box's own bounds and silently clipped.
 */
export function CameraCapture({ onCapture }: { onCapture: (file: File) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [active, setActive] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [capturedUrl, setCapturedUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => stopStream();
  }, []);

  useEffect(() => {
    if (!active || videoReady) return;
    const timer = setTimeout(() => {
      setError(
        "Camera preview never started. This can happen if another app has " +
          "the camera locked, or the browser blocked it silently. Try " +
          "reloading the page, or upload a photo instead.",
      );
      stopStream();
      setActive(false);
    }, READY_TIMEOUT_MS);
    return () => clearTimeout(timer);
  }, [active, videoReady]);

  function stopStream() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }

  async function enable() {
    setError(null);
    setCapturedUrl(null);
    setVideoReady(false);
    setActive(true); // mount is synchronous with this render; video element already exists in the DOM below
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) throw new Error("video element missing");
      video.srcObject = stream;
      try {
        await video.play();
      } catch {
        // Some browsers reject play() if metadata isn't loaded yet even
        // though playback starts fine once it is — not fatal on its own,
        // onLoadedMetadata below still fires and unblocks the Capture button.
      }
    } catch {
      setActive(false);
      setError("Camera unavailable. Allow camera access in your browser, or upload a photo instead.");
    }
  }

  function capture() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      setError("Video isn't ready yet. Wait a moment for the preview to appear, then try again.");
      return;
    }
    setError(null);

    const size = Math.min(video.videoWidth, video.videoHeight);
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setError("Could not read the camera frame in this browser.");
      return;
    }
    // Mirror horizontally so the captured still matches the mirrored
    // preview the user was actually looking at (see .video's CSS transform).
    ctx.translate(size, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(
      video,
      (video.videoWidth - size) / 2,
      (video.videoHeight - size) / 2,
      size,
      size,
      0,
      0,
      size,
      size,
    );
    canvas.toBlob((blob) => {
      if (!blob) {
        setError("Could not capture that frame. Try again.");
        return;
      }
      stopStream(); // turn the camera light off — capture is done
      setActive(false);
      setCapturedUrl(URL.createObjectURL(blob));
      onCapture(new File([blob], "capture.png", { type: "image/png" }));
    }, "image/png");
  }

  function retake() {
    setCapturedUrl(null);
    enable();
  }

  const showVideo = active && !capturedUrl;

  return (
    <div>
      {!active && !capturedUrl && (
        <button onClick={enable} style={{ width: "100%" }}>
          ENABLE CAMERA
        </button>
      )}

      {/* Always mounted once we've ever gone active, so the ref is stable;
          hidden via CSS rather than unmounted. */}
      <div className={styles.videoWrap} style={{ display: showVideo ? "block" : "none" }}>
        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <video
          ref={videoRef}
          className={styles.video}
          muted
          autoPlay
          playsInline
          onLoadedMetadata={() => setVideoReady(true)}
        />
        <div className={styles.guide} />
        {!videoReady && <div className={styles.starting}>starting camera…</div>}
      </div>
      {showVideo && (
        <button
          className="primary"
          onClick={capture}
          disabled={!videoReady}
          style={{ width: "100%", marginTop: 10 }}
        >
          {videoReady ? "CAPTURE ▶" : "WAITING FOR CAMERA…"}
        </button>
      )}

      {capturedUrl && (
        <div className={styles.videoWrap}>
          <img src={capturedUrl} alt="Captured frame" className={styles.capturedImg} />
          <div className={styles.captured}>✓ CAPTURED</div>
        </div>
      )}
      {capturedUrl && (
        <button onClick={retake} style={{ width: "100%", marginTop: 10 }}>
          RETAKE
        </button>
      )}

      {error && <div className={styles.error}>{error}</div>}
    </div>
  );
}
