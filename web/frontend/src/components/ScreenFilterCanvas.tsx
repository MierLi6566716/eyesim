import { useEffect, useRef, useState } from "react";
import { createScreenFilter } from "../shaders/screenFilter";

export function ScreenFilterCanvas({
  imageUrl,
  kernel,
}: {
  imageUrl: string;
  kernel: number[][];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let destroy: (() => void) | undefined;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      try {
        const filter = createScreenFilter(canvas, img, kernel);
        destroy = filter.destroy;
      } catch (e) {
        setError(String(e));
      }
    };
    img.onerror = () => setError("Could not load the image for the WebGL filter.");
    img.src = imageUrl;

    return () => destroy?.();
  }, [imageUrl, kernel]);

  if (error) {
    return <div style={{ color: "var(--danger)", fontSize: 12 }}>{error}</div>;
  }
  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: "auto", display: "block", border: "1px solid var(--border)" }}
    />
  );
}
