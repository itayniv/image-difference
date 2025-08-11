import { useEffect, useRef } from "react";

type Point = { x: number; y: number };

export type FaceOverlayProps = {
  imageURL: string;
  imageSize?: { width: number; height: number };
  // One landmarks array per detected face (values are normalized [0,1])
  landmarksPerFace: Point[][];
  // Optional maximum render width; height will keep aspect ratio
  maxWidth?: number;
  // Whether to show landmark overlays
  showOverlays?: boolean;
};

export default function FaceOverlay({
  imageURL,
  imageSize,
  landmarksPerFace,
  maxWidth = 512,
  showOverlays = true,
}: FaceOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let isCancelled = false;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if (isCancelled) return;

      const naturalW = imageSize?.width ?? img.naturalWidth;
      const naturalH = imageSize?.height ?? img.naturalHeight;

      const scale = Math.min(1, maxWidth / naturalW);
      const cw = Math.round(naturalW * scale);
      const ch = Math.round(naturalH * scale);

      const canvas = canvasRef.current!;
      const ctx = canvas.getContext("2d")!;
      canvas.width = cw;
      canvas.height = ch;

      // Draw image
      ctx.clearRect(0, 0, cw, ch);
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(img, 0, 0, cw, ch);

      // Draw landmarks per face (only if showOverlays is true)
      if (showOverlays) {
        const colors = [
          "#00E5FF",
          "#FF3D00",
          "#76FF03",
          "#FFC400",
          "#7C4DFF",
          "#FF4081",
        ];
        landmarksPerFace.forEach((points, faceIdx) => {
          const color = colors[faceIdx % colors.length];
          ctx.fillStyle = color;
          ctx.strokeStyle = color;
          const radius = Math.max(1.5, Math.min(cw, ch) * 0.001);

          for (const p of points) {
            const x = p.x * cw;
            const y = p.y * ch;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();
          }
        });
      }
    };
    img.src = imageURL;

    return () => {
      isCancelled = true;
    };
  }, [imageURL, imageSize?.width, imageSize?.height, landmarksPerFace, maxWidth, showOverlays]);

  return <canvas ref={canvasRef} style={{ width: "100%", height: "auto" }} />;
}


