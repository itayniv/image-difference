import React, { useCallback, useEffect, useRef, useState } from "react";
import "./FaceDetectorDemo.css";
import img9157 from "../assets/ref_photos/IMG_9157 Medium.jpeg";
import img9158 from "../assets/ref_photos/IMG_9158 Medium.jpeg";
import img9159 from "../assets/ref_photos/IMG_9159 Medium.jpeg";
import img9160 from "../assets/ref_photos/IMG_9160 Medium.jpeg";
import img9161 from "../assets/ref_photos/IMG_9161 Medium.jpeg";
import img9162 from "../assets/ref_photos/IMG_9162 Medium.jpeg";

import type {
  FaceDetector as FaceDetectorType,
  FilesetResolver as FilesetResolverType,
  Detection as DetectionType,
} from "@mediapipe/tasks-vision";

const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

export default function FaceDetectorDemo() {
  const [isReady, setIsReady] = useState(false);
  const faceDetectorRef = useRef<FaceDetectorType | null>(null);
  

  // Lazy import to avoid SSR/build hiccups and keep wasm on CDN
  useEffect(() => {
    let cancelled = false;
    const init = async () => {
      try {
        const visionModule = await import("@mediapipe/tasks-vision");
        const FilesetResolver: typeof FilesetResolverType = visionModule.FilesetResolver;
        const FaceDetector: typeof FaceDetectorType = visionModule.FaceDetector;

        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
        const detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: "GPU",
          },
          runningMode: "IMAGE",
        });
        if (!cancelled) {
          faceDetectorRef.current = detector;
          setIsReady(true);
        } else {
          detector.close();
        }
      } catch (error) {
        console.error("Failed to initialize FaceDetector", error);
      }
    };
    init();
    return () => {
      cancelled = true;
      if (faceDetectorRef.current) {
        faceDetectorRef.current.close();
        faceDetectorRef.current = null;
      }
    };
  }, []);

  const clearOverlays = useCallback((container: HTMLElement) => {
    const selector = ".highlighter, .info, .key-point";
    container.querySelectorAll(selector).forEach((el) => el.remove());
  }, []);

  const displayImageDetections = useCallback(
    (detections: DetectionType[], imgEl: HTMLImageElement) => {
      const container = imgEl.parentElement as HTMLElement;
      const ratio = imgEl.height / imgEl.naturalHeight;

      detections.forEach((detection) => {
        const bbox = detection.boundingBox;
        if (!bbox) {
          return;
        }
        const p = document.createElement("p");
        p.className = "info";
        p.innerText = `Confidence: ${Math.round(
          Number(detection.categories?.[0]?.score || 0) * 100
        )}% .`;
        p.style.left = `${bbox.originX * ratio}px`;
        p.style.top = `${bbox.originY * ratio - 30}px`;
        p.style.width = `${bbox.width * ratio - 10}px`;

        const highlighter = document.createElement("div");
        highlighter.className = "highlighter";
        highlighter.style.left = `${bbox.originX * ratio}px`;
        highlighter.style.top = `${bbox.originY * ratio}px`;
        highlighter.style.width = `${bbox.width * ratio}px`;
        highlighter.style.height = `${bbox.height * ratio}px`;

        container.appendChild(highlighter);
        container.appendChild(p);

        detection.keypoints?.forEach((keypoint) => {
          const keypointEl = document.createElement("span");
          keypointEl.className = "key-point";
          keypointEl.style.top = `${keypoint.y * imgEl.height - 3}px`;
          keypointEl.style.left = `${keypoint.x * imgEl.width - 3}px`;
          container.appendChild(keypointEl);
        });
      });
    },
    []
  );

  const onImageClick = useCallback(async (e: React.MouseEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    const container = img.parentElement as HTMLElement;
    clearOverlays(container);

    const faceDetector = faceDetectorRef.current;
    if (!faceDetector) return;

    const result = faceDetector.detect(img);
    displayImageDetections(result.detections as DetectionType[], img);
  }, [clearOverlays, displayImageDetections]);

  const images = [img9157, img9158, img9159, img9160, img9161, img9162];

  return (
    <div className="face-detector-demo">
      <h2 className="mb-2">MediaPipe Face Detector</h2>
      {!isReady && <p>Loading face detector...</p>}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        {images.map((src, idx) => (
          <div key={idx} className="image-container relative inline-block">
            <img
              src={src}
              alt={`ref ${idx}`}
              className="cursor-pointer max-w-full h-auto"
              onClick={onImageClick}
            />
          </div>
        ))}
      </div>

      {/* Webcam functionality removed as requested */}
    </div>
  );
}


