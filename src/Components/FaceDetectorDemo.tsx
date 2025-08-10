import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./FaceDetectorDemo.css";

import type {
  FaceDetector as FaceDetectorType,
  FilesetResolver as FilesetResolverType,
  Detection as DetectionType,
} from "@mediapipe/tasks-vision";

const WASM_PATH =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

type FaceDetectorDemoProps = {
  files: File[];
  onComplete?: () => void;
};

export default function FaceDetectorDemo({ files, onComplete }: FaceDetectorDemoProps) {
  const [isReady, setIsReady] = useState(false);
  const faceDetectorRef = useRef<FaceDetectorType | null>(null);
  const imageRefs = useRef<Array<HTMLImageElement | null>>([]);
  const processedSetRef = useRef<Set<number>>(new Set());
  
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
          console.log("FaceDetector initialized");
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


  const displayImageDetections = useCallback(
    (detections: DetectionType[], imgEl: HTMLImageElement) => {
      const container = imgEl.parentElement as HTMLElement;
      // Use rendered size to compute precise X/Y scale factors
      const renderedWidth = imgEl.clientWidth;
      const renderedHeight = imgEl.clientHeight;
      const ratioX = renderedWidth / imgEl.naturalWidth;
      const ratioY = renderedHeight / imgEl.naturalHeight;

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
        p.style.left = `${bbox.originX * ratioX}px`;
        p.style.top = `${bbox.originY * ratioY - 30}px`;
        p.style.width = `${bbox.width * ratioX - 10}px`;

        const highlighter = document.createElement("div");
        highlighter.className = "highlighter";
        highlighter.style.left = `${bbox.originX * ratioX}px`;
        highlighter.style.top = `${bbox.originY * ratioY}px`;
        highlighter.style.width = `${bbox.width * ratioX}px`;
        highlighter.style.height = `${bbox.height * ratioY}px`;

        container.appendChild(highlighter);
        container.appendChild(p);

        detection.keypoints?.forEach((keypoint) => {
          const keypointEl = document.createElement("span");
          keypointEl.className = "key-point";
          keypointEl.style.top = `${keypoint.y * renderedHeight - 3}px`;
          keypointEl.style.left = `${keypoint.x * renderedWidth - 3}px`;
          container.appendChild(keypointEl);
        });
      });
    },
    []
  );

  // Create object URLs for incoming files and clean them up when files change
  const objectUrls = useMemo(() => {
    return files.map((file) => ({
      url: URL.createObjectURL(file),
      name: file.name,
    }));
  }, [files]);

  useEffect(() => {
    return () => {
      objectUrls.forEach(({ url }) => URL.revokeObjectURL(url));
      processedSetRef.current.clear();
    };
  }, [objectUrls]);

  const detectOnImage = useCallback((img: HTMLImageElement | null, idx?: number) => {
    if (!img) return;
    const faceDetector = faceDetectorRef.current;
    if (!faceDetector) return;

    const result = faceDetector.detect(img);
    displayImageDetections(result.detections as DetectionType[], img);
    if (typeof idx === "number") {
      processedSetRef.current.add(idx);
    }
  }, [displayImageDetections]);

  // Run detection when the detector is ready and images are loaded
  useEffect(() => {
    if (!isReady) return;
    imageRefs.current.forEach((img, idx) => {
      if (img && img.complete && img.naturalWidth > 0) {
        detectOnImage(img, idx);
      }
    });
    // If all processed, notify
    if (
      objectUrls.length > 0 &&
      processedSetRef.current.size === objectUrls.length &&
      onComplete
    ) {
      onComplete();
    }
  }, [isReady, objectUrls, detectOnImage, onComplete]);

  const onImageLoad = useCallback((idx: number) => (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    if (!isReady) return; // will run via effect when ready
    detectOnImage(img, idx);
  }, [detectOnImage, isReady]);

  return (
    <div className="face-detector-demo">
      <h2 className="mb-2">MediaPipe Face Detector</h2>
      {!isReady && <p>Loading face detector...</p>}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        {objectUrls.map(({ url, name }, idx) => (
          <div key={idx} className="image-container relative inline-block">
            <img
              ref={(el) => {
                imageRefs.current[idx] = el;
              }}
              src={url}
              alt={name || `image ${idx}`}
              className="max-w-full h-auto"
              onLoad={onImageLoad(idx)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}


