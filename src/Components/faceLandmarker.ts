// faceLandmarker.ts
import {
    FilesetResolver,
    FaceLandmarker,
    type FaceLandmarkerResult,
  } from "@mediapipe/tasks-vision";
  
  export type FaceChip = {
    canvas: HTMLCanvasElement;
    blob: Blob;
    targetEyes: { left: { x: number; y: number }; right: { x: number; y: number } };
    transform: { scale: number; theta: number; tx: number; ty: number };
  };
  
  export type ProcessedFace = {
    landmarks: { x: number; y: number; z?: number }[];
    blendshapes?: FaceLandmarkerResult["faceBlendshapes"][number];
    matrices?: FaceLandmarkerResult["facialTransformationMatrixes"][number];
    chip: FaceChip;
  };
  
  export type ProcessResult = {
    file: File;
    imageSize: { width: number; height: number };
    faces: ProcessedFace[];
    raw: FaceLandmarkerResult;
  };
  
  export type ProcessOptions = {
    outputSize?: number; // default 256
    targetEyePos?: {
      left: { x: number; y: number };
      right: { x: number; y: number };
    }; // default L(0.35,0.40), R(0.65,0.40)
    interocularFrac?: number; // default 0.34 (ignored if targetEyePos provided)
    modelAssetPath?: string; // default hosted model
    enableBlendshapes?: boolean; // default false
    enableMatrices?: boolean;    // default false
    maxFaces?: number;           // default 10
    paddingFactor?: number;      // default 1.0 (1.5 = 50% more padding, 2.0 = 100% more)
  };
  
  // ---------- singleton loader ----------
  let _landmarker: FaceLandmarker | null = null;
  let _loading: Promise<FaceLandmarker> | null = null;
  
  async function getLandmarker(opts: ProcessOptions): Promise<FaceLandmarker> {
    if (_landmarker) return _landmarker;
    if (_loading) return _loading;
  
    const {
      modelAssetPath = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      enableBlendshapes = false,
      enableMatrices = false,
      maxFaces = 10,
    } = opts;
  
    _loading = (async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      _landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath },
        runningMode: "IMAGE",
        numFaces: maxFaces,
        outputFaceBlendshapes: enableBlendshapes,
        outputFacialTransformationMatrixes: enableMatrices,
      });
      return _landmarker!;
    })();
  
    return _loading;
  }
  
  // ---------- public API ----------
  export async function processFaceFile(
    file: File,
    options: ProcessOptions = {}
  ): Promise<ProcessResult> {
    const {
      outputSize = 256,
      targetEyePos,
      interocularFrac = 0.34,
      paddingFactor = 1.0,
    } = options;
  
    const landmarker = await getLandmarker(options);
    const img = await fileToImage(file);
    const raw = landmarker.detect(img);
  
    const size = { width: img.width, height: img.height };
    const faces: ProcessedFace[] = [];
  
    const chipW = outputSize;
    const chipH = outputSize;
  
    // Default eye targets if not explicitly provided
    const defaultTargets = {
      left: { x: 0.35 * chipW, y: 0.40 * chipH },
      right: { x: 0.65 * chipW, y: 0.40 * chipH },
    };
  
    for (let i = 0; i < (raw.faceLandmarks?.length ?? 0); i++) {
      const lm = raw.faceLandmarks![i];
  
      // Robust eye centers using multiple contour points (MediaPipe FaceMesh indices)
      const leftEyeIdx = [33, 133, 159, 145, 246, 161, 160, 157, 173, 155];
      const rightEyeIdx = [362, 263, 386, 374, 466, 384, 385, 387, 373, 380];
  
      const left = meanPts(lm, leftEyeIdx, size);
      const right = meanPts(lm, rightEyeIdx, size);
  
      let destL = targetEyePos?.left
        ? { x: targetEyePos.left.x * chipW, y: targetEyePos.left.y * chipH }
        : defaultTargets.left;
  
      let destR = targetEyePos?.right
        ? { x: targetEyePos.right.x * chipW, y: targetEyePos.right.y * chipH }
        : defaultTargets.right;
  
      // If caller specified only interocular distance, enforce symmetric targets
      if (!targetEyePos) {
        const cx = chipW / 2;
        const ey = defaultTargets.left.y;
        const half = (interocularFrac * chipW) / 2;
        destL = { x: cx - half, y: ey };
        destR = { x: cx + half, y: ey };
      }
  
            const { scale, theta } = solveSimilarity(left, right, destL, destR);

      // Apply padding by scaling down the transform (larger crop)
      const paddedScale = scale / paddingFactor;

      const canvas = document.createElement("canvas");
      canvas.width = chipW;
      canvas.height = chipH;
      const ctx = canvas.getContext("2d")!;
      ctx.imageSmoothingQuality = "high";
      ctx.save();
      ctx.translate(destL.x, destL.y);
      ctx.rotate(theta);
      ctx.scale(paddedScale, paddedScale);
      ctx.translate(-left.x, -left.y);
      ctx.drawImage(img, 0, 0);
      ctx.restore();
  
      const blob = await new Promise<Blob>((res) =>
        canvas.toBlob((b) => res(b!), "image/png")
      );
  
      faces.push({
        landmarks: lm.map((p) => ({ x: p.x, y: p.y, z: p.z })),
        blendshapes: raw.faceBlendshapes?.[i],
        matrices: raw.facialTransformationMatrixes?.[i],
        chip: {
          canvas,
          blob,
          targetEyes: { left: destL, right: destR },
          transform: { scale: paddedScale, theta, tx: destL.x, ty: destL.y },
        },
      });
    }
  
    return { file, imageSize: size, faces, raw };
  }
  
  // ---------- helpers ----------
  function fileToImage(file: File): Promise<HTMLImageElement> {
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = rej;
      img.crossOrigin = "anonymous";
      img.src = URL.createObjectURL(file);
    });
  }
  
  function meanPts(
    lm: { x: number; y: number }[],
    idxs: number[],
    size: { width: number; height: number }
  ): { x: number; y: number } {
    let sx = 0,
      sy = 0;
    for (const i of idxs) {
      sx += lm[i].x * size.width;
      sy += lm[i].y * size.height;
    }
    const n = idxs.length || 1;
    return { x: sx / n, y: sy / n };
  }
  
  function solveSimilarity(
    Ls: { x: number; y: number },
    Rs: { x: number; y: number },
    Ld: { x: number; y: number },
    Rd: { x: number; y: number }
  ): { scale: number; theta: number } {
    const vs = { x: Rs.x - Ls.x, y: Rs.y - Ls.y };
    const vd = { x: Rd.x - Ld.x, y: Rd.y - Ld.y };
    const ds = Math.hypot(vs.x, vs.y) || 1;
    const dd = Math.hypot(vd.x, vd.y) || 1;
    const scale = dd / ds;
    const angS = Math.atan2(vs.y, vs.x);
    const angD = Math.atan2(vd.y, vd.x);
    const theta = angD - angS;
    return { scale, theta };
  }