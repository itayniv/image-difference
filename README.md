# Image Analyzer (Face Similarity & Attribute Matching)

A React + TypeScript + Vite web app that detects faces, crops aligned headshots, computes embeddings with CLIP in the browser, and compares a target image to reference images. It also computes text-to-image similarity for interpretable attributes (age, gender, facial hair, etc.) using CLIP text embeddings.

## Key Features

- Detects faces and extracts aligned face chips via MediaPipe Face Landmarker (WebAssembly, no server).
- Generates image embeddings (full image and per-face) using `@huggingface/transformers` (CLIP vision) in-browser.
- Generates text embeddings for human-readable attributes using CLIP text encoder; computes per-face attribute matches.
- Compares each reference face to the target face via cosine similarity and sorts results accordingly.
- Visualizes landmarks, extracted face chips, and similarity/attribute scores.

## How It Works (Pipeline)

1) Input
- You provide one Target image and up to 10 Reference images.
- The UI bundles some sample images on first load for quick testing.

2) Face detection & chips
- `processFaceFile` in `src/Components/faceLandmarker.ts` uses MediaPipe Face Landmarker to detect faces and landmarks.
- It computes an affine transform that aligns eyes to target positions, draws to a square canvas, and returns a PNG "chip" Blob per face.

3) Embeddings
- Image embeddings: `pipeline('image-feature-extraction', 'Xenova/clip-vit-base-patch32')` returns normalized vectors for full images and for each face chip.
- Text embeddings: `AutoTokenizer` + `CLIPTextModelWithProjection` load CLIP text encoder and embed prompts generated from `ATTRIBUTES`.

4) Similarity calculations
- Cosine similarity between the Target's first face and each Reference face.
- Text-to-image similarity for each face vs. every text attribute prompt.
- Best attribute per category is selected and used to compare to the Target's best attributes.

5) Sorting & display
- Reference images are sorted by max similarity to the Target.
- UI shows the Target (left) and references (right) with landmarks, face chips, similarity to source, best text attributes, and a summary of text similarity to source.

## App Structure

- `src/App.tsx`: Orchestrates the entire pipeline.
  - Loads CLIP models (image feature extractor, tokenizer, text model) and sample images.
  - Manages the `imageDataset` state with face detection, embeddings, text similarity, and comparisons.
  - Provides handlers for uploads and kicks off `analyizeImages()`.
  - Computes and stores:
    - Face detection results and chip URLs
    - Embeddings (full + faces)
    - Text similarity results for attributes
    - Similarity to source (Target) and sorts by similarity
  - Renders:
    - Two `ImageDropZone` components (Target and References)
    - `ImageWithLandmarks` to visualize chips and overlays
    - `ActionFooter` with a progress loader and Analyze button

### Core Types
- `src/imageAnalysisTypes.ts` defines:
  - `ImageData`: metadata, faceDetection, embeddings, textSimilarity, computeSimilarityToSource, textSimilarityToSource, and processing stages.
  - `ImageAnalysisDataset`: the collection of `ImageData` and global metadata.

### Components
- `Components/ImageDropZone.tsx`
  - Drag/drop and click-to-upload for images.
  - Props: `onImagesUploaded(files)`, `maxFiles`, `acceptedTypes`, `files` (controlled display).
  - Generates previews and supports removing/clearing selections.

- `Components/ActionFooter.tsx`
  - Sticky footer showing model loading progress and an Analyze button.
  - Props: `isAnalyzing`, `onAnalyzeImages`, `isAnalyzeDisabled`, `isExtractorLoading`, `loadingProgress`.

- `Components/ImageWithLandmarksHelper.tsx`
  - Lays out a reference (Target) image on the left and other images on the right.
  - Toggles landmark overlays via `Toggle`.
  - Renders per-image metrics: similarity to Target, best text attributes, text similarity-to-source summary, and processing errors.
  - Uses `FaceOverlay` to draw landmarks over the original image, and lists extracted face chips.
  - Accepts `imageDataset` to fetch and display analysis details aligned with each rendered image.

- `Components/FaceOverlay.tsx`
  - Draws an image to a canvas and optionally overlays landmarks (per-face arrays of normalized points).
  - Props: `imageURL`, `imageSize`, `landmarksPerFace`, `maxWidth`, `showOverlays`.

- `Components/faceLandmarker.ts`
  - Wraps MediaPipe Face Landmarker: loads the wasm files and performs detection on images.
  - Produces `ProcessResult` with robust eye center estimation, similarity transform, configurable padding, and PNG face chips.
  - Public API: `processFaceFile(file, options)`.

- `Components/Attributes.ts`
  - Defines `ATTRIBUTES` with categories (gender, age, skinTone, etc.).
  - Each attribute provides a prompt template used to generate CLIP text prompts.

- `Components/Helpers.ts`
  - Utility functions for embeddings, converting chips to `File[]`, cosine similarity, and `compilePrompts`.

- `Components/LoadingComponent.tsx`
  - Progress bar with percentage and message.

- `Components/PageTitle.tsx`
  - Page header text.

- `Components/Toggle.tsx`
  - Accessible toggle component used to show/hide landmarks.

- `Components/FaceDetectorDemo.tsx`
  - Optional demo using MediaPipe Face Detector (bounding boxes & keypoints). Not used in the main flow.

## Installation & Run

Requirements: Node 18+ recommended.

```bash
npm install
npm run dev
```

Open the printed local URL. The app will load models in the background and show sample images. Add your own Target and References if you like, then click "Analyze Images".

## Configuration & Tuning

- Model IDs: Set CLIP model via `CLIP_MODEL_ID` in `src/App.tsx`.
- Face crop size and padding: `outputSize` and `paddingFactor` in calls to `processFaceFile`.
- Attributes: edit `ATTRIBUTES` in `Components/Attributes.ts` to add categories/options or change templates.
- Max files: adjust `maxFiles` prop on `ImageDropZone`.

## Data & Privacy

All computation runs locally in your browser (WebAssembly/WebGPU). Images are not uploaded to a server by this app.

## Notes

- CLIP models and MediaPipe wasm are downloaded at runtime and cached by the browser.
- First load can take time depending on connection and hardware.
- The app normalizes vectors for cosine similarity; values range [-1, 1].

## License

MIT (see repository).
