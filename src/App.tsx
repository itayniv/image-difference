import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import ImageDropZone from './Components/ImageDropZone'
import { pipeline, env } from '@huggingface/transformers'
import { compareOriginalToAI, extractVectorsFromFiles, convertChipsToFiles, convertProcessResultsToFiles } from './Components/Helpers';
import { UMAP } from 'umap-js';
import { processFaceFile } from './Components/faceLandmarker'
import FaceOverlay from './Components/FaceOverlay'
import { ATTRIBUTES } from './Components/Attributes'
import { compilePrompts } from './Components/Helpers'
// Temporarily disable browser cache to avoid using any corrupted ONNX files
env.useBrowserCache = false


const CLIP_MODEL_ID = "Xenova/clip-vit-base-patch32"; // image model
const TEXT_MODEL_ID = "Xenova/all-MiniLM-L6-v2"; // robust text embedding model

// Import reference images
import img1 from './assets/ref_photos/IMG_7812.jpeg'
import img2 from './assets/ref_photos/IMG_8236.jpeg'
import img3 from './assets/ref_photos/IMG_8910 Medium.jpeg'
import img4 from './assets/ref_photos/IMG_9157 Medium.jpeg'
import img5 from './assets/ref_photos/IMG_9158 Medium.jpeg'
import img6 from './assets/ref_photos/IMG_9159 Medium.jpeg'
import img7 from './assets/ref_photos/IMG_9160 Medium.jpeg'
import img8 from './assets/ref_photos/IMG_9161 Medium.jpeg'
import img9 from './assets/ref_photos/IMG_9162 Medium.jpeg'
import img10 from './assets/ref_photos/itay_ref_01.jpeg'
import img11 from './assets/ref_photos/itay_ref_02.jpeg'
import img12 from './assets/ref_photos/itay_ref_03.jpeg'
import img13 from './assets/ref_photos/itay_ref_04.jpeg'
import img14 from './assets/ref_photos/itay_ref_05.jpeg'
import img15 from './assets/ref_photos/itay_cu1.jpeg'
import img16 from './assets/ref_photos/itay_cu2.jpeg'
import img17 from './assets/ref_photos/itay_cu3.jpeg'
import img18 from './assets/ref_photos/itay_cu4.jpeg'


function App() {
  const [uploadedAIFiles, setUploadedAIFiles] = useState<File[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const textCacheRef = useRef<Map<string, Float32Array>>(new Map());

  // reserved for future face detection UI

  const [overlays, setOverlays] = useState<
    Array<{
      id: string
      imageURL: string
      imageSize: { width: number; height: number }
      landmarksPerFace: { x: number; y: number }[][]
      chipURLs: string[]
      sourceName: string
    }>
  >([])
  const previousURLsRef = useRef<string[]>([])

  // Cache the image feature extractor so the model loads only once
  const imageFeatureExtractorRef = useRef<any | null>(null)
  const textFeatureExtractorRef = useRef<any | null>(null)

  const [isExtractorLoading, setIsExtractorLoading] = useState<boolean>(true)

  // Helper to convert imported image URLs to File objects
  const urlToFile = useCallback(async (url: string, filename: string): Promise<File> => {
    const response = await fetch(url)
    const blob = await response.blob()
    return new File([blob], filename, { type: blob.type })
  }, [])


  // Load reference images and populate drop zones
  const loadReferenceImages = useCallback(async () => {
    try {
      const imageUrls = [
        // { url: img9, name: 'IMG_9162 Medium.jpeg' },
        // { url: img1, name: 'IMG_7812.jpeg' },
        // { url: img2, name: 'IMG_8236.jpeg' },
        // { url: img4, name: 'IMG_9157 Medium.jpeg' },
        // { url: img3, name: 'IMG_8910 Medium.jpeg' },
        // { url: img5, name: 'IMG_9158 Medium.jpeg' },
        // { url: img6, name: 'IMG_9159 Medium.jpeg' },
        // { url: img7, name: 'IMG_9160 Medium.jpeg' },
        // { url: img8, name: 'IMG_9161 Medium.jpeg' },
        // { url: img10, name: 'IMG_9163 Medium.jpeg' },
        // { url: img11, name: 'itay_ref_01.jpeg' },
        // { url: img12, name: 'itay_ref_02.jpeg' },
        // { url: img13, name: 'itay_ref_03.jpeg' },
        // { url: img14, name: 'itay_ref_04.jpeg' },
        // { url: img15, name: 'itay_cu1.jpeg' },
        { url: img16, name: 'itay_cu2.jpeg' },
        // { url: img17, name: 'itay_cu3.jpeg' },
        { url: img18, name: 'itay_cu4.jpeg' }
      ]

      const files = await Promise.all(
        imageUrls.map(({ url, name }) => urlToFile(url, name))
      )

      // Split files between the two drop zones (first half to original, second half to AI)

      const originalFiles = files.slice(0, 1)
      const aiFiles = files.slice(1)

      setUploadedFiles(originalFiles)
      setUploadedAIFiles(aiFiles)

    } catch (error) {
      console.error('Failed to load reference images:', error)
    }
  }, [urlToFile])

  const getImageFeatureExtractor = useCallback(async () => {
    if (!imageFeatureExtractorRef.current) {
      imageFeatureExtractorRef.current = await pipeline(
        'image-feature-extraction',
        // 'Xenova/vit-base-patch16-224-in21k'
        CLIP_MODEL_ID,
        {
          progress_callback: (status: any) => {
            try {
              console.log('[image model load]', status);
            } catch {}
          },
        }
      )
    }

    return imageFeatureExtractorRef.current
  }, [])

  const getTextFeatureExtractor = useCallback(async () => {
    if (!textFeatureExtractorRef.current) {
      textFeatureExtractorRef.current = await pipeline(
        'feature-extraction',
        TEXT_MODEL_ID,
        {
          progress_callback: (status: any) => {
            try {
              console.log('[text model load]', status);
            } catch {}
          },
        }
      )
    }
    return textFeatureExtractorRef.current
  }, [])



  const embedTexts = useCallback(async (prompts: string[]): Promise<Array<{prompt: string, vector: Float32Array}>> => {
    if (!textFeatureExtractorRef.current) await getTextFeatureExtractor();
    const cache = textCacheRef.current; 
    const toRun: string[] = []; 
    const order: number[] = [];
    
    // Find prompts that aren't cached yet
    prompts.forEach((p, i) => {
      if (!cache.has(p)) {
        toRun.push(p);
        order.push(i);
      }
    });

    // Generate embeddings for uncached prompts
    if (toRun.length > 0) {
      const extractor = await getTextFeatureExtractor();
      for (const prompt of toRun) {
        try {
          const result = await extractor(prompt, { pooling: 'mean', normalize: true });
          const vector = (result && (result as any).data)
            ? new Float32Array((result as any).data)
            : new Float32Array(result as unknown as Float32Array);
          cache.set(prompt, vector);
        } catch (error) {
          console.error(`Failed to generate embedding for prompt: "${prompt}"`, error);
          cache.set(prompt, new Float32Array(384));
        }
      }
    }

    console.log('Text embeddings:', cache)
    // Return all prompts with their corresponding vectors (from cache)
    return prompts.map(prompt => ({
      prompt,
      vector: cache.get(prompt)!
    }));
  }, []);



  const handleImagesUploaded = (files: File[]) => {
    setUploadedFiles(files)
    console.log('Uploaded files:', files)
  }

  const handleAIImagesUploaded = (files: File[]) => {
    setUploadedAIFiles(files)
    console.log('Uploaded AI files:', files)
  }

  const analyizeImages = async (originalFiles: File[], aiFiles: File[]) => {

    const prompts = compilePrompts(ATTRIBUTES)
    const textEmbeddings = await embedTexts(Object.values(prompts).flat())



    // Combine both sets (or adjust to pass only one set if desired)
    const combined = [...originalFiles, ...aiFiles]


    // run the face landmarker on all the files
    const results = await Promise.all(
      combined.map(async (file) => {
        const result = await processFaceFile(file, {
          outputSize: 512,    // Larger output size for better quality
          paddingFactor: 1.5  // 2x padding for much larger crops
        })
        const imageURL = URL.createObjectURL(file)
        const chipURLs = result.faces.map((f) => URL.createObjectURL(f.chip.blob))
        return {
          id: `${file.name}-${file.size}-${file.lastModified}`,
          imageURL,
          imageSize: result.imageSize,
          landmarksPerFace: result.faces.map((f) => f.landmarks.map((p) => ({ x: p.x, y: p.y }))),
          chipURLs,
          sourceName: file.name,
        }
      })
    )



    // Revoke previous URLs to avoid leaks
    previousURLsRef.current.forEach((u) => URL.revokeObjectURL(u))
    previousURLsRef.current = []

    const allURLs: string[] = []
    results.forEach((r) => {
      allURLs.push(r.imageURL, ...r.chipURLs)
    })
    previousURLsRef.current = allURLs

    // console.log('Results of all the files:', results)
    // setOverlays(results)


    const portraites = results.map(p => p.chipURLs[0])



    // Convert head shot crops (chips) to File[] format for vector extraction
    const headShotFiles = await convertChipsToFiles(results, { includeAllFaces: true })
   
    const extractor = await getImageFeatureExtractor()
    // get embedding from one image
    const imgEmbedding = await extractor(headShotFiles[0], { pooling: 'mean', normalize: true })
    console.log('image Embedding:', imgEmbedding.data)



    // console.log('Head shot files:', headShotFiles)

    // Compute cosine similarity between original and AI images (pairwise by index)
    // const extractor = await getImageFeatureExtractor()

    // const textExtractor = await getTextFeatureExtractor()


    const headShotResults = await compareOriginalToAI([headShotFiles[0]], [headShotFiles[1]], extractor)
    // console.log('Head shot results:', headShotResults)

    // const extractedVectors = await extractVectorsFromFiles(combined, extractor)
    // Extract vectors from head shot crops instead of original images
    const extractedVectors = await extractVectorsFromFiles(headShotFiles, extractor)
    // await compareOriginalToAI(originalFiles, aiFiles, extractor)


    // console.log('Extracted vectors from head shot crops:', extractedVectors)

    // const vectorMatrix = extractedVectors.map(v => v.vector)
    //      // TODO: add UMAP to visualize the similarity
    //  const umap = new UMAP({ 
    //    nNeighbors: Math.min(5, vectorMatrix.length - 1), 
    //    minDist: 0.1, 
    //    nComponents: 2 
    //  });
    //  const embedding = umap.fit(vectorMatrix);
    //  console.log('Embedded umap:', embedding);
  }

  useEffect(() => {
    // Warm up the extractors and load reference images on mount
    const initialize = async () => {
      try {
        const imageExtractorPromise = getImageFeatureExtractor()
        const textExtractorPromise = getTextFeatureExtractor()
        const refsPromise = loadReferenceImages()

        await Promise.all([imageExtractorPromise, textExtractorPromise, refsPromise])
        setIsExtractorLoading(false)
      } catch (e) {
        console.error('Failed to initialize:', e)
        setIsExtractorLoading(false)
      }
    }
    initialize()
  }, [getImageFeatureExtractor, getTextFeatureExtractor, loadReferenceImages])


  return (
    <>
      <div className="flex flex-row gap-4">

        <div className="card">
          <h2>Image Drop Zone</h2>
          <ImageDropZone
            onImagesUploaded={handleImagesUploaded}
            maxFiles={10}
            files={uploadedFiles}
          />
        </div>
        <div className="card">
          <h2>AI Image Drop Zone</h2>
          <ImageDropZone
            onImagesUploaded={handleAIImagesUploaded}
            maxFiles={10}
            files={uploadedAIFiles}
          />
        </div>
      </div>

      <div className="mt-4">
        <button
          type="button"
          onClick={() => analyizeImages(uploadedFiles, uploadedAIFiles)}
          className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
          disabled={uploadedFiles.length === 0 || uploadedAIFiles.length === 0}
        >
          analyize similarity
        </button>
      </div>

      <div className="mb-4">
        {isExtractorLoading ? (
          <div className="text-sm text-gray-600">Loading image feature extractor model...</div>
        ) : (
          <div className="text-sm text-green-700">image featuer extractor model</div>
        )}
      </div>
      {overlays.length > 0 && (
        <div className="mt-6 space-y-8">
          {overlays.map((ov) => (
            <div key={ov.id} className="border rounded p-4">
              <div className="mb-2 text-sm text-gray-700">{ov.sourceName}</div>
              <FaceOverlay
                imageURL={ov.imageURL}
                imageSize={ov.imageSize}
                landmarksPerFace={ov.landmarksPerFace}
                maxWidth={640}
              />
              {ov.chipURLs.length > 0 && (
                <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
                  {ov.chipURLs.map((url, idx) => (
                    <div key={`${ov.id}-chip-${idx}`} className="flex flex-col items-center">
                      <img src={url} alt={`chip-${idx}`} className="w-full h-auto rounded border" />
                      <span className="text-xs text-gray-500 mt-1">face {idx + 1}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      {/* reserved for future face detection UI */}
    </>
  )
}

export default App
