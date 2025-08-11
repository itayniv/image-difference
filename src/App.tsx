import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import ImageDropZone from './Components/ImageDropZone'
import { pipeline, env } from '@huggingface/transformers'
import { compareOriginalToAI, extractVectorsFromFiles, convertChipsToFiles, convertProcessResultsToFiles } from './Components/Helpers';
import { UMAP } from 'umap-js';
import { processFaceFile } from './Components/faceLandmarker'
import ImageWithLandmarks from './Components/ImageWithLandmarksHelper'
import { ATTRIBUTES } from './Components/Attributes'
import { compilePrompts } from './Components/Helpers'
import LoadingComponent from './Components/LoadingComponent'

import { AutoTokenizer, CLIPTextModelWithProjection } from '@huggingface/transformers'

// Temporarily disable browser cache to avoid using any corrupted ONNX files
env.useBrowserCache = false


const CLIP_MODEL_ID = "Xenova/clip-vit-base-patch32"; // image model


import img01 from './assets/ref_photos/ref_closeup_02.png'
import img02 from './assets/ref_photos/itay_cu3.jpeg'
import img03 from './assets/ref_photos/itay_cu2.jpeg'


function App() {
  const [uploadedAIFiles, setUploadedAIFiles] = useState<File[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const textCacheRef = useRef<Map<string, Float32Array>>(new Map());
  const [textEmbeddings, setTextEmbeddings] = useState<Array<{prompt: string, vector: Float32Array}>>([])

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
  
  // Cache CLIP tokenizer and text model for direct text embedding
  const clipTokenizerRef = useRef<any | null>(null)
  const clipTextModelRef = useRef<any | null>(null)

  const [isExtractorLoading, setIsExtractorLoading] = useState<boolean>(true)
  const [loadingProgress, setLoadingProgress] = useState<number>(0)

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

        { url: img01, name: 'ref_closeup_02.png' },
        { url: img02, name: 'itay_cu3.jpeg' }
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
              if (status.progress !== undefined) {
                setLoadingProgress(status.progress); 
              }
            } catch {}
          },
        }
      )
    }
    return imageFeatureExtractorRef.current
  }, [])

  useEffect(() => {
    const initialize = async () => {
      const prompts = compilePrompts(ATTRIBUTES);
      const textEmbeddings = await embedTexts(Object.values(prompts).flat());
      setTextEmbeddings(textEmbeddings)
    }
    initialize()
  },[])

  const getClipTokenizer = useCallback(async () => {
    if (!clipTokenizerRef.current) {
      console.log('[Loading CLIP tokenizer...]');
      setLoadingProgress(prev => Math.max(prev, 0.4)); // Start tokenizer at 40%
      clipTokenizerRef.current = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch32');
      setLoadingProgress(prev => Math.max(prev, 0.6)); // Complete tokenizer at 60%
    }
    return clipTokenizerRef.current;
  }, []);

  const getClipTextModel = useCallback(async () => {
    if (!clipTextModelRef.current) {
      console.log('[Loading CLIP text model...]');
      setLoadingProgress(prev => Math.max(prev, 0.6)); // Start text model at 60%
      clipTextModelRef.current = await CLIPTextModelWithProjection.from_pretrained(
        'Xenova/clip-vit-base-patch32'
      );
      setLoadingProgress(prev => Math.max(prev, 0.9)); // Complete text model at 90%
    }
    return clipTextModelRef.current;
  }, []);



  const embedTexts = useCallback(async (prompts: string[]): Promise<Array<{prompt: string, vector: Float32Array}>> => {
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

    async function getTextEmbedding(text: string) {
      // 1) Get cached tokenizer + text model (loaded only once)
      const tokenizer = await getClipTokenizer();
      const textModel = await getClipTextModel();
    
      // 2) Tokenize
      const inputs = await tokenizer(text, {
        return_tensors: 'pt',
        padding: true,
        truncation: true,
      });
    
      // 3) Forward through TEXT model (no pixels needed)
      const { text_embeds } = await textModel(inputs); // shape: [1, 512]
    
      // 4) (Optional) L2-normalize for cosine similarity workflows
      const v = text_embeds.data as Float32Array;
      let norm = 0;
      for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
      norm = Math.sqrt(norm);
      const embedding = new Float32Array(v.length);
      for (let i = 0; i < v.length; i++) embedding[i] = v[i] / (norm || 1);
    
      return embedding; // Float32Array length 512
    }

    // Generate embeddings for uncached prompts
    if (toRun.length > 0) {
      // const extractor = await getTextFeatureExtractor();
      for (const prompt of toRun) {
        try {
          const result = await  getTextEmbedding(prompt);
          const vector = result;
          console.log('vector:', vector)
          // const vector = (result && (result as any).data)
          //   ? new Float32Array((result as any).data)
          //   : new Float32Array(result as unknown as Float32Array);
          cache.set(prompt, vector);
        } catch (error) {
          console.error(`Failed to generate embedding for prompt: "${prompt}"`, error);
          cache.set(prompt, new Float32Array(384));
        }
      }
    }

    // Return all prompts with their corresponding vectors (from cache)
    return prompts.map(prompt => ({
      prompt,
      vector: cache.get(prompt)!
    }));
  }, [getClipTokenizer, getClipTextModel]);



  const handleImagesUploaded = (files: File[]) => {
    setUploadedFiles(files)
    console.log('Uploaded files:', files)
  }

  const handleAIImagesUploaded = (files: File[]) => {
    setUploadedAIFiles(files)
    console.log('Uploaded AI files:', files)
  }

  const analyizeImages = async (originalFiles: File[], aiFiles: File[]) => {

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
    setOverlays(results)

    // run image embedding on the results


    // run image embedding on the cropped results


    // run text similarity between the results and the text embeddings

    


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
        const clipTokenizerPromise = getClipTokenizer()
        const clipTextModelPromise = getClipTextModel()
        const refsPromise = loadReferenceImages()

        await Promise.all([
          imageExtractorPromise, 
          clipTokenizerPromise,
          clipTextModelPromise,
          refsPromise
        ])
        setLoadingProgress(1.0) // Complete loading
        setIsExtractorLoading(false)
      } catch (e) {
        console.error('Failed to initialize:', e)
        setIsExtractorLoading(false)
      }
    }
    initialize()
  }, [getImageFeatureExtractor, getClipTokenizer, getClipTextModel, loadReferenceImages])


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
          <LoadingComponent 
            progress={loadingProgress}
            message="Loading AI models and initializing..."
            size="medium"
            className="max-w-md"
          />
        ) : (
          <div className="text-sm text-green-700 font-medium">✓ AI models loaded successfully</div>
        )}
      </div>
      {overlays.length > 0 && (
        <div className="mt-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4">Face Detection Results</h2>
          <ImageWithLandmarks 
            results={overlays}
            maxWidth={640}
            showChips={true}
            className="mt-4"
          />
        </div>
      )}
      {/* reserved for future face detection UI */}
    </>
  )
}

export default App
