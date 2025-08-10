import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import ImageDropZone from './Components/ImageDropZone'
import { pipeline } from '@huggingface/transformers'
import { compareOriginalToAI, extractVectorsFromFiles } from './Components/Helpers';
import { UMAP } from 'umap-js';


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
 


function App() {
  const [uploadedAIFiles, setUploadedAIFiles] = useState<File[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  // reserved for future face detection UI


  // Cache the image feature extractor so the model loads only once
  const imageFeatureExtractorRef = useRef<any | null>(null)

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
        { url: img9, name: 'IMG_9162 Medium.jpeg' },
        { url: img1, name: 'IMG_7812.jpeg' },
        { url: img2, name: 'IMG_8236.jpeg' },
        { url: img4, name: 'IMG_9157 Medium.jpeg' },
        { url: img3, name: 'IMG_8910 Medium.jpeg' },
        { url: img5, name: 'IMG_9158 Medium.jpeg' },
        { url: img6, name: 'IMG_9159 Medium.jpeg' },
        { url: img7, name: 'IMG_9160 Medium.jpeg' },
        { url: img8, name: 'IMG_9161 Medium.jpeg' }
      ]

      const files = await Promise.all(
        imageUrls.map(({ url, name }) => urlToFile(url, name))
      )

      // Split files between the two drop zones (first half to original, second half to AI)
     
      const originalFiles = files.slice(0, 1)
      const aiFiles = files.slice(1)

      setUploadedFiles(originalFiles)
      setUploadedAIFiles(aiFiles)
      
      console.log('Loaded reference images:', { originalFiles, aiFiles })
    } catch (error) {
      console.error('Failed to load reference images:', error)
    }
  }, [urlToFile])

  const getImageFeatureExtractor = useCallback(async () => {
    if (!imageFeatureExtractorRef.current) {
      imageFeatureExtractorRef.current = await pipeline(
        'image-feature-extraction',
        'Xenova/vit-base-patch16-224-in21k'
      )
    }
    return imageFeatureExtractorRef.current
  }, [])



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
    console.log('Analyze similarity invoked with files:', combined)
    // Compute cosine similarity between original and AI images (pairwise by index)
    const extractor = await getImageFeatureExtractor()
    const extractedVectors = await extractVectorsFromFiles(combined, extractor)

    const results = await compareOriginalToAI(originalFiles, aiFiles, extractor)

    const vectorMatrix = extractedVectors.map(v => v.vector)

         // TODO: add UMAP to visualize the similarity
     const umap = new UMAP({ 
       nNeighbors: Math.min(5, vectorMatrix.length - 1), 
       minDist: 0.1, 
       nComponents: 2 
     });
     const embedding = umap.fit(vectorMatrix);
     console.log('Embedded umap:', embedding);
  }

  useEffect(() => {
    // Warm up the extractor and load reference images on mount
    const initialize = async () => {
      try {
        await Promise.all([
          getImageFeatureExtractor(),
          loadReferenceImages()
        ])
      } catch (e) {
        console.error('Failed to initialize:', e)
      }
    }
    initialize()
  }, [getImageFeatureExtractor, loadReferenceImages])


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

{/* reserved for future face detection UI */}
    </>
  )
}

export default App
