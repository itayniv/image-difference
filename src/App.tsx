import { useState } from 'react'
import './App.css'
import ImageDropZone from './Components/ImageDropZone'
import FaceDetectorDemo from './Components/FaceDetectorDemo'

function App() {
  const [uploadedAIFiles, setUploadedAIFiles] = useState<File[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])


  const handleImagesUploaded = (files: File[]) => {
    setUploadedFiles(files)
    console.log('Uploaded files:', files)
  }

  const handleAIImagesUploaded = (files: File[]) => {
    setUploadedAIFiles(files)
    console.log('Uploaded AI files:', files)
  }

  const analyizeImages = (originalFiles: File[], aiFiles: File[]) => {
    // Pass images to your analysis pipeline here
    console.log('Analyize similarity invoked with:', {
      originalFiles,
      aiFiles,
    })
    // e.g., analyzeSimilarity(originalFiles, aiFiles)
  }

  return (
    <>      
    <div className="flex flex-row gap-4">
      
    <div className="card">
        <h2>Image Drop Zone</h2>
        <ImageDropZone 
          onImagesUploaded={handleImagesUploaded}
          maxFiles={10}
        />
      </div>
      <div className="card">
        <h2>AI Image Drop Zone</h2>
        <ImageDropZone 
          onImagesUploaded={handleAIImagesUploaded}
          maxFiles={10}
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

    <div className="mt-8">
      <FaceDetectorDemo />
    </div>
      
    </>
  )
}

export default App
