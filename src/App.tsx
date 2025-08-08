import { useState } from 'react'
import './App.css'
import ImageDropZone from './Components/ImageDropZone'

function App() {
  const [count, setCount] = useState(0)
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

  return (
    <>      
    <div className="flex flex-row gap-4">
      
    <div className="card">
        <h2>Image Drop Zone</h2>
        <ImageDropZone 
          onImagesUploaded={handleImagesUploaded}
          maxFiles={10}
        />
        {uploadedFiles.length > 0 && (
          <div style={{ marginTop: '1rem', textAlign: 'left' }}>
            <h3>Files ready for processing:</h3>
            <ul>
              {uploadedFiles.map((file, index) => (
                <li key={index}>
                  {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div className="card">
        <h2>AI Image Drop Zone</h2>
        <ImageDropZone 
          onImagesUploaded={handleAIImagesUploaded}
          maxFiles={10}
        />
        {uploadedAIFiles.length > 0 && (
          <div style={{ marginTop: '1rem', textAlign: 'left' }}>
            <h3>Files ready for processing:</h3>
            <ul>
              {uploadedAIFiles.map((file, index) => (
                <li key={index}>
                  {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
      
    </>
  )
}

export default App
