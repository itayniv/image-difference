import React, { useState, useCallback } from 'react';
import './ImageDropZone.css';

interface ImageDropZoneProps {
  onImagesUploaded?: (files: File[]) => void;
  maxFiles?: number;
  acceptedTypes?: string[];
}

const ImageDropZone: React.FC<ImageDropZoneProps> = ({
  onImagesUploaded,
  maxFiles = 5,
  acceptedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadedImages, setUploadedImages] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const processFiles = useCallback((files: FileList) => {
    const imageFiles = Array.from(files).filter(file => 
      acceptedTypes.includes(file.type)
    );

    if (imageFiles.length === 0) {
      alert('Please upload valid image files (JPEG, PNG, GIF, WebP)');
      return;
    }

    const newFiles = imageFiles.slice(0, maxFiles - uploadedImages.length);
    
    if (newFiles.length < imageFiles.length) {
      alert(`Only ${maxFiles} files are allowed. ${newFiles.length} files will be uploaded.`);
    }

    const newPreviews: string[] = [];
    newFiles.forEach(file => {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (e.target?.result) {
          newPreviews.push(e.target.result as string);
          if (newPreviews.length === newFiles.length) {
            setPreviews(prev => [...prev, ...newPreviews]);
          }
        }
      };
      reader.readAsDataURL(file);
    });

    const updatedFiles = [...uploadedImages, ...newFiles];
    setUploadedImages(updatedFiles);
    onImagesUploaded?.(updatedFiles);
  }, [acceptedTypes, maxFiles, uploadedImages, onImagesUploaded]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    processFiles(files);
  }, [processFiles]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      processFiles(e.target.files);
    }
  }, [processFiles]);

  const removeImage = useCallback((index: number) => {
    const newFiles = uploadedImages.filter((_, i) => i !== index);
    const newPreviews = previews.filter((_, i) => i !== index);
    
    setUploadedImages(newFiles);
    setPreviews(newPreviews);
    onImagesUploaded?.(newFiles);
  }, [uploadedImages, previews, onImagesUploaded]);

  const clearAll = useCallback(() => {
    setUploadedImages([]);
    setPreviews([]);
    onImagesUploaded?.([]);
  }, [onImagesUploaded]);

  return (
    <div className="image-drop-zone-container">
      <div
        className={`image-drop-zone ${isDragOver ? 'drag-over' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="drop-zone-content">
          <div className="upload-icon">📁</div>
          <h3>Drag & Drop Images Here</h3>
          <p>or click to browse files</p>
          <input
            type="file"
            multiple
            accept={acceptedTypes.join(',')}
            onChange={handleFileInput}
            className="file-input"
          />
          <div className="file-info">
            <small>
              Supported formats: JPEG, PNG, GIF, WebP<br/>
              Maximum {maxFiles} files
            </small>
          </div>
        </div>
      </div>

      {uploadedImages.length > 0 && (
        <div className="uploaded-images">
          <div className="images-header">
            <h4>Uploaded Images ({uploadedImages.length})</h4>
            <button onClick={clearAll} className="clear-all-btn">
              Clear All
            </button>
          </div>
          <div className="image-previews">
            {previews.map((preview, index) => (
              <div key={index} className="image-preview">
                <img src={preview} alt={`Preview ${index + 1}`} />
                <div className="image-overlay">
                  <span className="image-name">{uploadedImages[index].name}</span>
                  <button
                    onClick={() => removeImage(index)}
                    className="remove-btn"
                  >
                    ×
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageDropZone; 