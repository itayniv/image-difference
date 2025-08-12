import React, { useState, useCallback, useEffect } from 'react';

interface ImageDropZoneProps {
  onImagesUploaded?: (files: File[]) => void;
  maxFiles?: number;
  acceptedTypes?: string[];
  files?: File[]; // External files to display
}

const ImageDropZone: React.FC<ImageDropZoneProps> = ({
  onImagesUploaded,
  maxFiles = 5,
  acceptedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
  files = []
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadedImages, setUploadedImages] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);

  // Sync with external files prop and generate previews
  useEffect(() => {
    if (files.length > 0) {
      setUploadedImages(files);
      
      // Generate previews for external files
      const generatePreviews = async () => {
        const newPreviews: string[] = [];
        for (const file of files) {
          try {
            const url = URL.createObjectURL(file);
            newPreviews.push(url);
          } catch (error) {
            console.error('Failed to create preview for', file.name, error);
          }
        }
        setPreviews(newPreviews);
      };
      
      generatePreviews();
    } else if (uploadedImages.length > 0 && files.length === 0) {
      // Clear if external files are removed
      setUploadedImages([]);
      setPreviews([]);
    }

    // Cleanup URLs when component unmounts or files change
    return () => {
      if (files.length > 0) {
        previews.forEach(url => {
          try {
            URL.revokeObjectURL(url);
          } catch (e) {
            // URL might already be revoked
          }
        });
      }
    };
  }, [files]); // Only depend on files prop

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
    <div className="w-full max-w-4xl mx-auto">
      <div
        className={`
          border-2 border-dashed rounded-xl p-8 md:p-12 text-center transition-all duration-300 cursor-pointer relative min-h-[150px] md:min-h-[200px] flex items-center justify-center
          ${isDragOver 
            ? 'border-green-400 bg-green-50 shadow-lg shadow-green-200' 
            : 'border-gray-300 bg-slate-50 hover:border-indigo-500 hover:bg-blue-50 hover:-translate-y-0.5 hover:shadow-lg hover:shadow-indigo-200'
          }
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {uploadedImages.length === 0 ? (
          <div className="pointer-events-none">
            <div className="text-4xl md:text-5xl mb-4 opacity-70">📁</div>
            <h3 className="text-md md:text-lg font-semibold text-gray-700 mb-2">Drag & Drop Images Here</h3>
            <p className="text-gray-500 mb-4">or click to browse files</p>
            <input
              type="file"
              multiple
              accept={acceptedTypes.join(',')}
              onChange={handleFileInput}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer pointer-events-auto"
            />
            <div className="text-gray-400 text-sm leading-relaxed">
              <small>
                Supported formats: JPEG, PNG, GIF, WebP<br/>
              </small>
            </div>
          </div>
        ) : (
          <div className="w-full">
            <div className="flex flex-col md:flex-row justify-between items-stretch md:items-center gap-3 md:gap-2 mb-4 pb-3 border-b border-gray-200">
              <h4 className="text-lg font-semibold text-gray-700">Uploaded Images ({uploadedImages.length})</h4>
              <button 
                onClick={clearAll} 
                className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-md text-xs font-medium transition-colors duration-200"
              >
                Clear All
              </button>
            </div>
            <div className="relative h-[180px] w-full mx-auto">
              {uploadedImages.map((file, index) => {
                const preview = previews[index];
                if (!preview) return null; // guard against async mismatch
                
                // Calculate stacking positions for fan effect
                const offsets = [-48, -24, 0, 24, 48, 72, 96, 120];
                const rotations = [-4, 2, -1, 3, -2, 1, -3, 2];
                const offset = offsets[index] || 0;
                const rotation = rotations[index] || 0;
                const zIndex = index + 1;
                
                return (
                  <div 
                    key={`${file.name}-${index}`} 
                    className="absolute left-1/2 w-[150px] md:w-[200px] rounded-lg overflow-hidden bg-gray-50 shadow-md transition-all duration-200 hover:shadow-xl hover:-translate-y-1 hover:scale-105 group"
                    style={{
                      transform: `translateX(calc(-50% + ${offset}px)) rotate(${rotation}deg)`,
                      zIndex: zIndex
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.zIndex = '20';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.zIndex = zIndex.toString();
                    }}
                  >
                    <img 
                      src={preview} 
                      alt={`Preview ${index + 1}`} 
                      className="w-full h-[120px] md:h-[150px] object-cover block"
                    />
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent text-white p-4 pb-3 flex justify-between items-end opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                      <span className="text-xs font-medium max-w-[calc(100%-2rem)] overflow-hidden text-ellipsis whitespace-nowrap">
                        {file.name}
                      </span>
                      <button
                        onClick={() => removeImage(index)}
                        className="bg-red-500 hover:bg-red-600 text-white border-none w-6 h-6 rounded-full cursor-pointer text-sm font-bold flex items-center justify-center transition-colors duration-200 flex-shrink-0 leading-none"
                      >
                        ×
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};


export default ImageDropZone; 