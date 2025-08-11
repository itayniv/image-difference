import React from 'react';
import FaceOverlay from './FaceOverlay';

export interface ProcessedImageResult {
  id: string;
  imageURL: string;
  imageSize: { width: number; height: number };
  landmarksPerFace: { x: number; y: number }[][];
  chipURLs: string[];
  sourceName: string;
}

export interface ImageWithLandmarksProps {
  results: ProcessedImageResult[];
  maxWidth?: number;
  showChips?: boolean;
  className?: string;
}

/**
 * Helper component that renders images with landmark overlays and their corresponding face chips
 * @param results - Array of processed image results from face detection
 * @param maxWidth - Maximum width for the original image display (default: 640)
 * @param showChips - Whether to show the cropped face chips (default: true)
 * @param className - Additional CSS classes for the container
 */
export const ImageWithLandmarks: React.FC<ImageWithLandmarksProps> = ({
  results,
  maxWidth = 640,
  showChips = true,
  className = ''
}) => {
  if (!results || results.length === 0) {
    return (
      <div className={`text-gray-500 text-center py-8 ${className}`}>
        No processed images to display
      </div>
    );
  }

  return (
    <div className={`space-y-8 ${className}`}>
      {results.map((result) => (
        <div key={result.id} className="border rounded-lg p-6 bg-white shadow-sm">
          {/* Source image name */}
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-gray-800">{result.sourceName}</h3>
            <p className="text-sm text-gray-600">
              {result.landmarksPerFace.length} face(s) detected
            </p>
          </div>

          {/* Original image with landmark overlay */}
          <div className="mb-6">
            <h4 className="text-md font-medium text-gray-700 mb-2">Original with Landmarks</h4>
            <FaceOverlay
              imageURL={result.imageURL}
              imageSize={result.imageSize}
              landmarksPerFace={result.landmarksPerFace}
              maxWidth={maxWidth}
            />
          </div>

          {/* Face chips/crops */}
          {showChips && result.chipURLs.length > 0 && (
            <div className="mt-4">
              <h4 className="text-md font-medium text-gray-700 mb-3">
                Extracted Faces ({result.chipURLs.length})
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
                {result.chipURLs.map((url, idx) => (
                  <div key={`${result.id}-chip-${idx}`} className="flex flex-col items-center">
                    <div className="relative group">
                      <img 
                        src={url} 
                        alt={`Face ${idx + 1} from ${result.sourceName}`}
                        className="w-full h-auto rounded-lg border-2 border-gray-200 shadow-sm hover:shadow-md transition-shadow duration-200"
                      />
                      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-10 rounded-lg transition-all duration-200"></div>
                    </div>
                    <span className="text-xs text-gray-500 mt-1 font-medium">
                      Face {idx + 1}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

/**
 * Simple helper function that creates the ProcessedImageResult format from face detection results
 * This can be used to convert the results from processFaceFile into the format expected by ImageWithLandmarks
 */
export const createProcessedImageResults = async (
  files: File[],
  faceDetectionResults: Array<{
    imageSize: { width: number; height: number };
    faces: Array<{
      landmarks: { x: number; y: number }[];
      chip: { blob: Blob };
    }>;
  }>
): Promise<ProcessedImageResult[]> => {
  return Promise.all(
    files.map(async (file, index) => {
      const result = faceDetectionResults[index];
      const imageURL = URL.createObjectURL(file);
      const chipURLs = result.faces.map((face) => URL.createObjectURL(face.chip.blob));
      
      return {
        id: `${file.name}-${file.size}-${file.lastModified}`,
        imageURL,
        imageSize: result.imageSize,
        landmarksPerFace: result.faces.map((face) => 
          face.landmarks.map((point) => ({ x: point.x, y: point.y }))
        ),
        chipURLs,
        sourceName: file.name,
      };
    })
  );
};

export default ImageWithLandmarks;
