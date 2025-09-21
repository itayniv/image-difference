import React, { useState } from 'react';
import FaceOverlay from './FaceOverlay';
import Toggle from './Toggle';
import type { ImageAnalysisDataset, ImageData } from '../imageAnalysisTypes';



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
  imageDataset?: ImageAnalysisDataset;
}

/**
 * Helper component that renders images with landmark overlays and their corresponding face chips
 * @param results - Array of processed image results from face detection
 * @param maxWidth - Maximum width for the original image display (default: 640)
 * @param showChips - Whether to show the cropped face chips (default: true)
 * @param className - Additional CSS classes for the container
 * @param imageDataset - Optional image analysis dataset to display processing information
 */
export const ImageWithLandmarks: React.FC<ImageWithLandmarksProps> = ({
  results,
  maxWidth: _maxWidth = 640,
  showChips = true,
  className = '',
  imageDataset
}) => {
  const [showOverlays, setShowOverlays] = useState(true);

  // Helper function to find corresponding ImageData from dataset
  const findImageData = (result: ProcessedImageResult): ImageData | undefined => {
    if (!imageDataset) return undefined;
    return imageDataset.images.find(img => img.name === result.sourceName);
  };

  if (!results || results.length === 0) {
    return (
      <div className={`text-gray-500 text-left py-8 ${className}`}>
        No processed images to display
      </div>
    );
  }

  const [referenceImage, ...otherImages] = results;

  if (!referenceImage) {
    return (
      <div className={`text-gray-500 text-left py-8 ${className}`}>
        No processed images to display
      </div>
    );
  }

  // Component to render processing information
  const renderProcessingInfo = (result: ProcessedImageResult) => {
    const imageData = findImageData(result);
    if (!imageData) return null;

    return (
      <div className="mt-3 p-3 bg-gray-50 rounded-lg border">

        {/* Similarity to Source */}
        {imageData.computeSimilarityToSource?.similarities && imageData.computeSimilarityToSource.similarities.length > 0 && (
          <div className="mb-2">
            <h6 className="text-xs font-medium text-gray-700 mb-1">Source Similarity:</h6>
            <div className="space-y-1">
              {imageData.computeSimilarityToSource.similarities.map((sim, idx) => (
                <div key={idx} className="text-xs bg-white p-2 rounded border">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Face {sim.faceIndex + 1} → Source Face {sim.sourceFaceIndex + 1}</span>
                    <span className="font-medium text-green-600">{(sim.similarity * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* OpenAI Narrative Summary */}
        {imageData.openAiNarrative?.summary && (
          <div className="mb-2">
            <h6 className="text-xs font-medium text-gray-700 mb-1">AI Summary:</h6>
            <div className="text-xs bg-white p-2 rounded border">
              <p className="text-gray-700 text-left leading-relaxed">{imageData.openAiNarrative.summary}</p>
            </div>
          </div>
        )}

        {/* Text Similarity Best Matches */}
        {imageData.textSimilarity?.bestMatches && imageData.textSimilarity.bestMatches.length > 0 && (
          <div className="mb-2">
            <h6 className="text-xs font-medium text-gray-700 mb-1 -ml-3">Best Matches:</h6>
            <div className="space-y-1">
              {imageData.textSimilarity.bestMatches.map((match, idx) => (
                <div key={idx} className="text-xs bg-white p-2 rounded border">
                  <div className="flex justify-between items-center">
                    <span className="font-regular text-gray-700">{match.category}: <span className="text-gray-500 font-bold">{match.attribute}</span></span>
                    <span className="text-blue-600">{(match.similarity * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Text Similarity to Source */}
        {imageData.textSimilarityToSource?.categoryComparisons && imageData.textSimilarityToSource.categoryComparisons.length > 0 && (
          <div className="mb-2">
            <h6 className="text-xs font-medium text-gray-700 mb-1">Text Similarity to Source:</h6>
            <div className="mb-2 text-xs bg-gray-50 p-2 rounded border">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Overall Match</span>
                <span className={`font-medium ${imageData.textSimilarityToSource.summary.matchingPercentage > 70 ? 'text-green-600' :
                    imageData.textSimilarityToSource.summary.matchingPercentage > 50 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                  {imageData.textSimilarityToSource.summary.matchingCategories}/{imageData.textSimilarityToSource.summary.totalCategories}
                  ({imageData.textSimilarityToSource.summary.matchingPercentage.toFixed(1)}%)
                </span>
              </div>
            </div>
            <div className="space-y-1">
              {imageData.textSimilarityToSource.categoryComparisons.map((comp, idx) => (
                <div key={idx} className={`text-xs p-2 rounded border ${comp.isMatching ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
                  }`}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-medium text-gray-700 capitalize">{comp.category}</span>
                    <span className={`text-xs ${comp.isMatching ? 'text-green-600' : 'text-red-600'}`}>
                      {comp.isMatching ? '✓ Match' : '✗ Different'}
                    </span>
                  </div>
                  <div className="text-xs text-gray-600">
                    <div>Source: <span className="font-medium">{comp.sourceAttribute}</span> ({(comp.sourceSimilarity * 100).toFixed(1)}%)</div>
                    <div>Current: <span className="font-medium">{comp.currentAttribute}</span> ({(comp.currentSimilarity * 100).toFixed(1)}%)</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Processing Errors */}
        {imageData.processing.errors && imageData.processing.errors.length > 0 && (
          <div className="mb-2">
            <h6 className="text-xs font-medium text-red-700 mb-1">Errors:</h6>
            <div className="space-y-1">
              {imageData.processing.errors.map((error, idx) => (
                <div key={idx} className="text-xs bg-red-50 p-2 rounded border border-red-200">
                  <div className="font-medium text-red-700">{error.stage}</div>
                  <div className="text-red-600">{error.error}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Image source indicator */}
        <div className="text-xs -ml-3 flex flex-row items-center  p-2">
          <span className={`ml-1 px-2 py-0.5 rounded text-white text-xs ${imageData.source === 'original' ? 'bg-blue-500' :
              imageData.source === 'ai' ? 'bg-purple-500' : 'bg-green-500'
            }`}>
            {imageData.source === 'original' ? "TARGET" : "REFERENCE"}
          </span>
        </div>
      </div>
    );
  };

  const renderImageCard = (result: ProcessedImageResult, isReference = false) => (
    <div
      key={result.id}
      className={`border rounded-xl p-4 bg-white shadow-sm pb-8 ${isReference ? 'sticky top-4' : 'flex-grow min-w-0 max-w-full sm:max-w-[calc(50%-0.5rem)] lg:max-w-[calc(33.333%-0.75rem)]'
        }`}
      style={!isReference ? { flexBasis: 'calc(33.333% - 0.75rem)' } : undefined}
    >
      {/* Source image name */}
      <div className="mb-3">
        <h3 className={`font-semibold text-gray-800 truncate ${isReference ? 'text-base' : 'text-sm'}`} title={result.sourceName}>
          {isReference && <span className="text-blue-600 mr-2">📌</span>}
          {result.sourceName}
        </h3>
        <p className={`text-gray-600 ${isReference ? 'text-sm' : 'text-xs'}`}>
          {result.landmarksPerFace.length} face(s) detected
        </p>
      </div>

      {/* Original image with landmark overlay */}
      <div className="mb-4">
        <h4 className={`font-medium text-gray-700 mb-2 ${isReference ? 'text-sm' : 'text-xs'}`}>
          {isReference ? 'Reference Image with Landmarks' : 'Original with Landmarks'}
        </h4>
        <div className="w-full">
          <FaceOverlay
            imageURL={result.imageURL}
            imageSize={result.imageSize}
            landmarksPerFace={result.landmarksPerFace}
            maxWidth={isReference ? 400 : 250}
            showOverlays={showOverlays}
          />
        </div>
      </div>

      {/* Face chips/crops */}
      {showChips && result.chipURLs.length > 0 && (
        <div className="mt-3">
          <h4 className={`font-medium text-gray-700 mb-2 ${isReference ? 'text-sm' : 'text-xs'}`}>
            Extracted Faces ({result.chipURLs.length})
          </h4>
          <div className={`grid gap-2 ${isReference ? 'grid-cols-3' : 'grid-cols-2'}`}>
            {result.chipURLs.map((url, idx) => (
              <div key={`${result.id}-chip-${idx}`} className="flex flex-col items-center">
                <div className="relative group w-full">
                  <img
                    src={url}
                    alt={`Face ${idx + 1} from ${result.sourceName}`}
                    className="w-full h-auto rounded border border-gray-200 shadow-sm hover:shadow-md transition-shadow duration-200"
                  />
                </div>
                <span className="text-xs text-gray-500 mt-1 font-medium">
                  Face {idx + 1}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Processing Information */}
      {renderProcessingInfo(result)}
    </div>
  );

  return (
    <div className={className}>
      <div className="mb-4 flex items-center gap-2">
        <Toggle
          checked={showOverlays}
          onChange={setShowOverlays}
          aria-label="Show Face Landmarks"
        />
        <span className="text-sm font-medium text-gray-700">Show Face Landmarks</span>
      </div>

      <div className={`flex gap-6`}>
        {/* Reference image - 1/3 width, sticky */}
        <div className="w-1/3 flex-shrink-0">
          {renderImageCard(referenceImage, true)}
        </div>

        {/* Other images - 2/3 width, flex container */}
        <div className="w-2/3 flex flex-wrap gap-4">
          {otherImages.map((result) => renderImageCard(result, false))}
        </div>
      </div>
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
