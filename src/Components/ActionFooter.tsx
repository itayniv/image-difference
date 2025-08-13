import React from 'react'
import LoadingComponent from './LoadingComponent'
import './ActionFooter.css'

interface ActionFooterProps {
  isAnalyzing: boolean
  onAnalyzeImages: () => void
  isAnalyzeDisabled: boolean
  isExtractorLoading: boolean
  loadingProgress: number
}

const ActionFooter: React.FC<ActionFooterProps> = ({
  isAnalyzing,
  onAnalyzeImages,
  isAnalyzeDisabled,
  isExtractorLoading,
  loadingProgress
}) => {
  return (
    <footer className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 shadow-lg z-40">
      <div className="mx-auto px-4 py-4">
        <div className="flex gap-4 justify-between items-center flex-wrap">
          {/* Loader section on the left */}
          <div className="flex-shrink-0">
            {isExtractorLoading && (
              <LoadingComponent
                progress={loadingProgress}
                message="Loading models and initializing..."
                size="small"
                className="w-72 shadow-lg bg-white"
              />
            )}

            {/* Success message when models are loaded */}
            {!isExtractorLoading && (
              <div className="bg-green-50 rounded-lg p-3 shadow-lg">
                <div className="text-xs text-green-700 font-regular">✓ Models loaded successfully</div>
              </div>
            )}
          </div>

          {/* Action buttons on the right */}
          <div className="flex gap-4 flex-wrap">
            <button
              type="button"
              onClick={onAnalyzeImages}
              className="px-4 py-2 rounded bg-gray-100 text-gray-900 hover:bg-gray-200 border border-gray-200 hover:shadow-md disabled:opacity-50 transition-colors flex items-center gap-2"
              disabled={isAnalyzeDisabled || isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <div className="loader"></div>
                  <span>Analyzing images</span>
                </>
              ) : (
                'Analyze Images'
              )}
            </button>

            {/* <button
              type="button"
              onClick={onLogDatasetSummary}
              className="px-4 py-2 rounded bg-green-600 text-white hover:bg-green-700 disabled:opacity-50 transition-colors"
              disabled={isLogDisabled}
            >
              Log Dataset Summary
            </button> */}
  {/* 
            <button
              type="button"
              onClick={onComputeTextSimilarity}
              className="px-4 py-2 rounded bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50 transition-colors"
              disabled={isTextSimilarityDisabled}
            >
              Compute Text Similarity
            </button> */}

            {/* <button
              type="button"
              onClick={onComputeSimilarityToSource}
              className="px-4 py-2 rounded bg-orange-600 text-white hover:bg-orange-700 disabled:opacity-50 transition-colors"
              disabled={isSimilarityToSourceDisabled}
            >
              Compute Similarity to Source
            </button> */}
          </div>
        </div>
      </div>
    </footer>
  )
}

export default ActionFooter
