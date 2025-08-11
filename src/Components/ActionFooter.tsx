import React from 'react'

interface ActionFooterProps {
  onAnalyzeImages: () => void
  onLogDatasetSummary: () => void
  onComputeTextSimilarity: () => void
  onComputeSimilarityToSource: () => void
  isAnalyzeDisabled: boolean
  isLogDisabled: boolean
  isTextSimilarityDisabled: boolean
  isSimilarityToSourceDisabled: boolean
}

const ActionFooter: React.FC<ActionFooterProps> = ({
  onAnalyzeImages,
  onLogDatasetSummary,
  onComputeTextSimilarity,
  onComputeSimilarityToSource,
  isAnalyzeDisabled,
  isLogDisabled,
  isTextSimilarityDisabled,
  isSimilarityToSourceDisabled
}) => {
  return (
    <footer className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 shadow-lg z-40">
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex gap-4 justify-center flex-wrap">
          <button
            type="button"
            onClick={onAnalyzeImages}
            className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
            disabled={isAnalyzeDisabled}
          >
            Compute face landmarks and embeddings
          </button>

          {/* <button
            type="button"
            onClick={onLogDatasetSummary}
            className="px-4 py-2 rounded bg-green-600 text-white hover:bg-green-700 disabled:opacity-50 transition-colors"
            disabled={isLogDisabled}
          >
            Log Dataset Summary
          </button> */}

          <button
            type="button"
            onClick={onComputeTextSimilarity}
            className="px-4 py-2 rounded bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50 transition-colors"
            disabled={isTextSimilarityDisabled}
          >
            Compute Text Similarity
          </button>

          <button
            type="button"
            onClick={onComputeSimilarityToSource}
            className="px-4 py-2 rounded bg-orange-600 text-white hover:bg-orange-700 disabled:opacity-50 transition-colors"
            disabled={isSimilarityToSourceDisabled}
          >
            Compute Similarity to Source
          </button>
        </div>
      </div>
    </footer>
  )
}

export default ActionFooter
