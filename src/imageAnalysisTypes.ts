// Core types for image analysis and processing pipeline

export interface ImageData {
  // Basic image information
  id: string;
  file: File;
  name: string;
  url: string; // Object URL for display
  size: {
    width: number;
    height: number;
    fileSize: number;
  };
  
  // Source classification
  source: 'original' | 'ai' | 'reference';
  
  // Face detection results
  faceDetection?: {
    landmarks: Array<{
      faceIndex: number;
      points: { x: number; y: number; z?: number }[];
      blendshapes?: any; // MediaPipe blendshapes
      matrices?: any; // MediaPipe transformation matrices
    }>;
    chips: Array<{
      faceIndex: number;
      url: string; // Object URL for the cropped face
      blob: Blob;
      canvas: HTMLCanvasElement;
      targetEyes: { 
        left: { x: number; y: number }; 
        right: { x: number; y: number }; 
      };
      transform: { 
        scale: number; 
        theta: number; 
        tx: number; 
        ty: number; 
      };
    }>;
    imageSize: { width: number; height: number };
    processingOptions?: {
      outputSize: number;
      paddingFactor: number;
    };
  };
  
  // Image embeddings
  embeddings?: {
    // Full image embedding
    fullImage?: {
      vector: Float32Array | number[];
      model: string; // e.g., "Xenova/clip-vit-base-patch32"
      dimensions: number;
      timestamp: Date;
    };
    
    // Face crop embeddings (one per detected face)
    faces?: Array<{
      faceIndex: number;
      vector: Float32Array | number[];
      model: string;
      dimensions: number;
      timestamp: Date;
    }>;
  };
  
  // Text similarity analysis
  textSimilarity?: {
    // Similarity scores against predefined text prompts/attributes
    attributes: Array<{
      category: string; // e.g., "age", "gender", "expression"
      attribute: string; // e.g., "young", "male", "smiling"
      prompt: string; // the actual text prompt used
      similarity: number; // cosine similarity score
      faceIndex?: number; // if calculated per face
    }>;
    
    // Best matching attributes per category
    bestMatches: Array<{
      category: string;
      attribute: string;
      similarity: number;
      faceIndex?: number;
    }>;
  };
  
  // Comparison results with other images
  comparisons?: {
    similarities: Array<{
      targetImageId: string;
      targetImageName: string;
      similarity: number;
      comparisonType: 'fullImage' | 'face';
      faceIndex?: number; // if comparing faces
      targetFaceIndex?: number;
      timestamp: Date;
    }>;
    
    // UMAP or other dimensionality reduction results
    dimensionalityReduction?: {
      coordinates: { x: number; y: number }[];
      method: 'umap' | 'tsne' | 'pca';
      parameters: Record<string, any>;
    };
  };
  
  // Similarity to source (first image's cropped face)
  computeSimilarityToSource?: {
    similarities: Array<{
      faceIndex: number; // which face in this image
      sourceImageId: string; // ID of the source (first) image
      sourceFaceIndex: number; // which face in the source image (typically 0)
      similarity: number; // cosine similarity score
      timestamp: Date;
    }>;
  };
  
  // Text similarity comparison to source image
  textSimilarityToSource?: {
    // Comparison of bestMatches between this image and source image
    categoryComparisons: Array<{
      category: string; // e.g., "age", "gender", "expression"
      sourceAttribute: string; // best match attribute from source image
      currentAttribute: string; // best match attribute from current image
      sourceSimilarity: number; // similarity score from source image
      currentSimilarity: number; // similarity score from current image
      isMatching: boolean; // true if attributes are the same
      faceIndex?: number; // if calculated per face
      sourceFaceIndex?: number; // which face in the source image
    }>;
    
    // Summary of matching/non-matching categories
    summary: {
      totalCategories: number;
      matchingCategories: number;
      nonMatchingCategories: number;
      matchingPercentage: number;
      timestamp: Date;
    };
  };
  
  // OpenAI-generated narrative summarizing similarities/differences to source
  openAiNarrative?: {
    summary: string; // short paragraph from OpenAI
    model?: string; // model used
    promptPreview?: string; // optional for debugging
    timestamp: Date;
    error?: string; // if generation failed
  };
  
  // Processing metadata
  processing: {
    stages: {
      uploaded: Date;
      faceDetectionCompleted?: Date;
      embeddingCompleted?: Date;
      textSimilarityCompleted?: Date;
      comparisonCompleted?: Date;
      similarityToSourceCompleted?: Date;
      textSimilarityToSourceCompleted?: Date;
      openAiNarrativeCompleted?: Date;
    };
    errors?: Array<{
      stage: string;
      error: string;
      timestamp: Date;
    }>;
  };
}

// Type for the complete dataset being analyzed
export interface ImageAnalysisDataset {
  images: ImageData[];
  metadata: {
    created: Date;
    lastUpdated: Date;
    totalImages: number;
    originalImages: number;
    aiImages: number;
    referenceImages: number;
  };
  
  // Global analysis results
  globalAnalysis?: {
    // Overall similarity matrix
    similarityMatrix?: number[][];
    
    // Clustering results
    clusters?: Array<{
      id: string;
      imageIds: string[];
      centroid?: number[];
      averageSimilarity: number;
    }>;
    
    // Summary statistics
    statistics?: {
      averageSimilarity: number;
      similarityRange: { min: number; max: number };
      mostSimilarPair: { image1Id: string; image2Id: string; similarity: number };
      leastSimilarPair: { image1Id: string; image2Id: string; similarity: number };
    };
  };
}

// Helper types for specific operations
export interface ProcessingOptions {
  faceDetection?: {
    outputSize?: number;
    paddingFactor?: number;
    maxFaces?: number;
    enableBlendshapes?: boolean;
    enableMatrices?: boolean;
  };
  
  embeddings?: {
    model?: string;
    pooling?: 'mean' | 'max' | 'cls';
    normalize?: boolean;
  };
  
  textSimilarity?: {
    attributes?: string[];
    threshold?: number;
  };
}
