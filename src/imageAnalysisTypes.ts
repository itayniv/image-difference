// Core types for image analysis state management

export interface ImageEmbedding {
  vector: Float32Array;
  model: string; // e.g., "clip-vit-base-patch32"
  timestamp: Date;
}

export interface FaceEmbedding {
  vector: Float32Array;
  faceIndex: number; // Which face from the image (0, 1, 2...)
  model: string;
  timestamp: Date;
}

export interface TextSimilarityResult {
  prompt: string;
  similarity: number;
  attribute?: string; // Optional attribute category
}

export interface FaceSimilarityResult {
  targetImageId: string;
  targetFaceIndex: number;
  similarity: number;
  comparisonType: 'face-to-face' | 'face-to-image';
}

export interface ImageSimilarityResult {
  targetImageId: string;
  similarity: number;
  comparisonType: 'image-to-image' | 'image-to-face';
}

export interface ProcessingStatus {
  faceDetection: 'pending' | 'processing' | 'completed' | 'error';
  imageEmbedding: 'pending' | 'processing' | 'completed' | 'error';
  faceEmbedding: 'pending' | 'processing' | 'completed' | 'error';
  textSimilarity: 'pending' | 'processing' | 'completed' | 'error';
  imageSimilarity: 'pending' | 'processing' | 'completed' | 'error';
  faceSimilarity: 'pending' | 'processing' | 'completed' | 'error';
}

export interface AnalyzedImage {
  // Core identification
  id: string;
  file: File;
  category: 'reference' | 'ai-generated';
  
  // Face detection results
  imageURL: string;
  imageSize: { width: number; height: number };
  landmarksPerFace: { x: number; y: number }[][];
  faceCrops: Array<{
    blob: Blob;
    url: string;
    faceIndex: number;
  }>;
  
  // Embeddings
  imageEmbedding?: ImageEmbedding;
  faceEmbeddings: FaceEmbedding[]; // One per detected face
  
  // Similarity results
  textSimilarities: TextSimilarityResult[];
  imageSimilarities: ImageSimilarityResult[];
  faceSimilarities: FaceSimilarityResult[];
  
  // Processing status
  status: ProcessingStatus;
  
  // Metadata
  createdAt: Date;
  lastUpdated: Date;
  errors?: Array<{
    step: keyof ProcessingStatus;
    message: string;
    timestamp: Date;
  }>;
}

// State management interface
export interface ImageAnalysisState {
  images: Map<string, AnalyzedImage>;
  
  // Global analysis results
  crossImageSimilarities: Array<{
    sourceId: string;
    targetId: string;
    imageToImageSimilarity?: number;
    faceToFaceSimilarities: Array<{
      sourceFaceIndex: number;
      targetFaceIndex: number;
      similarity: number;
    }>;
  }>;
  
  // Analysis configuration
  config: {
    models: {
      imageModel: string;
      textModel: string;
    };
    thresholds: {
      minFaceConfidence: number;
      minSimilarityThreshold: number;
    };
  };
  
  // Global state
  isAnalyzing: boolean;
  analysisProgress: number; // 0-1
  lastAnalysisTimestamp?: Date;
}

// Helper functions for state management
export interface ImageAnalysisActions {
  // Image management
  addImage: (file: File, category: 'reference' | 'ai-generated') => string; // Returns image ID
  removeImage: (id: string) => void;
  updateImageStatus: (id: string, step: keyof ProcessingStatus, status: ProcessingStatus[keyof ProcessingStatus]) => void;
  
  // Face detection results
  setFaceDetectionResults: (
    id: string, 
    results: {
      imageSize: { width: number; height: number };
      landmarksPerFace: { x: number; y: number }[][];
      faceCrops: Array<{ blob: Blob; faceIndex: number }>;
    }
  ) => void;
  
  // Embeddings
  setImageEmbedding: (id: string, embedding: ImageEmbedding) => void;
  addFaceEmbedding: (id: string, embedding: FaceEmbedding) => void;
  
  // Similarity results
  addTextSimilarity: (id: string, result: TextSimilarityResult) => void;
  addImageSimilarity: (id: string, result: ImageSimilarityResult) => void;
  addFaceSimilarity: (id: string, result: FaceSimilarityResult) => void;
  setCrossImageSimilarities: (similarities: ImageAnalysisState['crossImageSimilarities']) => void;
  
  // Error handling
  addError: (id: string, step: keyof ProcessingStatus, message: string) => void;
  
  // Bulk operations
  resetAnalysis: () => void;
  startAnalysis: () => void;
  completeAnalysis: () => void;
}

// Utility types for querying the state
export interface ImageQueryOptions {
  category?: 'reference' | 'ai-generated';
  hasImageEmbedding?: boolean;
  hasFaceEmbeddings?: boolean;
  minFaces?: number;
  maxFaces?: number;
  status?: Partial<ProcessingStatus>;
}

export interface SimilarityQueryOptions {
  minSimilarity?: number;
  maxSimilarity?: number;
  comparisonType?: 'face-to-face' | 'face-to-image' | 'image-to-image' | 'image-to-face';
  sourceCategory?: 'reference' | 'ai-generated';
  targetCategory?: 'reference' | 'ai-generated';
}