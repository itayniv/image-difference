// Helper functions for image similarity analysis
import type { AttributeSpec } from './Attributes';
import type { ProcessResult } from './faceLandmarker';

export interface VectorResult {
  fileName: string;
  vector: number[];
}

export interface SimilarityResult {
  original: string;
  ai: string;
  similarity: number;
}

export interface EmbeddingExtractor {
  (file: File, options: { pooling: 'mean'; normalize: boolean }): Promise<{
    data: Float32Array | number[] | ArrayBuffer;
  } | Float32Array | number[] | ArrayBuffer>;
}

// Get a single normalized embedding vector for each image file
export const extractVectorsFromFiles = async (
  files: File[], 
  extractor: EmbeddingExtractor
): Promise<VectorResult[]> => {
  const results: VectorResult[] = [];
  for (const file of files) {
    try {
      const out = await extractor(file, { pooling: 'mean', normalize: true });
      const data = (out && (out as any).data) ? (out as any).data : out;
      const vector = Array.from(data as Iterable<number>);
      results.push({ fileName: file.name, vector });
    } catch (error) {
      console.error('Failed to extract embedding for', file.name, error);
    }
  }
  console.log('Extracted vectors:', results);
  return results;
};

// Convert face detection results (head shot crops) to File[] for use with extractVectorsFromFiles
export const convertChipsToFiles = async (
  results: Array<{
    id: string;
    imageURL: string;
    imageSize: { width: number; height: number };
    landmarksPerFace: { x: number; y: number }[][];
    chipURLs: string[];
    sourceName: string;
  }>,
  options: {
    includeAllFaces?: boolean; // If true, includes all detected faces; if false, only the first face per image
  } = {}
): Promise<File[]> => {
  const { includeAllFaces = false } = options;
  
  const filePromises: Promise<File>[] = [];
  
  results.forEach((result) => {
    // Determine which chip URLs to process
    const urlsToProcess = includeAllFaces 
      ? result.chipURLs 
      : result.chipURLs.slice(0, 1);
    
    urlsToProcess.forEach((url, faceIndex) => {
      const fileName = includeAllFaces 
        ? `${result.sourceName}_face_${faceIndex}.png`
        : `${result.sourceName}_face.png`;
      
      const filePromise = fetch(url)
        .then(response => response.blob())
        .then(blob => new File([blob], fileName, { type: blob.type }));
      
      filePromises.push(filePromise);
    });
  });
  
  return Promise.all(filePromises);
};

// Alternative version that works directly with ProcessResult objects
export const convertProcessResultsToFiles = (
  processResults: ProcessResult[],
  options: {
    includeAllFaces?: boolean; // If true, includes all detected faces; if false, only the first face per image
  } = {}
): File[] => {
  const { includeAllFaces = false } = options;
  
  const files: File[] = [];
  
  processResults.forEach((result) => {
    const facesToProcess = includeAllFaces 
      ? result.faces 
      : result.faces.slice(0, 1);
    
    facesToProcess.forEach((face, faceIndex) => {
      const fileName = includeAllFaces 
        ? `${result.file.name}_face_${faceIndex}.png`
        : `${result.file.name}_face.png`;
      
      const file = new File([face.chip.blob], fileName, { type: 'image/png' });
      files.push(file);
    });
  });
  
  return files;
};

// Compute cosine similarity between two vectors
export const cosineSimilarity = (a: number[], b: number[]): number => {
  if (a.length !== b.length) return NaN;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i];
    const bi = b[i];
    dot += ai * bi;
    normA += ai * ai;
    normB += bi * bi;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
};

// Compare original images to AI images and return similarity scores
export const compareOriginalToAI = async (
  original: File[], 
  ai: File[], 
  extractor: EmbeddingExtractor
): Promise<SimilarityResult[]> => {
  const [origVecs, aiVecs] = await Promise.all([
    extractVectorsFromFiles(original, extractor),
    extractVectorsFromFiles(ai, extractor),
  ]);
  const pairs = Math.min(origVecs.length, aiVecs.length);
  const results = Array.from({ length: pairs }, (_, i) => ({
    original: origVecs[i].fileName,
    ai: aiVecs[i].fileName,
    similarity: cosineSimilarity(origVecs[i].vector, aiVecs[i].vector),
  }));
  
  console.log('Cosine similarities:', results);

  return results;
};



export function compilePrompts(attrs: AttributeSpec[]): Record<string, string[]> {
    const byAttr: Record<string, string[]> = {};
    attrs.forEach(a => { byAttr[a.key] = a.options.map(opt => a.template(opt)); });
    return byAttr;
  }
