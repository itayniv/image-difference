// Helper functions for image similarity analysis

export interface VectorResult {
  fileName: string;
  vector: number[];
}

export interface SimilarityResult {
  original: string;
  ai: string;
  similarity: number;
}

// Get a single normalized embedding vector for each image file
export const extractVectorsFromFiles = async (
  files: File[], 
  extractor: any
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
  extractor: any
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
