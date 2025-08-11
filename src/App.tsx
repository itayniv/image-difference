import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import ImageDropZone from './Components/ImageDropZone'
import { pipeline, env } from '@huggingface/transformers'
// import { compareOriginalToAI, extractVectorsFromFiles, convertChipsToFiles, convertProcessResultsToFiles } from './Components/Helpers';
// import { UMAP } from 'umap-js';
import { processFaceFile } from './Components/faceLandmarker'
import ImageWithLandmarks from './Components/ImageWithLandmarksHelper'
import { ATTRIBUTES } from './Components/Attributes'
import { compilePrompts } from './Components/Helpers'
import LoadingComponent from './Components/LoadingComponent'
import type { ImageData, ImageAnalysisDataset } from './imageAnalysisTypes'

import { AutoTokenizer, CLIPTextModelWithProjection } from '@huggingface/transformers'

// Temporarily disable browser cache to avoid using any corrupted ONNX files
env.useBrowserCache = false


const CLIP_MODEL_ID = "Xenova/clip-vit-base-patch32"; // image model


import img01 from './assets/ref_photos/ref_closeup_02.png'
import img02 from './assets/ref_photos/itay_cu3.jpeg'
// import img03 from './assets/ref_photos/itay_cu2.jpeg'


function App() {

  // Cache the image feature extractor so the model loads only once
  const imageFeatureExtractorRef = useRef<any | null>(null)

  // Cache CLIP tokenizer and text model for direct text embedding
  const clipTokenizerRef = useRef<any | null>(null)
  const clipTextModelRef = useRef<any | null>(null)

  const [uploadedAIFiles, setUploadedAIFiles] = useState<File[]>([])
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const textCacheRef = useRef<Map<string, Float32Array>>(new Map());
  const [textEmbeddings, setTextEmbeddings] = useState<Array<{ prompt: string, vector: Float32Array }>>([])

  // Main image analysis dataset
  const [imageDataset, setImageDataset] = useState<ImageAnalysisDataset>({
    images: [],
    metadata: {
      created: new Date(),
      lastUpdated: new Date(),
      totalImages: 0,
      originalImages: 0,
      aiImages: 0,
      referenceImages: 0
    }
  })

  // reserved for future face detection UI (keep for compatibility with existing components)
  const [overlays, setOverlays] = useState<
    Array<{
      id: string
      imageURL: string
      imageSize: { width: number; height: number }
      landmarksPerFace: { x: number; y: number }[][]
      chipURLs: string[]
      sourceName: string
    }>
  >([])
  const previousURLsRef = useRef<string[]>([])




  const [isExtractorLoading, setIsExtractorLoading] = useState<boolean>(true)
  const [loadingProgress, setLoadingProgress] = useState<number>(0)

  // Helper to convert imported image URLs to File objects
  const urlToFile = useCallback(async (url: string, filename: string): Promise<File> => {
    const response = await fetch(url)
    const blob = await response.blob()
    return new File([blob], filename, { type: blob.type })
  }, [])


  // Load reference images and populate drop zones
  const loadReferenceImages = useCallback(async () => {
    try {
      const imageUrls = [

        { url: img01, name: 'ref_closeup_02.png' },
        { url: img02, name: 'itay_cu3.jpeg' }
      ]

      const files = await Promise.all(
        imageUrls.map(({ url, name }) => urlToFile(url, name))
      )

      // Split files between the two drop zones (first half to original, second half to AI)

      const originalFiles = files.slice(0, 1)
      const aiFiles = files.slice(1)

      setUploadedFiles(originalFiles)
      setUploadedAIFiles(aiFiles)

    } catch (error) {
      console.error('Failed to load reference images:', error)
    }
  }, [urlToFile])

  const getImageFeatureExtractor = useCallback(async () => {
    if (!imageFeatureExtractorRef.current) {
      imageFeatureExtractorRef.current = await pipeline(
        'image-feature-extraction',
        // 'Xenova/vit-base-patch16-224-in21k'
        CLIP_MODEL_ID,
        {
          progress_callback: (status: any) => {
            try {
              if (status.progress !== undefined) {
                setLoadingProgress(status.progress);
              }
            } catch { }
          },
        }
      )
    }
    return imageFeatureExtractorRef.current
  }, [])

  useEffect(() => {
    const initialize = async () => {
      const prompts = compilePrompts(ATTRIBUTES);
      const textEmbeddings = await embedTexts(Object.values(prompts).flat());
      setTextEmbeddings(textEmbeddings)
    }
    initialize()
  }, [])


  // Monitor imageDataset changes
  useEffect(() => {
    console.log('Image Dataset Updated:', imageDataset)
    console.log('Total Images:', imageDataset.metadata.totalImages)
    console.log('Original Images:', imageDataset.metadata.originalImages)
    console.log('AI Images:', imageDataset.metadata.aiImages)

    // Example: Access individual image data
    if (imageDataset.images.length > 0) {
      console.log('First image data:', imageDataset.images[0])

      // Check if first image has face detection results
      if (imageDataset.images[0].faceDetection) {
        console.log('First image has', imageDataset.images[0].faceDetection.landmarks.length, 'faces detected')
      }

      // Check if first image has embeddings
      if (imageDataset.images[0].embeddings) {
        console.log('First image has embeddings:', {
          fullImage: !!imageDataset.images[0].embeddings.fullImage,
          faces: imageDataset.images[0].embeddings.faces?.length || 0
        })
      }
    }
  }, [imageDataset])

  const getClipTokenizer = useCallback(async () => {
    if (!clipTokenizerRef.current) {
      console.log('[Loading CLIP tokenizer...]');
      setLoadingProgress(prev => Math.max(prev, 0.4)); // Start tokenizer at 40%
      clipTokenizerRef.current = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch32');
      setLoadingProgress(prev => Math.max(prev, 0.6)); // Complete tokenizer at 60%
    }
    return clipTokenizerRef.current;
  }, []);

  const getClipTextModel = useCallback(async () => {
    if (!clipTextModelRef.current) {
      console.log('[Loading CLIP text model...]');
      setLoadingProgress(prev => Math.max(prev, 0.6)); // Start text model at 60%
      clipTextModelRef.current = await CLIPTextModelWithProjection.from_pretrained(
        'Xenova/clip-vit-base-patch32'
      );
      setLoadingProgress(prev => Math.max(prev, 0.9)); // Complete text model at 90%
    }
    return clipTextModelRef.current;
  }, []);



  const embedTexts = useCallback(async (prompts: string[]): Promise<Array<{ prompt: string, vector: Float32Array }>> => {
    const cache = textCacheRef.current;
    const toRun: string[] = [];
    const order: number[] = [];

    // Find prompts that aren't cached yet
    prompts.forEach((p, i) => {
      if (!cache.has(p)) {
        toRun.push(p);
        order.push(i);
      }
    });

    async function getTextEmbedding(text: string) {
      // 1) Get cached tokenizer + text model (loaded only once)
      const tokenizer = await getClipTokenizer();
      const textModel = await getClipTextModel();

      // 2) Tokenize
      const inputs = await tokenizer(text, {
        return_tensors: 'pt',
        padding: true,
        truncation: true,
      });

      // 3) Forward through TEXT model (no pixels needed)
      const { text_embeds } = await textModel(inputs); // shape: [1, 512]

      // 4) (Optional) L2-normalize for cosine similarity workflows
      const v = text_embeds.data as Float32Array;
      let norm = 0;
      for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
      norm = Math.sqrt(norm);
      const embedding = new Float32Array(v.length);
      for (let i = 0; i < v.length; i++) embedding[i] = v[i] / (norm || 1);

      return embedding; // Float32Array length 512
    }

    // Generate embeddings for uncached prompts
    if (toRun.length > 0) {
      // const extractor = await getTextFeatureExtractor();
      for (const prompt of toRun) {
        try {
          const result = await getTextEmbedding(prompt);
          const vector = result;
          cache.set(prompt, vector);
        } catch (error) {
          console.error(`Failed to generate embedding for prompt: "${prompt}"`, error);
          cache.set(prompt, new Float32Array(384));
        }
      }
    }

    // Return all prompts with their corresponding vectors (from cache)
    return prompts.map(prompt => ({
      prompt,
      vector: cache.get(prompt)!
    }));
  }, [getClipTokenizer, getClipTextModel]);



  const handleImagesUploaded = (files: File[]) => {
    setUploadedFiles(files)
    console.log('Uploaded files:', files)
  }

  const handleAIImagesUploaded = (files: File[]) => {
    setUploadedAIFiles(files)
    console.log('Uploaded AI files:', files)
  }

  const analyizeImages = async () => {
    // Initialize image data list from uploaded files
    const now = new Date()

    // Create initial ImageData objects for all uploaded files
    const initialImageDataList: ImageData[] = [
      ...uploadedFiles.map((file) => ({
        id: `original-${file.name}-${file.size}-${file.lastModified}`,
        file,
        name: file.name,
        url: URL.createObjectURL(file),
        size: {
          width: 0, // Will be updated after face detection
          height: 0, // Will be updated after face detection
          fileSize: file.size
        },
        source: 'original' as const,
        processing: {
          stages: {
            uploaded: now
          }
        }
      })),
      ...uploadedAIFiles.map((file) => ({
        id: `ai-${file.name}-${file.size}-${file.lastModified}`,
        file,
        name: file.name,
        url: URL.createObjectURL(file),
        size: {
          width: 0, // Will be updated after face detection
          height: 0, // Will be updated after face detection
          fileSize: file.size
        },
        source: 'ai' as const,
        processing: {
          stages: {
            uploaded: now
          }
        }
      }))
    ]

    // Update the dataset with initial data
    setImageDataset(prev => ({
      ...prev,
      images: initialImageDataList,
      metadata: {
        ...prev.metadata,
        lastUpdated: now,
        totalImages: initialImageDataList.length,
        originalImages: uploadedFiles.length,
        aiImages: uploadedAIFiles.length,
        referenceImages: 0
      }
    }))

    console.log('Initialized image data list:', initialImageDataList)

    // Process face detection for all images
    const faceDetectionResults = await Promise.all(
      initialImageDataList.map(async (imageData) => {
        try {
          const result = await processFaceFile(imageData.file, {
            outputSize: 512,    // Larger output size for better quality
            paddingFactor: 1.5  // 2x padding for much larger crops
          })

          // Create chip URLs and blobs
          const chips = result.faces.map((face, faceIndex) => ({
            faceIndex,
            url: URL.createObjectURL(face.chip.blob),
            blob: face.chip.blob,
            canvas: face.chip.canvas,
            targetEyes: face.chip.targetEyes,
            transform: face.chip.transform
          }))

          // Create landmarks data
          const landmarks = result.faces.map((face, faceIndex) => ({
            faceIndex,
            points: face.landmarks.map(p => ({ x: p.x, y: p.y, z: p.z })),
            blendshapes: face.blendshapes,
            matrices: face.matrices
          }))

          return {
            ...imageData,
            size: {
              ...imageData.size,
              width: result.imageSize.width,
              height: result.imageSize.height
            },
            faceDetection: {
              landmarks,
              chips,
              imageSize: result.imageSize,
              processingOptions: {
                outputSize: 512,
                paddingFactor: 1.5
              }
            },
            processing: {
              ...imageData.processing,
              stages: {
                ...imageData.processing.stages,
                faceDetectionCompleted: new Date()
              }
            }
          }
        } catch (error) {
          console.error(`Failed to process face detection for ${imageData.name}:`, error)
          return {
            ...imageData,
            processing: {
              ...imageData.processing,
              errors: [{
                stage: 'faceDetection',
                error: error instanceof Error ? error.message : String(error),
                timestamp: new Date()
              }]
            }
          }
        }
      })
    )

    // Update dataset with face detection results
    setImageDataset(prev => ({
      ...prev,
      images: faceDetectionResults,
      metadata: {
        ...prev.metadata,
        lastUpdated: new Date()
      }
    }))

    // Create overlay data for backward compatibility with existing components
    const overlayResults = faceDetectionResults
      .filter(img => img.faceDetection)
      .map(img => ({
        id: img.id,
        imageURL: img.url,
        imageSize: img.faceDetection!.imageSize,
        landmarksPerFace: img.faceDetection!.landmarks.map(lm =>
          lm.points.map(p => ({ x: p.x, y: p.y }))
        ),
        chipURLs: img.faceDetection!.chips.map(chip => chip.url),
        sourceName: img.name,
      }))


    // Revoke previous URLs to avoid leaks
    previousURLsRef.current.forEach((u) => URL.revokeObjectURL(u))
    previousURLsRef.current = []

    const allURLs: string[] = []
    overlayResults.forEach((r) => {
      allURLs.push(r.imageURL, ...r.chipURLs)
    })
    previousURLsRef.current = allURLs

    // Set overlays for backward compatibility
    setOverlays(overlayResults)

    // Convert head shot crops (chips) to File[] format for vector extraction
    // const headShotFiles = await convertChipsToFiles(results, { includeAllFaces: true })

    const extractor = await getImageFeatureExtractor()
    const embeddingResults = await Promise.all(
      faceDetectionResults.map(async (imageData) => {
        if (!imageData.faceDetection) return imageData

        try {
          // Generate embeddings for all face chips
          const faceEmbeddings = await Promise.all(
            imageData.faceDetection.chips.map(async (chip, faceIndex) => {
              const response = await fetch(chip.url)
              const blob = await response.blob()
              const file = new File([blob], `${imageData.name}_face_${faceIndex}.png`, { type: blob.type })

              const embedding = await extractor(file, { pooling: 'mean', normalize: true })
              const vector = Array.from(embedding.data as Float32Array)

              return {
                faceIndex,
                vector,
                model: "Xenova/clip-vit-base-patch32",
                dimensions: vector.length,
                timestamp: new Date()
              }
            })
          )

          // Also generate full image embedding
          const fullImageEmbedding = await extractor(imageData.file, { pooling: 'mean', normalize: true })
          const fullImageVector = Array.from(fullImageEmbedding.data as Float32Array)

          return {
            ...imageData,
            embeddings: {
              fullImage: {
                vector: fullImageVector,
                model: "Xenova/clip-vit-base-patch32",
                dimensions: fullImageVector.length,
                timestamp: new Date()
              },
              faces: faceEmbeddings
            },
            processing: {
              ...imageData.processing,
              stages: {
                ...imageData.processing.stages,
                embeddingCompleted: new Date()
              }
            }
          }
        } catch (error) {
          console.error(`Failed to generate embeddings for ${imageData.name}:`, error)
          return {
            ...imageData,
            processing: {
              ...imageData.processing,
              errors: [
                ...(imageData.processing.errors || []),
                {
                  stage: 'embedding',
                  error: error instanceof Error ? error.message : String(error),
                  timestamp: new Date()
                }
              ]
            }
          }
        }
      })
    )

    // Update dataset with embedding results
    setImageDataset(prev => ({
      ...prev,
      images: embeddingResults,
      metadata: {
        ...prev.metadata,
        lastUpdated: new Date()
      }
    }))

    console.log('Image analysis complete:', embeddingResults)

    // Example: Compare first face embeddings between images (if available)
    const imagesWithFaces = embeddingResults.filter(img =>
      img.embeddings?.faces && img.embeddings.faces.length > 0
    )

    if (imagesWithFaces.length >= 2) {
      const firstImage = imagesWithFaces[0]
      const secondImage = imagesWithFaces[1]

      if (firstImage.embeddings?.faces?.[0] && secondImage.embeddings?.faces?.[0]) {
        const similarity = cosineSimilarity(
          firstImage.embeddings.faces[0].vector as number[],
          secondImage.embeddings.faces[0].vector as number[]
        )
        console.log(`Face similarity between ${firstImage.name} and ${secondImage.name}:`, similarity)
      }
    }

    // Run text similarity analysis if text embeddings are available
    if (textEmbeddings.length > 0) {
      console.log('Running text similarity analysis...')
      await computeTextSimilarityForAllFaces()
    } else {
      console.log('Text embeddings not ready, skipping text similarity analysis')
    }

    // Compute similarity to source (first image's face)
    console.log('Computing similarity to source image...')
    computeSimilarityToSource()

    // TODO: Add UMAP visualization
    // TODO: Add comparison matrix generation
  }

  useEffect(() => {
    // Warm up the extractors and load reference images on mount
    const initialize = async () => {
      try {
        const imageExtractorPromise = getImageFeatureExtractor()
        const clipTokenizerPromise = getClipTokenizer()
        const clipTextModelPromise = getClipTextModel()
        const refsPromise = loadReferenceImages()

        await Promise.all([
          imageExtractorPromise,
          clipTokenizerPromise,
          clipTextModelPromise,
          refsPromise
        ])
        setLoadingProgress(1.0) // Complete loading
        setIsExtractorLoading(false)
      } catch (e) {
        console.error('Failed to initialize:', e)
        setIsExtractorLoading(false)
      }
    }
    initialize()
  }, [getImageFeatureExtractor, getClipTokenizer, getClipTextModel, loadReferenceImages])

  // Helper function for cosine similarity (moved from Helpers.ts for direct use)
  const cosineSimilarity = (a: number[], b: number[]): number => {
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

  // Function to compute text similarity for all face embeddings
  const computeTextSimilarityForAllFaces = useCallback(async () => {
    if (textEmbeddings.length === 0) {
      console.warn('No text embeddings available. Please wait for text embeddings to load.');
      return;
    }

    // Get current images from state at the time of execution to avoid dependency loop
    setImageDataset(currentDataset => {
      const imagesWithTextSimilarity = currentDataset.images.map(imageData => {
        if (!imageData.embeddings?.faces || imageData.embeddings.faces.length === 0) {
          return imageData; // Return unchanged if no face embeddings
        }

        // Compute similarity scores for each face
        const faceTextSimilarities = imageData.embeddings.faces.map(faceEmbedding => {
          const faceVector = Array.from(faceEmbedding.vector as Float32Array | number[]);

          // Compute similarity with all text embeddings
          const attributeSimilarities = textEmbeddings.map(textEmb => {
            const textVector = Array.from(textEmb.vector);
            const similarity = cosineSimilarity(faceVector, textVector);

            // Extract category and attribute from prompt by matching exact original strings
            let category = 'unknown';
            let attribute = textEmb.prompt;

            // Find exact match from ATTRIBUTES
            for (const attr of ATTRIBUTES) {
              for (const option of attr.options) {
                const expectedPrompt = attr.template(option);
                if (textEmb.prompt === expectedPrompt) {
                  category = attr.key;
                  attribute = option;
                  break;
                }
              }
              if (category !== 'unknown') break;
            }

            return {
              category,
              attribute,
              prompt: textEmb.prompt,
              similarity,
              faceIndex: faceEmbedding.faceIndex
            };
          });

          return {
            faceIndex: faceEmbedding.faceIndex,
            attributeSimilarities,
            bestMatches: getBestMatchesPerCategory(attributeSimilarities)
          };
        });

        // Aggregate all face similarities for this image
        const allAttributeSimilarities = faceTextSimilarities.flatMap(face =>
          face.attributeSimilarities
        );

        const allBestMatches = faceTextSimilarities.flatMap(face =>
          face.bestMatches
        );

        return {
          ...imageData,
          textSimilarity: {
            attributes: allAttributeSimilarities,
            bestMatches: allBestMatches
          },
          processing: {
            ...imageData.processing,
            stages: {
              ...imageData.processing.stages,
              textSimilarityCompleted: new Date()
            }
          }
        };
      });

      // Return the updated dataset
      const updatedDataset = {
        ...currentDataset,
        images: imagesWithTextSimilarity,
        metadata: {
          ...currentDataset.metadata,
          lastUpdated: new Date()
        }
      };

      console.log('Text similarity analysis completed for all faces');
      return updatedDataset;
    });
  }, [textEmbeddings]);

  // Helper function to find best matching attributes per category
  const getBestMatchesPerCategory = (attributeSimilarities: Array<{
    category: string;
    attribute: string;
    prompt: string;
    similarity: number;
    faceIndex: number;
  }>) => {
    const byCategory = attributeSimilarities.reduce((acc, item) => {
      if (!acc[item.category]) {
        acc[item.category] = [];
      }
      acc[item.category].push(item);
      return acc;
    }, {} as Record<string, typeof attributeSimilarities>);

    return Object.entries(byCategory).map(([category, items]) => {
      const best = items.reduce((max, item) =>
        item.similarity > max.similarity ? item : max
      );

      return {
        category,
        attribute: best.attribute,
        similarity: best.similarity,
        faceIndex: best.faceIndex
      };
    });
  };

  // Helper functions to access imageDataset data
  const getImagesBySource = useCallback((source: 'original' | 'ai' | 'reference') => {
    return imageDataset.images.filter(img => img.source === source)
  }, [imageDataset.images])

  const getImagesWithFaceDetection = useCallback(() => {
    return imageDataset.images.filter(img => img.faceDetection && img.faceDetection.landmarks.length > 0)
  }, [imageDataset.images])

  const getImagesWithEmbeddings = useCallback(() => {
    return imageDataset.images.filter(img => img.embeddings)
  }, [imageDataset.images])

  const getImageById = useCallback((id: string) => {
    return imageDataset.images.find(img => img.id === id)
  }, [imageDataset.images])

  const getAllFaceEmbeddings = useCallback(() => {
    const faceEmbeddings: Array<{ imageId: string, imageName: string, faceIndex: number, vector: number[] | Float32Array }> = []

    imageDataset.images.forEach(img => {
      if (img.embeddings?.faces) {
        img.embeddings.faces.forEach(face => {
          faceEmbeddings.push({
            imageId: img.id,
            imageName: img.name,
            faceIndex: face.faceIndex,
            vector: face.vector
          })
        })
      }
    })

    return faceEmbeddings
  }, [imageDataset.images])

  // Helper function to get images with text similarity analysis
  const getImagesWithTextSimilarity = useCallback(() => {
    return imageDataset.images.filter(img => img.textSimilarity && img.textSimilarity.attributes.length > 0)
  }, [imageDataset.images])

  // Helper function to get top text similarity matches across all images
  const getTopTextSimilarityMatches = useCallback((topN: number = 10) => {
    const allMatches: Array<{
      imageId: string;
      imageName: string;
      faceIndex: number;
      category: string;
      attribute: string;
      prompt: string;
      similarity: number;
    }> = [];

    imageDataset.images.forEach(img => {
      if (img.textSimilarity?.attributes) {
        img.textSimilarity.attributes.forEach(attr => {
          allMatches.push({
            imageId: img.id,
            imageName: img.name,
            faceIndex: attr.faceIndex || 0,
            category: attr.category,
            attribute: attr.attribute,
            prompt: attr.prompt,
            similarity: attr.similarity
          });
        });
      }
    });

    // Sort by similarity (highest first) and take top N
    return allMatches
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topN);
  }, [imageDataset.images])

  // Function to compute similarity of all cropped images to the first image's cropped face
  const computeSimilarityToSource = useCallback(() => {
    if (imageDataset.images.length === 0) {
      console.warn('No images available for similarity comparison');
      return;
    }

    // Find the first image (source image)
    const sourceImage = imageDataset.images[0];

    if (!sourceImage.embeddings?.faces || sourceImage.embeddings.faces.length === 0) {
      console.warn('Source image has no face embeddings available');
      return;
    }

    // Use the first face of the source image as reference
    const sourceFaceEmbedding = sourceImage.embeddings.faces[0];
    const sourceVector = Array.from(sourceFaceEmbedding.vector as Float32Array | number[]);

    console.log(`Computing similarity to source: ${sourceImage.name} (face ${sourceFaceEmbedding.faceIndex})`);

    // Update all images with similarity calculations
    setImageDataset(currentDataset => {
      const updatedImages = currentDataset.images.map((imageData, imageIndex) => {
        // Skip the source image itself
        if (imageIndex === 0) {
          return imageData;
        }

        if (!imageData.embeddings?.faces || imageData.embeddings.faces.length === 0) {
          return imageData; // Return unchanged if no face embeddings
        }

        // Calculate similarity for each face in this image
        const similarities = imageData.embeddings.faces.map(faceEmbedding => {
          const faceVector = Array.from(faceEmbedding.vector as Float32Array | number[]);
          const similarity = cosineSimilarity(sourceVector, faceVector);

          return {
            faceIndex: faceEmbedding.faceIndex,
            sourceImageId: sourceImage.id,
            sourceFaceIndex: sourceFaceEmbedding.faceIndex,
            similarity,
            timestamp: new Date()
          };
        });

        // Update the image with similarity results
        return {
          ...imageData,
          computeSimilarityToSource: {
            similarities
          },
          processing: {
            ...imageData.processing,
            stages: {
              ...imageData.processing.stages,
              similarityToSourceCompleted: new Date()
            }
          }
        };
      });

      return {
        ...currentDataset,
        images: updatedImages,
        metadata: {
          ...currentDataset.metadata,
          lastUpdated: new Date()
        }
      };
    });

    console.log('Similarity to source computation completed');
  }, [imageDataset.images, cosineSimilarity])

  // Example function to demonstrate accessing data
  const logDatasetSummary = useCallback(() => {
    console.log('=== Dataset Summary ===')
    console.log('Total images:', imageDataset.metadata.totalImages)
    console.log('Original images:', getImagesBySource('original').length)
    console.log('AI images:', getImagesBySource('ai').length)
    console.log('Images with face detection:', getImagesWithFaceDetection().length)
    console.log('Images with embeddings:', getImagesWithEmbeddings().length)
    console.log('Images with text similarity:', getImagesWithTextSimilarity().length)
    console.log('Total face embeddings:', getAllFaceEmbeddings().length)
    console.log('Total text embeddings:', textEmbeddings.length)

    // Log processing stages
    imageDataset.images.forEach(img => {
      console.log(`${img.name} processing stages:`, {
        uploaded: img.processing.stages.uploaded,
        faceDetection: img.processing.stages.faceDetectionCompleted ? 'completed' : 'pending',
        embeddings: img.processing.stages.embeddingCompleted ? 'completed' : 'pending',
        textSimilarity: img.processing.stages.textSimilarityCompleted ? 'completed' : 'pending',
        similarityToSource: img.processing.stages.similarityToSourceCompleted ? 'completed' : 'pending',
        errors: img.processing.errors?.length || 0
      })

      // Log text similarity results if available
      if (img.textSimilarity) {
        console.log(`${img.name} text similarity results:`)
        img.textSimilarity.bestMatches.forEach(match => {
          console.log(`  ${match.category}: ${match.attribute} (similarity: ${match.similarity.toFixed(3)}, face: ${match.faceIndex})`)
        })
      }

      // Log similarity to source results if available
      if (img.computeSimilarityToSource) {
        console.log(`${img.name} similarity to source results:`)
        img.computeSimilarityToSource.similarities.forEach(sim => {
          console.log(`  Face ${sim.faceIndex}: ${sim.similarity.toFixed(3)} (source: face ${sim.sourceFaceIndex})`)
        })
      }
    })

    // Log top text similarity matches across all images
    if (getImagesWithTextSimilarity().length > 0) {
      console.log('\n=== Top Text Similarity Matches ===')
      const topMatches = getTopTextSimilarityMatches(15)
      topMatches.forEach((match, index) => {
        console.log(`${index + 1}. ${match.imageName} (face ${match.faceIndex}) - ${match.category}: ${match.attribute} (${match.similarity.toFixed(3)})`)
      })
    }
  }, [imageDataset, getImagesBySource, getImagesWithFaceDetection, getImagesWithEmbeddings, getImagesWithTextSimilarity, getAllFaceEmbeddings, textEmbeddings.length, getTopTextSimilarityMatches])

  return (
    <div className="relative w-full h-full min-h-screen bg-gray-100">

      <div id='loader' className='fixed top-4 right-4 z-50'>
        {isExtractorLoading && (
          <LoadingComponent
            progress={loadingProgress}
            message="Loading AI models and initializing..."
            size="small"
            className="w-72 shadow-lg bg-white"
          />
        )}

        {/* Success message when models are loaded */}
        {!isExtractorLoading && (
          <div className="bg-green-50 rounded-lg p-3 shadow-lg">
            <div className="text-xs text-green-700 font-regular">✓ AI models loaded successfully</div>
          </div>
        )}
      </div>

      <div className="flex flex-row gap-4">
        <div className="card">
          <h2>Image Drop Zone</h2>
          <ImageDropZone
            onImagesUploaded={handleImagesUploaded}
            maxFiles={10}
            files={uploadedFiles}
          />
        </div>
        <div className="card">
          <h2>AI Image Drop Zone</h2>
          <ImageDropZone
            onImagesUploaded={handleAIImagesUploaded}
            maxFiles={10}
            files={uploadedAIFiles}
          />
        </div>
      </div>

      <div className="mt-4 flex gap-4">
        <button
          type="button"
          onClick={() => analyizeImages()}
          className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
          disabled={uploadedFiles.length === 0 || uploadedAIFiles.length === 0}
        >
          analyize similarity
        </button>

        <button
          type="button"
          onClick={logDatasetSummary}
          className="px-4 py-2 rounded bg-green-600 text-white hover:bg-green-700 disabled:opacity-50"
          disabled={imageDataset.images.length === 0}
        >
          Log Dataset Summary
        </button>

        <button
          type="button"
          onClick={computeTextSimilarityForAllFaces}
          className="px-4 py-2 rounded bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50"
          disabled={imageDataset.images.length === 0 || textEmbeddings.length === 0}
        >
          Compute Text Similarity
        </button>

        <button
          type="button"
          onClick={computeSimilarityToSource}
          className="px-4 py-2 rounded bg-orange-600 text-white hover:bg-orange-700 disabled:opacity-50"
          disabled={imageDataset.images.length < 2 || !imageDataset.images[0]?.embeddings?.faces?.[0]}
        >
          Compute Similarity to Source
        </button>
      </div>
      {overlays.length > 0 && (
        <div className="mt-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4">Face Detection Results</h2>
          <ImageWithLandmarks
            results={overlays}
            maxWidth={640}
            showChips={true}
            className="mt-4"
          />
        </div>
      )}
      {/* reserved for future face detection UI */}
    </div>
  )
}

export default App
