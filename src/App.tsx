import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import ImageDropZone from './Components/ImageDropZone'
import { pipeline, env } from '@huggingface/transformers'
import { processFaceFile } from './Components/faceLandmarker'
import ImageWithLandmarks from './Components/ImageWithLandmarksHelper'
import { ATTRIBUTES } from './Components/Attributes'
import { compilePrompts } from './Components/Helpers'
import OpenaiHandler from './Components/OpenAIHandler'

import ActionFooter from './Components/ActionFooter'
import PageTitle from './Components/PageTitle'
import type { ImageData, ImageAnalysisDataset } from './imageAnalysisTypes'

import { AutoTokenizer, CLIPTextModelWithProjection } from '@huggingface/transformers'

// Temporarily disable browser cache to avoid using any corrupted ONNX files
env.useBrowserCache = false


const CLIP_MODEL_ID = "Xenova/clip-vit-base-patch32"; // image model


import img01 from './assets/ref_photos/ref_closeup_01.png'
import img02 from './assets/ref_photos/ref_closeup_02.png'
import img03 from './assets/ref_photos/man_beard.jpeg'
import img04 from './assets/ref_photos/woman_01.jpeg'
import img05 from './assets/ref_photos/smiling.jpeg'
import img06 from './assets/ref_photos/african_american_01.jpeg'
import img07 from './assets/ref_photos/man_beard_glasses.jpeg'
import img08 from './assets/ref_photos/young_girl.jpeg'


function App() {

  // Cache the image feature extractor so the model loads only once
  const imageFeatureExtractorRef = useRef<any | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false)

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
        { url: img01, name: 'ref_closeup_01.png' },
        { url: img02, name: 'ref_closeup_02.jpeg' },
        { url: img03, name: 'man_beard.jpeg' },
        { url: img04, name: 'woman_01.jpeg' },
        { url: img05, name: 'smiling.jpeg' },
        { url: img06, name: 'african_american_01.jpeg' },
        { url: img07, name: 'man_beard_glasses.jpeg' },
        { url: img08, name: 'young_girl.jpeg' }
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

    // Example: Access individual image data
    if (imageDataset.images.length > 0) {
      // console.log('First image data:', imageDataset.images[0])

      // Check if first image has face detection results
      if (imageDataset.images[0].faceDetection) {
        // console.log('First image has', imageDataset.images[0].faceDetection.landmarks.length, 'faces detected')
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
    setIsAnalyzing(true)
    
    try {
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
      try {
        await computeSimilarityToSource()

        // Sort images by similarity to source after analysis is complete
        console.log('Sorting images by similarity to source...')
        await sortImagesBySimilarityToSource()
        console.log('Sorting completed successfully')
      } catch (error) {
        console.error('Error in computeSimilarityToSource:', error)
      }
      
      
      // TODO: Add UMAP visualization
      // TODO: Add comparison matrix generation
    } catch (error) {
      console.error('Error during image analysis:', error)
    } finally {
      // Always reset isAnalyzing to false, regardless of success or failure
      setIsAnalyzing(false)
    }

    // sort all the images by similarity to source
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
  const cosineSimilarity = useCallback((a: number[], b: number[]): number => {
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
  }, []);




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

      return updatedDataset;
    });
  }, [textEmbeddings]);

  // Function to analyze text similarity to source image by comparing bestMatches
  const analyzeTextSimilarityToSource = useCallback((dataset: ImageAnalysisDataset) => {
    if (dataset.images.length === 0) {
      console.warn('No images available for text similarity to source comparison');
      return;
    }

    // Find the source image (first image)
    const sourceImage = dataset.images[0];

    if (!sourceImage.textSimilarity?.bestMatches || sourceImage.textSimilarity.bestMatches.length === 0) {
      console.warn('Source image has no text similarity bestMatches available');
      return;
    }

    console.log(`Analyzing text similarity to source: ${sourceImage.name}`);

    // Update all images with text similarity to source analysis
    setImageDataset(currentDataset => {
      const updatedImages = currentDataset.images.map((imageData, imageIndex) => {
        // Skip the source image itself
        if (imageIndex === 0) {
          return imageData;
        }

        if (!imageData.textSimilarity?.bestMatches || imageData.textSimilarity.bestMatches.length === 0) {
          return imageData; // Return unchanged if no text similarity data
        }

        // Create a map of source image's best matches by category
        const sourceBestMatchesByCategory = sourceImage.textSimilarity!.bestMatches.reduce((acc, match) => {
          acc[match.category] = match;
          return acc;
        }, {} as Record<string, typeof sourceImage.textSimilarity.bestMatches[0]>);

        // Create a map of current image's best matches by category
        const currentBestMatchesByCategory = imageData.textSimilarity.bestMatches.reduce((acc, match) => {
          acc[match.category] = match;
          return acc;
        }, {} as Record<string, typeof imageData.textSimilarity.bestMatches[0]>);

        // Compare categories
        const categoryComparisons = Object.keys(sourceBestMatchesByCategory).map(category => {
          const sourceMatch = sourceBestMatchesByCategory[category];
          const currentMatch = currentBestMatchesByCategory[category];

          if (!currentMatch) {
            // Current image doesn't have this category
            return {
              category,
              sourceAttribute: sourceMatch.attribute,
              currentAttribute: 'N/A',
              sourceSimilarity: sourceMatch.similarity,
              currentSimilarity: 0,
              isMatching: false,
              faceIndex: undefined,
              sourceFaceIndex: sourceMatch.faceIndex
            };
          }

          return {
            category,
            sourceAttribute: sourceMatch.attribute,
            currentAttribute: currentMatch.attribute,
            sourceSimilarity: sourceMatch.similarity,
            currentSimilarity: currentMatch.similarity,
            isMatching: sourceMatch.attribute === currentMatch.attribute,
            faceIndex: currentMatch.faceIndex,
            sourceFaceIndex: sourceMatch.faceIndex
          };
        });

        // Calculate summary statistics
        const totalCategories = categoryComparisons.length;
        const matchingCategories = categoryComparisons.filter(comp => comp.isMatching).length;
        const nonMatchingCategories = totalCategories - matchingCategories;
        const matchingPercentage = totalCategories > 0 ? (matchingCategories / totalCategories) * 100 : 0;

        // Update the image with text similarity to source results
        return {
          ...imageData,
          textSimilarityToSource: {
            categoryComparisons,
            summary: {
              totalCategories,
              matchingCategories,
              nonMatchingCategories,
              matchingPercentage,
              timestamp: new Date()
            }
          },
          processing: {
            ...imageData.processing,
            stages: {
              ...imageData.processing.stages,
              textSimilarityToSourceCompleted: new Date()
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

    console.log('Text similarity to source analysis completed');
  }, []);

  // Run text similarity to source analysis after text similarity is completed
  useEffect(() => {
    const imagesWithTextSimilarity = imageDataset.images.filter(img => 
      img.textSimilarity && img.textSimilarity.bestMatches.length > 0
    );

    // Check if we have text similarity data for at least 2 images (source + others)
    if (imagesWithTextSimilarity.length >= 2) {
      // Check if any image is missing textSimilarityToSource analysis
      const needsAnalysis = imageDataset.images.some((img, index) => 
        index > 0 && // Skip source image
        img.textSimilarity?.bestMatches && img.textSimilarity.bestMatches.length > 0 && // Has text similarity
        !img.textSimilarityToSource // Missing text similarity to source
      );

      if (needsAnalysis) {
        console.log('Running text similarity to source analysis...');
        analyzeTextSimilarityToSource(imageDataset);
      }
    }
  }, [imageDataset.images, analyzeTextSimilarityToSource]);

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

  // // Helper functions to access imageDataset data
  // const getImagesBySource = useCallback((source: 'original' | 'ai' | 'reference') => {
  //   return imageDataset.images.filter(img => img.source === source)
  // }, [imageDataset.images])

  // const getImagesWithFaceDetection = useCallback(() => {
  //   return imageDataset.images.filter(img => img.faceDetection && img.faceDetection.landmarks.length > 0)
  // }, [imageDataset.images])

  // const getImagesWithEmbeddings = useCallback(() => {
  //   return imageDataset.images.filter(img => img.embeddings)
  // }, [imageDataset.images])



  // const getAllFaceEmbeddings = useCallback(() => {
  //   const faceEmbeddings: Array<{ imageId: string, imageName: string, faceIndex: number, vector: number[] | Float32Array }> = []

  //   imageDataset.images.forEach(img => {
  //     if (img.embeddings?.faces) {
  //       img.embeddings.faces.forEach(face => {
  //         faceEmbeddings.push({
  //           imageId: img.id,
  //           imageName: img.name,
  //           faceIndex: face.faceIndex,
  //           vector: face.vector
  //         })
  //       })
  //     }
  //   })

  //   return faceEmbeddings
  // }, [imageDataset.images])

  // // Helper function to get images with text similarity analysis
  // const getImagesWithTextSimilarity = useCallback(() => {
  //   return imageDataset.images.filter(img => img.textSimilarity && img.textSimilarity.attributes.length > 0)
  // }, [imageDataset.images])

  // // Helper function to get top text similarity matches across all images
  // const getTopTextSimilarityMatches = useCallback((topN: number = 10) => {
  //   const allMatches: Array<{
  //     imageId: string;
  //     imageName: string;
  //     faceIndex: number;
  //     category: string;
  //     attribute: string;
  //     prompt: string;
  //     similarity: number;
  //   }> = [];

  //   imageDataset.images.forEach(img => {
  //     if (img.textSimilarity?.attributes) {
  //       img.textSimilarity.attributes.forEach(attr => {
  //         allMatches.push({
  //           imageId: img.id,
  //           imageName: img.name,
  //           faceIndex: attr.faceIndex || 0,
  //           category: attr.category,
  //           attribute: attr.attribute,
  //           prompt: attr.prompt,
  //           similarity: attr.similarity
  //         });
  //       });
  //     }
  //   });

  //   // Sort by similarity (highest first) and take top N
  //   return allMatches
  //     .sort((a, b) => b.similarity - a.similarity)
  //     .slice(0, topN);
  // }, [imageDataset.images])

  // Function to compute similarity of all cropped images to the first image's cropped face
  const computeSimilarityToSource = useCallback(async () => {
    console.log('computeSimilarityToSource: Starting execution');
    
    // Get current dataset state to avoid stale closure
    setImageDataset(currentDataset => {
      console.log('computeSimilarityToSource: Inside setImageDataset callback, images count:', currentDataset.images.length);
      
      if (currentDataset.images.length === 0) {
        console.warn('No images available for similarity comparison');
        return currentDataset;
      }

      // Find the first image (source image)
      const sourceImage = currentDataset.images[0];
      console.log('computeSimilarityToSource: Source image:', sourceImage.name, 'has embeddings:', !!sourceImage.embeddings?.faces);

      if (!sourceImage.embeddings?.faces || sourceImage.embeddings.faces.length === 0) {
        console.warn('Source image has no face embeddings available');
        return currentDataset;
      }

      // Use the first face of the source image as reference
      const sourceFaceEmbedding = sourceImage.embeddings.faces[0];
      const sourceVector = Array.from(sourceFaceEmbedding.vector as Float32Array | number[]);

      console.log(`Computing similarity to source: ${sourceImage.name} (face ${sourceFaceEmbedding.faceIndex})`);

      // Update all images with similarity calculations
      const updatedImages = currentDataset.images.map((imageData, imageIndex) => {
        console.log(`Processing image ${imageIndex}: ${imageData.name}`);
        
        // Skip the source image itself
        if (imageIndex === 0) {
          console.log('Skipping source image');
          return imageData;
        }

        if (!imageData.embeddings?.faces || imageData.embeddings.faces.length === 0) {
          console.log(`Image ${imageData.name} has no face embeddings, skipping`);
          return imageData; // Return unchanged if no face embeddings
        }

        // Calculate similarity for each face in this image
        console.log(`Computing similarities for ${imageData.embeddings.faces.length} faces in ${imageData.name}`);
        const similarities = imageData.embeddings.faces.map(faceEmbedding => {
          const faceVector = Array.from(faceEmbedding.vector as Float32Array | number[]);
          const similarity = cosineSimilarity(sourceVector, faceVector);
          console.log(`Face ${faceEmbedding.faceIndex} similarity: ${similarity}`);

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

      console.log('Similarity to source computation completed, updated images count:', updatedImages.length);

      return {
        ...currentDataset,
        images: updatedImages,
        metadata: {
          ...currentDataset.metadata,
          lastUpdated: new Date()
        }
      };
    });
    
    console.log('computeSimilarityToSource: Finished execution');
  }, [cosineSimilarity])

  // Function to sort images by similarity to source while keeping the first image as reference
  const sortImagesBySimilarityToSource = useCallback(async () => {
    console.log('sortImagesBySimilarityToSource: Starting execution');
    
    setImageDataset(currentDataset => {
      console.log('sortImagesBySimilarityToSource: Inside setImageDataset callback');
      
      if (currentDataset.images.length <= 1) {
        console.warn('Not enough images to sort');
        return currentDataset;
      }

      // Keep the first image as reference (source)
      const [sourceImage, ...otherImages] = currentDataset.images;
      
      // Sort other images by their highest similarity score to source
      const sortedOtherImages = [...otherImages].sort((a, b) => {
        // Get the highest similarity score for each image
        const getSimilarityScore = (imageData: ImageData): number => {
          if (!imageData.computeSimilarityToSource?.similarities || imageData.computeSimilarityToSource.similarities.length === 0) {
            return -1; // Place images without similarity data at the end
          }
          
          // Return the highest similarity score among all faces in this image
          return Math.max(...imageData.computeSimilarityToSource.similarities.map(sim => sim.similarity));
        };

        const similarityA = getSimilarityScore(a);
        const similarityB = getSimilarityScore(b);
        
        // Sort in descending order (highest similarity first)
        return similarityB - similarityA;
      });

      console.log('Sorted images by similarity:');
      sortedOtherImages.forEach((img, index) => {
        const maxSimilarity = img.computeSimilarityToSource?.similarities ? 
          Math.max(...img.computeSimilarityToSource.similarities.map(sim => sim.similarity)) : 0;
        console.log(`${index + 1}. ${img.name}: ${(maxSimilarity * 100).toFixed(1)}%`);
      });

      const sortedImages = [sourceImage, ...sortedOtherImages];

      // Update overlays to match the new sorted order
      const sortedOverlayResults = sortedImages
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
        }));

      // Update overlays state to reflect the new order
      setOverlays(sortedOverlayResults);

      return {
        ...currentDataset,
        images: sortedImages,
        metadata: {
          ...currentDataset.metadata,
          lastUpdated: new Date()
        }
      };
    });
    
    console.log('sortImagesBySimilarityToSource: Finished execution');
  }, [])

  // Build a concise prompt for OpenAI describing similarities and differences
  const buildOpenAiPromptForImage = useCallback((image: ImageData, source: ImageData) => {
    const header = `You are comparing a target image to a source (reference) face. Write a concise 2-4 sentence paragraph, plain English, describing the main similarities and differences. Focus on what stands out. Avoid hedging. Keep it under 90 words.`;

    const summary = image.textSimilarityToSource?.summary
      ? `Overall: ${image.textSimilarityToSource.summary.matchingCategories}/${image.textSimilarityToSource.summary.totalCategories} categories match (${image.textSimilarityToSource.summary.matchingPercentage.toFixed(1)}%).`
      : 'Overall: summary unavailable.';

    const lines = (image.textSimilarityToSource?.categoryComparisons || []).map(c => {
      const status = c.isMatching ? 'match' : 'different';
      const src = `${c.sourceAttribute} (${(c.sourceSimilarity * 100).toFixed(0)}%)`;
      const cur = `${c.currentAttribute} (${(c.currentSimilarity * 100).toFixed(0)}%)`;
      return `- ${c.category}: ${status}; source=${src}; current=${cur}`;
    });

    const detailsBlock = lines.slice(0, 24).join('\n'); // cap to avoid huge prompts

    return `${header}\nSource image name: ${source.name}\nTarget image name: ${image.name}\n${summary}\nDetails:\n${detailsBlock}\n\nWrite the paragraph now:`;
  }, []);

  // Generate OpenAI narratives for all images that have textSimilarityToSource
  const generateOpenAiNarrativesForAllImages = useCallback(async () => {
    try {
      const datasetSnapshot = imageDataset; // local snapshot
      if (!datasetSnapshot || datasetSnapshot.images.length === 0) return;

      const sourceImage = datasetSnapshot.images[0];

      // Prepare OpenAI calls for all eligible images in parallel
      const tasks = datasetSnapshot.images.map(async (img, index) => {
        // Only generate for non-source images with available text similarity-to-source
        if (index === 0 || !img.textSimilarityToSource || !img.textSimilarityToSource.categoryComparisons?.length) {
          return { id: img.id, result: null as null | { summary: string; model?: string; promptPreview?: string; error?: string } };
        }

        const prompt = buildOpenAiPromptForImage(img, sourceImage);

        try {
          const data = await OpenaiHandler({ prompt });
          const modelUsed: string | undefined = (data && (data.model as string)) || 'gpt-4.1-nano';
          const choice0 = data?.choices?.[0];
          const content: string = (choice0?.message?.content ?? choice0?.text ?? '').toString();

          return {
            id: img.id,
            result: {
              summary: content,
              model: modelUsed,
              promptPreview: prompt.slice(0, 200)
            } as { summary: string; model?: string; promptPreview?: string; error?: string }
          };
        } catch (err: any) {
          const errorMessage = err instanceof Error ? err.message : String(err);
          return {
            id: img.id,
            result: {
              summary: '',
              promptPreview: prompt.slice(0, 200),
              error: errorMessage
            } as { summary: string; model?: string; promptPreview?: string; error?: string }
          };
        }
      });

      const results = await Promise.all(tasks);

      // Apply results to dataset in a single state update
      setImageDataset(currentDataset => {
        const resultsById = new Map(results.map(r => [r.id, r.result]));

        const updatedImages = currentDataset.images.map(imageData => {
          const res = resultsById.get(imageData.id);
          if (!res) return imageData; // no-op when not processed

          // If result is null, leave unchanged
          if (res === null) return imageData;

          // If generation produced an error, record it
          if (res.error) {
            return {
              ...imageData,
              processing: {
                ...imageData.processing,
                errors: [
                  ...(imageData.processing.errors || []),
                  {
                    stage: 'openAiNarrative',
                    error: res.error,
                    timestamp: new Date()
                  }
                ]
              }
            };
          }

          // Normal successful update
          return {
            ...imageData,
            openAiNarrative: {
              summary: res.summary,
              model: res.model,
              promptPreview: res.promptPreview,
              timestamp: new Date()
            },
            processing: {
              ...imageData.processing,
              stages: {
                ...imageData.processing.stages,
                openAiNarrativeCompleted: new Date()
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

    } catch (e) {
      console.error('generateOpenAiNarrativesForAllImages failed:', e);
    }
  }, [imageDataset, buildOpenAiPromptForImage]);



  // After textSimilarityToSource is computed, trigger OpenAI narratives for images missing it
  useEffect(() => {
    const imagesNeedingNarrative = imageDataset.images.filter((img, index) =>
      index > 0 &&
      !!img.textSimilarityToSource?.categoryComparisons?.length &&
      !img.openAiNarrative
    );

    if (imagesNeedingNarrative.length > 0) {
      (async () => {
        console.log('Generating OpenAI narratives for', imagesNeedingNarrative.length, 'images...')
        await generateOpenAiNarrativesForAllImages();
        console.log('OpenAI narratives generation completed')
      })();
    }
  }, [imageDataset.images, generateOpenAiNarrativesForAllImages]);

  return (
    <div className="relative w-full h-full min-h-screen bg-gray-100 pb-20">
      <div className="p-6">
        <PageTitle />
      </div>
    

      <div className="px-6">
        <div className="flex flex-row gap-2">
          <div className="card flex flex-col gap-4 items-baseline w-[50vw]">
            <p className="text-sm font-regular text-gray-600">Target Image</p>
            <ImageDropZone
              onImagesUploaded={handleImagesUploaded}
              maxFiles={10}
              files={uploadedFiles}
            />
          </div>
          <div className="card flex flex-col gap-4 items-baseline w-[50vw]">
          <p className="text-sm font-regular text-gray-600">Reference Images (up too 10)</p>
            <ImageDropZone
              onImagesUploaded={handleAIImagesUploaded}
              maxFiles={10}
              files={uploadedAIFiles}
            />
          </div>
        </div>
      </div>


      {overlays.length > 0 && (
        <div className="mt-6 p-8 pb-24 flex flex-col gap-4">
          <div className="flex flex-row gap-4">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Face Detection Results</h2>
          </div>
          <ImageWithLandmarks
            results={overlays}
            maxWidth={640}
            showChips={true}
            className="mt-4"
            imageDataset={imageDataset}
          />
        </div>
      )}
      {/* reserved for future face detection UI */}
      
      <ActionFooter
        isAnalyzing={isAnalyzing}
        onAnalyzeImages={() => analyizeImages()}
        isAnalyzeDisabled={uploadedFiles.length === 0 || uploadedAIFiles.length === 0}
        isExtractorLoading={isExtractorLoading}
        loadingProgress={loadingProgress}
      />
    </div>
  )
}

export default App
