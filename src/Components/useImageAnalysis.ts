// Custom hook for image analysis operations
// This hook will contain reusable image analysis logic

import { useState, useCallback } from 'react'
import type { ImageData, ImageAnalysisDataset } from '../imageAnalysisTypes'

export const useImageAnalysis = () => {
  const [dataset, setDataset] = useState<ImageAnalysisDataset>({
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

  const addImages = useCallback((files: File[], source: 'original' | 'ai' | 'reference') => {
    const now = new Date()
    const newImages: ImageData[] = files.map(file => ({
      id: `${source}-${file.name}-${file.size}-${file.lastModified}`,
      file,
      name: file.name,
      url: URL.createObjectURL(file),
      size: {
        width: 0,
        height: 0,
        fileSize: file.size
      },
      source,
      processing: {
        stages: {
          uploaded: now
        }
      }
    }))

    setDataset(prev => ({
      ...prev,
      images: [...prev.images, ...newImages],
      metadata: {
        ...prev.metadata,
        lastUpdated: now,
        totalImages: prev.images.length + newImages.length,
        [source === 'original' ? 'originalImages' : source === 'ai' ? 'aiImages' : 'referenceImages']: 
          prev.metadata[source === 'original' ? 'originalImages' : source === 'ai' ? 'aiImages' : 'referenceImages'] + newImages.length
      }
    }))

    return newImages
  }, [])

  const updateImage = useCallback((imageId: string, updates: Partial<ImageData>) => {
    setDataset(prev => ({
      ...prev,
      images: prev.images.map(img => 
        img.id === imageId ? { ...img, ...updates } : img
      ),
      metadata: {
        ...prev.metadata,
        lastUpdated: new Date()
      }
    }))
  }, [])

  const clearDataset = useCallback(() => {
    // Revoke all object URLs to prevent memory leaks
    dataset.images.forEach(img => {
      URL.revokeObjectURL(img.url)
      img.faceDetection?.chips.forEach(chip => URL.revokeObjectURL(chip.url))
    })

    setDataset({
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
  }, [dataset.images])

  return {
    dataset,
    addImages,
    updateImage,
    clearDataset,
    setDataset
  }
}
