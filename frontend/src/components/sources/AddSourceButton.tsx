'use client'

import { useState, useRef, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { LoaderIcon } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { WizardContainer, WizardStep } from '@/components/ui/wizard-container'
import { SourceTypeStep } from './steps/SourceTypeStep'
import { NotebooksStep } from './steps/NotebooksStep'
import { ProcessingStep } from './steps/ProcessingStep'
import { useNotebooks } from '@/lib/hooks/use-notebooks'
import { useTransformations } from '@/lib/hooks/use-transformations'
import { useCreateSource } from '@/lib/hooks/use-sources'
import { useSettings } from '@/lib/hooks/use-settings'
import { CreateSourceRequest } from '@/lib/types/api'

const createSourceSchema = z.object({
  type: z.enum(['link', 'upload', 'text']),
  title: z.string().optional(),
  url: z.string().optional(),
  content: z.string().optional(),
  file: z.any().optional(),
  notebooks: z.array(z.string()).optional(),
  transformations: z.array(z.string()).optional(),
  embed: z.boolean(),
  build_pageindex: z.boolean(),
  async_processing: z.boolean(),
}).refine((data) => {
  if (data.type === 'link') {
    return !!data.url && data.url.trim() !== ''
  }
  if (data.type === 'text') {
    return !!data.content && data.content.trim() !== ''
  }
  if (data.type === 'upload') {
    if (data.file instanceof FileList) {
      return data.file.length > 0
    }
    return !!data.file
  }
  return true
}, {
  message: 'Please provide the required content for the selected source type',
  path: ['type'],
}).refine((data) => {
  // Make title mandatory for text sources
  if (data.type === 'text') {
    return !!data.title && data.title.trim() !== ''
  }
  return true
}, {
  message: 'Title is required for text sources',
  path: ['title'],
})

type CreateSourceFormData = z.infer<typeof createSourceSchema>

interface AddSourceDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  defaultNotebookId?: string
}

const WIZARD_STEPS: readonly WizardStep[] = [
  { number: 1, title: 'Source & Content', description: 'Choose type and add content' },
  { number: 2, title: 'Organization', description: 'Select notebooks' },
  { number: 3, title: 'Processing', description: 'Choose transformations and options' },
]

interface ProcessingState {
  message: string
  progress?: number
  current?: number
  total?: number
  failed?: number
  successful?: number
}

export function AddSourceDialog({ 
  open, 
  onOpenChange, 
  defaultNotebookId 
}: AddSourceDialogProps) {
  // Simplified state management
  const [currentStep, setCurrentStep] = useState(1)
  const [processing, setProcessing] = useState(false)
  const [processingStatus, setProcessingStatus] = useState<ProcessingState | null>(null)
  const [selectedNotebooks, setSelectedNotebooks] = useState<string[]>(
    defaultNotebookId ? [defaultNotebookId] : []
  )
  const [selectedTransformations, setSelectedTransformations] = useState<string[]>([])

  // Cleanup timeouts to prevent memory leaks
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  // API hooks
  const createSource = useCreateSource()
  const { data: notebooks = [], isLoading: notebooksLoading } = useNotebooks()
  const { data: transformations = [], isLoading: transformationsLoading } = useTransformations()
  const { data: settings } = useSettings()

  // Form setup
  const {
    register,
    handleSubmit,
    control,
    watch,
    formState: { errors },
    reset,
  } = useForm<CreateSourceFormData>({
    resolver: zodResolver(createSourceSchema),
    defaultValues: {
      notebooks: defaultNotebookId ? [defaultNotebookId] : [],
      embed: settings?.default_embedding_option === 'always' || settings?.default_embedding_option === 'ask',
      build_pageindex: false,
      async_processing: true,
      transformations: [],
    },
  })

  // Initialize form values when settings and transformations are loaded
  useEffect(() => {
    if (settings && transformations.length > 0) {
      const defaultTransformations = transformations
        .filter(t => t.apply_default)
        .map(t => t.id)

      setSelectedTransformations(defaultTransformations)

      // Reset form with proper embed value based on settings
      const embedValue = settings.default_embedding_option === 'always' ||
                         (settings.default_embedding_option === 'ask')

      reset({
        notebooks: defaultNotebookId ? [defaultNotebookId] : [],
        embed: embedValue,
        build_pageindex: false,
        async_processing: true,
        transformations: [],
      })
    }
  }, [settings, transformations, defaultNotebookId, reset])

  // Cleanup effect
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  const selectedType = watch('type')
  const watchedUrl = watch('url')
  const watchedContent = watch('content')
  const watchedFile = watch('file')
  const watchedTitle = watch('title')

  // Step validation - now reactive with watched values
  const isStepValid = (step: number): boolean => {
    switch (step) {
      case 1:
        if (!selectedType) return false
        if (selectedType === 'link') {
          return !!watchedUrl && watchedUrl.trim() !== ''
        }
        if (selectedType === 'text') {
          return !!watchedContent && watchedContent.trim() !== '' &&
                 !!watchedTitle && watchedTitle.trim() !== ''
        }
        if (selectedType === 'upload') {
          if (watchedFile instanceof FileList) {
            return watchedFile.length > 0
          }
          return !!watchedFile
        }
        return true
      case 2:
      case 3:
        return true
      default:
        return false
    }
  }

  // Navigation
  const handleNextStep = (e?: React.MouseEvent) => {
    e?.preventDefault()
    e?.stopPropagation()
    if (currentStep < 3 && isStepValid(currentStep)) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handlePrevStep = (e?: React.MouseEvent) => {
    e?.preventDefault()
    e?.stopPropagation()
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleStepClick = (step: number) => {
    if (step <= currentStep || (step === currentStep + 1 && isStepValid(currentStep))) {
      setCurrentStep(step)
    }
  }

  // Selection handlers
  const handleNotebookToggle = (notebookId: string) => {
    const updated = selectedNotebooks.includes(notebookId)
      ? selectedNotebooks.filter(id => id !== notebookId)
      : [...selectedNotebooks, notebookId]
    setSelectedNotebooks(updated)
  }

  const handleTransformationToggle = (transformationId: string) => {
    const updated = selectedTransformations.includes(transformationId)
      ? selectedTransformations.filter(id => id !== transformationId)
      : [...selectedTransformations, transformationId]
    setSelectedTransformations(updated)
  }

  // Helper function to get filename without extension
  const getFileNameWithoutExtension = (filename: string): string => {
    const lastDotIndex = filename.lastIndexOf('.')
    return lastDotIndex > 0 ? filename.substring(0, lastDotIndex) : filename
  }

  // Form submission
  const onSubmit = async (data: CreateSourceFormData) => {
    try {
      setProcessing(true)

      // Check if this is a batch upload (multiple files)
      const isBatchUpload = data.type === 'upload' && 
                           data.file && 
                           data.file instanceof FileList && 
                           data.file.length > 1

      if (isBatchUpload) {
        // Batch upload mode
        const files = Array.from(data.file as FileList)
        const totalFiles = files.length
        let successful = 0
        let failed = 0

        setProcessingStatus({ 
          message: `Uploading files... (0/${totalFiles})`,
          current: 0,
          total: totalFiles,
          successful: 0,
          failed: 0
        })

        // Process each file sequentially
        for (let i = 0; i < files.length; i++) {
          const file = files[i]
          const fileName = getFileNameWithoutExtension(file.name)

          setProcessingStatus({ 
            message: `Uploading "${file.name}"... (${i + 1}/${totalFiles})`,
            current: i + 1,
            total: totalFiles,
            successful,
            failed
          })

          try {
            const createRequest: CreateSourceRequest & { file?: File } = {
              type: 'upload',
              notebooks: selectedNotebooks,
              title: fileName, // Use filename as title
              transformations: selectedTransformations,
              embed: data.embed,
              build_pageindex: data.build_pageindex,
              delete_source: false,
              async_processing: true,
              file: file, // Include file in the request
            }

            await createSource.mutateAsync(createRequest)
            successful++
          } catch (error) {
            console.error(`Error uploading file "${file.name}":`, error)
            failed++
            // Continue with next file even if this one fails
          }
        }

        // Show final status
        setProcessingStatus({ 
          message: `Upload complete: ${successful} successful, ${failed} failed`,
          current: totalFiles,
          total: totalFiles,
          successful,
          failed
        })

        // Wait a bit before closing to show final status
        setTimeout(() => {
          handleClose()
        }, 2000)
      } else {
        // Single file or non-upload mode (existing behavior)
        setProcessingStatus({ message: 'Submitting source for processing...' })

        const createRequest: CreateSourceRequest = {
          type: data.type,
          notebooks: selectedNotebooks,
          url: data.type === 'link' ? data.url : undefined,
          content: data.type === 'text' ? data.content : undefined,
          title: data.title,
          transformations: selectedTransformations,
          embed: data.embed,
          build_pageindex: data.build_pageindex,
          delete_source: false,
          async_processing: true, // Always use async processing for frontend submissions
        }

        
        if (data.type === 'upload' && data.file) {
          const file = data.file instanceof FileList ? data.file[0] : data.file
          const requestWithFile = createRequest as CreateSourceRequest & { file?: File }
          requestWithFile.file = file
          
          // If no title provided, use filename
          if (!createRequest.title && file instanceof File) {
            createRequest.title = getFileNameWithoutExtension(file.name)
          }
        }

        await createSource.mutateAsync(createRequest)

        // Close immediately - the toast will show the success message
        handleClose()
      }
    } catch (error) {
      console.error('Error creating source:', error)
      setProcessingStatus({ 
        message: 'Error creating source. Please try again.',
      })
      timeoutRef.current = setTimeout(() => {
        setProcessing(false)
        setProcessingStatus(null)
      }, 3000)
    }
  }

  // Dialog management
  const handleClose = () => {
    // Clear any pending timeouts
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }

    reset()
    setCurrentStep(1)
    setProcessing(false)
    setProcessingStatus(null)
    setSelectedNotebooks(defaultNotebookId ? [defaultNotebookId] : [])

    // Reset to default transformations
    if (transformations.length > 0) {
      const defaultTransformations = transformations
        .filter(t => t.apply_default)
        .map(t => t.id)
      setSelectedTransformations(defaultTransformations)
    } else {
      setSelectedTransformations([])
    }

    onOpenChange(false)
  }

  // Processing view
  if (processing) {
    return (
      <Dialog open={open} onOpenChange={handleClose}>
        <DialogContent className="sm:max-w-[500px]" showCloseButton={true}>
          <DialogHeader>
            <DialogTitle>Processing Source</DialogTitle>
            <DialogDescription>
              Your source is being processed. This may take a few moments.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="flex items-center gap-3">
              <LoaderIcon className="h-5 w-5 animate-spin text-primary" />
              <span className="text-sm text-muted-foreground">
                {processingStatus?.message || 'Processing...'}
              </span>
            </div>
            
            {/* Progress bar - show for batch uploads or if progress is set */}
            {(processingStatus?.total || processingStatus?.progress) && (
              <div className="space-y-2">
                <div className="w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-primary h-2 rounded-full transition-all duration-300" 
                    style={{ 
                      width: processingStatus.total 
                        ? `${((processingStatus.current || 0) / processingStatus.total) * 100}%`
                        : `${processingStatus.progress || 0}%`
                    }}
                  />
                </div>
                {processingStatus.total && (
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>
                      {processingStatus.current || 0} / {processingStatus.total} files
                    </span>
                    {processingStatus.successful !== undefined && processingStatus.failed !== undefined && (
                      <span>
                        {processingStatus.successful} successful, {processingStatus.failed} failed
                      </span>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    )
  }

  const currentStepValid = isStepValid(currentStep)

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[700px] p-0">
        <DialogHeader className="px-6 pt-6 pb-0">
          <DialogTitle>Add New Source</DialogTitle>
          <DialogDescription>
            Add content from links, uploads, or text to your notebooks.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit(onSubmit)}>
          <WizardContainer
            currentStep={currentStep}
            steps={WIZARD_STEPS}
            onStepClick={handleStepClick}
            className="border-0"
          >
            {currentStep === 1 && (
              <SourceTypeStep
                // @ts-expect-error - Type inference issue with zod schema
                control={control}
                register={register}
                // @ts-expect-error - Type inference issue with zod schema
                errors={errors}
              />
            )}
            
            {currentStep === 2 && (
              <NotebooksStep
                notebooks={notebooks}
                selectedNotebooks={selectedNotebooks}
                onToggleNotebook={handleNotebookToggle}
                loading={notebooksLoading}
              />
            )}
            
            {currentStep === 3 && (
              <ProcessingStep
                // @ts-expect-error - Type inference issue with zod schema
                control={control}
                transformations={transformations}
                selectedTransformations={selectedTransformations}
                onToggleTransformation={handleTransformationToggle}
                loading={transformationsLoading}
                settings={settings}
              />
            )}
          </WizardContainer>

          {/* Navigation */}
          <div className="flex justify-between items-center px-6 py-4 border-t border-border bg-muted">
            <Button 
              type="button" 
              variant="outline" 
              onClick={handleClose}
            >
              Cancel
            </Button>

            <div className="flex gap-2">
              {currentStep > 1 && (
                <Button
                  type="button"
                  variant="outline"
                  onClick={handlePrevStep}
                >
                  Back
                </Button>
              )}

              {/* Show Next button on steps 1 and 2, styled as outline/secondary */}
              {currentStep < 3 && (
                <Button
                  type="button"
                  variant="outline"
                  onClick={(e) => handleNextStep(e)}
                  disabled={!currentStepValid}
                >
                  Next
                </Button>
              )}

              {/* Show Done button on all steps, styled as primary */}
              <Button
                type="submit"
                disabled={!currentStepValid || createSource.isPending}
                className="min-w-[120px]"
              >
                {createSource.isPending ? 'Creating...' : 'Done'}
              </Button>
            </div>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}
