"use client"

import { useState, useEffect } from "react"
import { Control, FieldErrors, UseFormRegister, useWatch } from "react-hook-form"
import { FileIcon, LinkIcon, FileTextIcon, X } from "lucide-react"
import { FormSection } from "@/components/ui/form-section"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Controller } from "react-hook-form"

interface CreateSourceFormData {
  type: 'link' | 'upload' | 'text'
  title?: string
  url?: string
  content?: string
  file?: FileList | File
  notebooks?: string[]
  transformations?: string[]
  embed: boolean
  build_pageindex: boolean
  async_processing: boolean
}

const SOURCE_TYPES = [
  {
    value: 'link' as const,
    label: 'Link',
    icon: LinkIcon,
    description: 'Add a web page or URL',
  },
  {
    value: 'upload' as const,
    label: 'Upload',
    icon: FileIcon,
    description: 'Upload a document or file',
  },
  {
    value: 'text' as const,
    label: 'Text',
    icon: FileTextIcon,
    description: 'Add text content directly',
  },
]

interface SourceTypeStepProps {
  control: Control<CreateSourceFormData>
  register: UseFormRegister<CreateSourceFormData>
  errors: FieldErrors<CreateSourceFormData>
}

export function SourceTypeStep({ control, register, errors }: SourceTypeStepProps) {
  // Watch the selected type to make title conditional
  const selectedType = useWatch({ control, name: 'type' })
  const watchedFile = useWatch({ control, name: 'file' })
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])

  // Sync selectedFiles with watchedFile when it changes
  useEffect(() => {
    if (watchedFile) {
      if (watchedFile instanceof FileList) {
        setSelectedFiles(Array.from(watchedFile))
      } else if (watchedFile instanceof File) {
        setSelectedFiles([watchedFile])
      } else {
        setSelectedFiles([])
      }
    } else {
      setSelectedFiles([])
    }
  }, [watchedFile])

  // Convert FileList to array and update state when files change
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      const fileArray = Array.from(files)
      setSelectedFiles(fileArray)
    } else {
      setSelectedFiles([])
    }
  }

  // Remove a file from the list
  const handleRemoveFile = (index: number) => {
    const newFiles = selectedFiles.filter((_, i) => i !== index)
    setSelectedFiles(newFiles)
    
    // Create a new FileList-like object
    const dt = new DataTransfer()
    newFiles.forEach(file => dt.items.add(file))
    
    // Update the input element
    const input = document.getElementById('file') as HTMLInputElement
    if (input) {
      input.files = dt.files
      // Trigger change event to update form
      const event = new Event('change', { bubbles: true })
      input.dispatchEvent(event)
    }
  }

  // Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  return (
    <div className="space-y-6">
      <FormSection
        title="Source Type"
        description="Choose how you want to add your content"
      >
        <Controller
          control={control}
          name="type"
          render={({ field }) => (
            <Tabs 
              value={field.value || ''} 
              onValueChange={(value) => field.onChange(value as 'link' | 'upload' | 'text')}
              className="w-full"
            >
              <TabsList className="grid w-full grid-cols-3">
                {SOURCE_TYPES.map((type) => {
                  const Icon = type.icon
                  return (
                    <TabsTrigger key={type.value} value={type.value} className="gap-2">
                      <Icon className="h-4 w-4" />
                      {type.label}
                    </TabsTrigger>
                  )
                })}
              </TabsList>
              
              {SOURCE_TYPES.map((type) => (
                <TabsContent key={type.value} value={type.value} className="mt-4">
                  <p className="text-sm text-muted-foreground mb-4">{type.description}</p>
                  
                  {/* Type-specific fields */}
                  {type.value === 'link' && (
                    <div>
                      <Label htmlFor="url" className="mb-2 block">URL *</Label>
                      <Input
                        id="url"
                        {...register('url')}
                        placeholder="https://example.com/article"
                        type="url"
                      />
                      {errors.url && (
                        <p className="text-sm text-destructive mt-1">{errors.url.message}</p>
                      )}
                    </div>
                  )}
                  
                  {type.value === 'upload' && (
                    <div className="w-full">
                      <Label htmlFor="file" className="mb-2 block">
                        {selectedFiles.length > 0 
                          ? `Files * (${selectedFiles.length} selected)`
                          : 'File(s) *'}
                      </Label>
                      <div className="w-full overflow-x-auto">
                        <Input
                          id="file"
                          type="file"
                          multiple
                          accept=".pdf,.doc,.docx,.txt,.md,.epub"
                          {...register('file', {
                            onChange: handleFileChange
                          })}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Supported formats: PDF, DOC, DOCX, TXT, MD, EPUB. You can select multiple files.
                      </p>
                      
                      {/* Display selected files */}
                      {selectedFiles.length > 0 && (
                        <div className="mt-3 space-y-2 w-full">
                          <p className="text-sm font-medium">Selected files:</p>
                          <div className="space-y-1 max-h-48 overflow-y-auto w-full">
                            {selectedFiles.map((file, index) => (
                              <div
                                key={index}
                                className="flex items-center justify-between gap-2 p-2 bg-muted rounded-md text-sm w-full max-w-full min-w-0"
                              >
                                <div className="flex-1 min-w-0 overflow-hidden">
                                  <p 
                                    className="font-medium truncate" 
                                    title={file.name}
                                  >
                                    {file.name}
                                  </p>
                                  <p className="text-xs text-muted-foreground truncate">
                                    {formatFileSize(file.size)}
                                  </p>
                                </div>
                                <Button
                                  type="button"
                                  variant="ghost"
                                  size="icon"
                                  className="h-6 w-6 shrink-0"
                                  onClick={() => handleRemoveFile(index)}
                                >
                                  <X className="h-4 w-4" />
                                </Button>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {errors.file && (
                        <p className="text-sm text-destructive mt-1">{errors.file.message}</p>
                      )}
                    </div>
                  )}
                  
                  {type.value === 'text' && (
                    <div>
                      <Label htmlFor="content" className="mb-2 block">Text Content *</Label>
                      <Textarea
                        id="content"
                        {...register('content')}
                        placeholder="Paste or type your content here..."
                        rows={6}
                      />
                      {errors.content && (
                        <p className="text-sm text-destructive mt-1">{errors.content.message}</p>
                      )}
                    </div>
                  )}
                </TabsContent>
              ))}
            </Tabs>
          )}
        />
        {errors.type && (
          <p className="text-sm text-destructive mt-1">{errors.type.message}</p>
        )}
      </FormSection>

      <FormSection
        title={selectedType === 'text' ? "Title *" : "Title (optional)"}
        description={selectedType === 'text'
          ? "A title is required for text content"
          : "If left empty, a title will be generated from the content"
        }
      >
        <Input
          id="title"
          {...register('title')}
          placeholder="Give your source a descriptive title"
        />
        {errors.title && (
          <p className="text-sm text-destructive mt-1">{errors.title.message}</p>
        )}
      </FormSection>
    </div>
  )
}
