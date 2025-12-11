'use client'

import { EyeOff, Lightbulb, FileText } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

export type DefaultContextMode = 'off' | 'insights' | 'full'

interface DefaultContextModeToggleProps {
  mode: DefaultContextMode
  onChange: (mode: DefaultContextMode) => void
  className?: string
}

const MODE_CONFIG = {
  off: {
    icon: EyeOff,
    label: 'Not included',
    description: 'Sources will not be included in chat context by default',
    color: 'text-muted-foreground',
    bgColor: 'hover:bg-muted'
  },
  insights: {
    icon: Lightbulb,
    label: 'Insights only',
    description: 'Only insights will be used in chat context by default',
    color: 'text-amber-600',
    bgColor: 'hover:bg-amber-50'
  },
  full: {
    icon: FileText,
    label: 'Full content',
    description: 'Full source content will be used in chat context by default',
    color: 'text-primary',
    bgColor: 'hover:bg-primary/10'
  }
} as const

export function DefaultContextModeToggle({ mode, onChange, className }: DefaultContextModeToggleProps) {
  const config = MODE_CONFIG[mode]
  const Icon = config.icon

  // Available modes in cycle order
  const availableModes: DefaultContextMode[] = ['off', 'insights', 'full']

  const handleClick = () => {
    // Cycle to next mode
    const currentIndex = availableModes.indexOf(mode)
    const nextIndex = (currentIndex + 1) % availableModes.length
    onChange(availableModes[nextIndex])
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            size="default"
            className={cn(
              'flex items-center gap-2 transition-colors',
              config.bgColor,
              className
            )}
            onClick={handleClick}
          >
            <Icon className={cn('h-4 w-4', config.color)} />
            <span className={cn('text-sm', config.color)}>{config.label}</span>
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p className="text-xs font-medium">{config.label}</p>
          <p className="text-[10px] text-muted-foreground mt-1 max-w-xs">
            {config.description}
          </p>
          <p className="text-[10px] text-muted-foreground mt-1">
            Click to cycle
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

