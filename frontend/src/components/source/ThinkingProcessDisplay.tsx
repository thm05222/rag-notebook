'use client'

import { useState, useEffect } from 'react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  ChevronDown,
  Brain,
  Search,
  CheckCircle,
  Wrench,
  Sparkles,
  Eye,
  EyeOff
} from 'lucide-react'
import { AgentThinkingProcess } from '@/lib/types/api'
import { formatDistanceToNow } from 'date-fns'

interface ThinkingProcessDisplayProps {
  thinkingProcess: AgentThinkingProcess
  /** 是否預設展開（用於即時顯示思考過程） */
  defaultOpen?: boolean
}

export function ThinkingProcessDisplay({ thinkingProcess, defaultOpen = false }: ThinkingProcessDisplayProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen)
  const [showDetails, setShowDetails] = useState(false)

  // [關鍵修復] 同步 defaultOpen prop 變化到 isOpen 狀態
  // useState 只在首次掛載時使用初始值，後續 prop 變化不會自動更新狀態
  useEffect(() => {
    if (defaultOpen) {
      setIsOpen(true)
    }
  }, [defaultOpen])

  // 調試：記錄組件渲染狀態
  useEffect(() => {
    console.log('[ThinkingProcessDisplay] Rendered with:', {
      stepsCount: thinkingProcess?.steps?.length || 0,
      isOpen,
      defaultOpen
    })
  }, [thinkingProcess?.steps?.length, isOpen, defaultOpen])

  if (!thinkingProcess || thinkingProcess.steps.length === 0) {
    console.log('[ThinkingProcessDisplay] Returning null - no steps')
    return null
  }

  const getStepIcon = (stepType: string) => {
    switch (stepType) {
      case 'decision':
        return <Brain className="h-4 w-4" />
      case 'tool_call':
        return <Wrench className="h-4 w-4" />
      case 'search':
        return <Search className="h-4 w-4" />
      case 'evaluation':
        return <CheckCircle className="h-4 w-4" />
      case 'refinement':
        return <Sparkles className="h-4 w-4" />
      case 'synthesis':
        return <Sparkles className="h-4 w-4" />
      default:
        return <Brain className="h-4 w-4" />
    }
  }

  const getStepColor = (stepType: string) => {
    switch (stepType) {
      case 'decision':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
      case 'tool_call':
        return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
      case 'search':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'evaluation':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
      case 'refinement':
        return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
      case 'synthesis':
        return 'bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-200'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const formatTimestamp = (timestamp: number) => {
    try {
      return formatDistanceToNow(new Date(timestamp * 1000), { addSuffix: true })
    } catch {
      return 'just now'
    }
  }

  console.log('[ThinkingProcessDisplay] Rendering Collapsible with isOpen:', isOpen)

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="mt-2">
      <CollapsibleTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-between text-xs text-muted-foreground hover:text-foreground"
        >
          <span className="flex items-center gap-2">
            <Brain className="h-3 w-3" />
            Thinking Process
            <Badge variant="secondary" className="ml-2">
              {thinkingProcess.steps.length} steps
            </Badge>
            {thinkingProcess.total_iterations > 0 && (
              <Badge variant="outline" className="ml-1">
                {thinkingProcess.total_iterations} iterations
              </Badge>
            )}
          </span>
          <ChevronDown className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <Card className="mt-2 border-dashed">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Agent Thinking Process</CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowDetails(!showDetails)}
                className="h-7"
              >
                {showDetails ? (
                  <EyeOff className="h-3 w-3" />
                ) : (
                  <Eye className="h-3 w-3" />
                )}
              </Button>
            </div>
            <div className="flex flex-wrap gap-2 mt-2 text-xs text-muted-foreground">
              {thinkingProcess.total_tool_calls > 0 && (
                <span>Tools: {thinkingProcess.total_tool_calls}</span>
              )}
              {thinkingProcess.search_count > 0 && (
                <span>Searches: {thinkingProcess.search_count}</span>
              )}
              {thinkingProcess.evaluation_scores && (
                <span>
                  Score: {(thinkingProcess.evaluation_scores.combined_score || 0).toFixed(2)}
                </span>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            {/* Steps Timeline */}
            <div className="space-y-3">
              {thinkingProcess.steps.map((step, index) => (
                <div key={index} className="flex gap-3">
                  {/* Timeline line */}
                  {index < thinkingProcess.steps.length - 1 && (
                    <div className="flex flex-col items-center">
                      <div className={`rounded-full p-1.5 ${getStepColor(step.step_type)}`}>
                        {getStepIcon(step.step_type)}
                      </div>
                      <div className="w-0.5 h-full bg-border mt-1" />
                    </div>
                  )}
                  {index === thinkingProcess.steps.length - 1 && (
                    <div className="flex flex-col items-center">
                      <div className={`rounded-full p-1.5 ${getStepColor(step.step_type)}`}>
                        {getStepIcon(step.step_type)}
                      </div>
                    </div>
                  )}

                  {/* Step content */}
                  <div className="flex-1 pb-2">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge
                        variant="outline"
                        className={`text-xs ${getStepColor(step.step_type)}`}
                      >
                        {step.step_type}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {formatTimestamp(step.timestamp)}
                      </span>
                    </div>
                    <p className="text-sm">{step.content}</p>
                    {showDetails && step.metadata && (
                      <div className="mt-2 p-2 bg-muted rounded text-xs">
                        <pre className="whitespace-pre-wrap break-words">
                          {JSON.stringify(step.metadata, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Reasoning Trace */}
            {thinkingProcess.reasoning_trace && thinkingProcess.reasoning_trace.length > 0 && (
              <div className="mt-4 pt-4 border-t">
                <h4 className="text-xs font-medium mb-2">Reasoning Trace</h4>
                <div className="space-y-1">
                  {thinkingProcess.reasoning_trace.map((trace, index) => (
                    <p key={index} className="text-xs text-muted-foreground">
                      • {trace}
                    </p>
                  ))}
                </div>
              </div>
            )}

            {/* Evaluation Scores */}
            {thinkingProcess.evaluation_scores && showDetails && (
              <div className="mt-4 pt-4 border-t">
                <h4 className="text-xs font-medium mb-2">Evaluation Scores</h4>
                <div className="space-y-1">
                  {Object.entries(thinkingProcess.evaluation_scores).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-xs">
                      <span className="text-muted-foreground">{key}:</span>
                      <span className="font-medium">{value.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </CollapsibleContent>
    </Collapsible>
  )
}

