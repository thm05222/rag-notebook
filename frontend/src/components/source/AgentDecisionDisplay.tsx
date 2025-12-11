'use client'

import { useState } from 'react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  ChevronDown,
  Brain,
  GitBranch
} from 'lucide-react'
import { AgentThinkingProcess } from '@/lib/types/api'

interface AgentDecisionDisplayProps {
  thinkingProcess?: AgentThinkingProcess
}

export function AgentDecisionDisplay({ thinkingProcess }: AgentDecisionDisplayProps) {
  const [isOpen, setIsOpen] = useState(false)

  // 如果沒有思考過程，不顯示
  if (!thinkingProcess) {
    return null
  }

  // 提取決策相關的步驟
  const decisionSteps = thinkingProcess.steps.filter(step => step.step_type === 'decision')
  
  // 提取 reasoning_trace
  const reasoningTrace = thinkingProcess.reasoning_trace || []

  // 如果沒有決策步驟和推理追踪，不顯示
  if (decisionSteps.length === 0 && reasoningTrace.length === 0) {
    return null
  }

  // 構建簡潔的決策摘要
  const getDecisionSummary = () => {
    const summaries: string[] = []
    
    // 從決策步驟中提取關鍵信息
    decisionSteps.forEach((step) => {
      const metadata = step.metadata || {}
      const action = metadata.action as string || ''
      const toolName = metadata.tool_name as string
      
      if (action) {
        let summary = action
        if (toolName) {
          summary += ` (${toolName})`
        }
        summaries.push(summary)
      } else if (step.content) {
        // 如果沒有 action，使用 content 的前100個字符
        const content = step.content.replace(/^Decision: /, '')
        summaries.push(content.length > 100 ? content.substring(0, 100) + '...' : content)
      }
    })

    return summaries
  }

  const decisionSummaries = getDecisionSummary()
  const hasContent = decisionSummaries.length > 0 || reasoningTrace.length > 0

  if (!hasContent) {
    return null
  }

  // 構建簡潔的預覽文本（收起時顯示）
  const getPreviewText = () => {
    if (decisionSummaries.length > 0) {
      return decisionSummaries[0]
    }
    if (reasoningTrace.length > 0) {
      return reasoningTrace[0]
    }
    return '查看決策過程'
  }

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen} className="mb-2">
      <CollapsibleTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-between text-xs text-muted-foreground hover:text-foreground h-auto py-1.5 px-2"
        >
          <span className="flex items-center gap-2 flex-1 text-left">
            <Brain className="h-3 w-3 flex-shrink-0" />
            <span className="truncate flex-1">
              {isOpen ? '決策過程' : getPreviewText()}
            </span>
            {!isOpen && (decisionSummaries.length > 1 || reasoningTrace.length > 0) && (
              <Badge variant="secondary" className="ml-2 text-xs">
                {decisionSummaries.length + reasoningTrace.length}
              </Badge>
            )}
          </span>
          <ChevronDown className={`h-3 w-3 flex-shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-1 p-2 rounded-md bg-muted/50 border border-dashed text-xs space-y-2">
          {/* 決策摘要 */}
          {decisionSummaries.length > 0 && (
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 text-muted-foreground font-medium">
                <GitBranch className="h-3 w-3" />
                <span>決策歷程</span>
              </div>
              <div className="space-y-1 pl-4">
                {decisionSummaries.map((summary, index) => (
                  <div key={index} className="text-foreground">
                    <span className="text-muted-foreground">#{index + 1}</span> {summary}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 推理追踪 */}
          {reasoningTrace.length > 0 && (
            <div className="space-y-1">
              {decisionSummaries.length > 0 && (
                <div className="border-t pt-2 mt-2" />
              )}
              <div className="flex items-center gap-1.5 text-muted-foreground font-medium">
                <Brain className="h-3 w-3" />
                <span>推理過程</span>
              </div>
              <div className="space-y-1 pl-4">
                {reasoningTrace.map((trace, index) => (
                  <div key={index} className="text-foreground">
                    • {trace}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

