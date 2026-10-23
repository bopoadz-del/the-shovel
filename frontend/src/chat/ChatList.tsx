/* ChatList — scrollable message thread.
 *
 * Owns the auto-scroll-to-bottom behaviour and the empty state with
 * suggestion chips. Bubbles render via ChatBubble. The download handler
 * walks assistant messages in order so the server-side per-index lookup
 * works the same as before the redesign.
 */
import { useEffect, useRef } from 'react'
import ChatBubble from './ChatBubble'
import type { ChatMessage, ExportDescriptor } from './types'
import './ChatList.css'

interface Props {
  messages: ChatMessage[]
  documentCount: number
  onSuggestion: (text: string) => void
  suggestionsDisabled: boolean
  onDownloadMessage?: (assistantIndex: number, format: 'docx' | 'xlsx') => void
  onExport?: (descriptor: ExportDescriptor) => void
}

const EMPTY_SUGGESTIONS = [
  'What is the IT load specification?',
  'Summarise the key BOQ items',
  'What are the main project risks?',
]

export default function ChatList({
  messages,
  documentCount,
  onSuggestion,
  suggestionsDisabled,
  onDownloadMessage,
  onExport,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    const docLabel =
      documentCount === 0
        ? 'No documents indexed yet for this project'
        : documentCount === 1
          ? 'I have access to 1 document in this project'
          : `I have access to ${documentCount} documents in this project`
    return (
      <div className="chat-list__empty">
        <p className="chat-list__empty-title">Ask anything about your project</p>
        <p className="chat-list__empty-hint">{docLabel}</p>
        <div className="chat-list__empty-chips" role="group" aria-label="Suggested questions">
          {EMPTY_SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              className="chat-list__chip"
              disabled={suggestionsDisabled}
              onClick={() => onSuggestion(s)}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    )
  }

  // Assistant-turn index per message, computed BEFORE render rather than by
  // mutating a counter inside .map(). The counter version worked, but mutating
  // a variable while rendering is the pattern React's compiler lint rejects
  // ("Cannot reassign variable after render completes") — and it silently
  // breaks the moment this list is memoised or rendered out of order.
  //
  // Numbering is unchanged: EVERY assistant turn advances the index, including
  // streaming/error/empty ones, so `onDownloadMessage(idx)` still addresses
  // the same turn the caller expects.
  const assistantIndexById = new Map<string, number>()
  let assistantSeen = 0
  for (const msg of messages) {
    if (msg.role === 'assistant') {
      assistantIndexById.set(msg.id, assistantSeen)
      assistantSeen += 1
    }
  }

  return (
    <div className="chat-list" role="log" aria-live="polite" aria-label="Conversation">
      {messages.map((msg) => {
        let downloadHandler: ((format: 'docx' | 'xlsx') => void) | undefined
        const isDownloadable =
          msg.role === 'assistant' && !msg.streaming && !msg.error && msg.content
        if (isDownloadable && onDownloadMessage) {
          const idx = assistantIndexById.get(msg.id) ?? 0
          downloadHandler = (format) => onDownloadMessage(idx, format)
        }
        return <ChatBubble key={msg.id} message={msg} onDownload={downloadHandler} onExport={onExport} />
      })}
      <div ref={bottomRef} />
    </div>
  )
}
