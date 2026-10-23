/* ChatBubble — one message rendered as a chat bubble.
 *
 * Behaviour preserved from the pre-redesign ProjectWorkspace bubble:
 *   • user vs assistant role (right vs left aligned via CSS)
 *   • error variant with the alert SVG + role="alert"
 *   • streaming dots animation when content is empty
 *   • tool-status mini-bubble shown above content
 *   • markdown rendering with ReactMarkdown + remark-gfm
 *   • download button on completed assistant messages
 *
 * REMOVED: the inline `<details>` sources footer. Sources now live in
 * RightPanel/SourcesList for the LATEST answer (operator spec — moved
 * to right panel for visibility).
 */
import { memo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AlertTriangle, Download } from 'lucide-react'
import './ChatBubble.css'

import type { ChatMessage, ExportDescriptor } from './types'
import { plainMetadata } from './userFacingText'

interface Props {
  message: ChatMessage
  onDownload?: (format: 'docx' | 'xlsx') => void
  onExport?: (descriptor: ExportDescriptor) => void
}

// Memoised: every streamed token calls setMessages on the parent, which
// re-renders the whole list. Without memo, each tick re-parses ReactMarkdown
// for every prior bubble — janky in long sessions, and a hard main-thread
// freeze once the conversation is large (render cost is O(messages) PER token).
// Bubbles are immutable once settled, so we skip them — but see propsEqual: the
// default shallow compare could NOT skip them, because ChatList builds a fresh
// onDownload closure for every bubble on every render, so that prop always
// "changed". We compare the fields a bubble actually renders instead.
function ChatBubble({ message, onDownload, onExport }: Props) {
  const isUser = message.role === 'user'
  const exports = message.exports ?? []
  const toolStatus = message.toolStatus ? plainMetadata(message.toolStatus) : ''

  if (message.error) {
    return (
      <div className="chat-bubble chat-bubble--error" role="alert">
        <AlertTriangle size={18} className="chat-bubble__error-icon" />
        <span>{message.content || 'Something went wrong. Please try again.'}</span>
      </div>
    )
  }

  return (
    <div className={`chat-bubble chat-bubble--${message.role}`}>
      {!isUser && <div className="chat-bubble__avatar" aria-hidden="true" title="The SHovel">TSH</div>}

      <div className="chat-bubble__body">
        {toolStatus && (
          <div className="chat-bubble__tool-status" aria-live="polite">
            {toolStatus}
          </div>
        )}

        <div className="chat-bubble__content">
          {isUser ? (
            <span className="chat-bubble__text">{message.content}</span>
          ) : message.streaming && !message.content ? (
            <span className="chat-bubble__typing" aria-label="Assistant is thinking">
              <span className="chat-bubble__typing-dot" />
              <span className="chat-bubble__typing-dot" />
              <span className="chat-bubble__typing-dot" />
            </span>
          ) : (
            <div className="chat-bubble__markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
              {message.streaming && <span className="chat-bubble__cursor" aria-hidden="true" />}
            </div>
          )}
        </div>

        {message.role === 'assistant' && !message.streaming && message.content && (onDownload || exports.length > 0) && (
          <div className="chat-bubble__actions">
            {onDownload && (
              <>
                <button
                  type="button"
                  className="chat-bubble__download"
                  onClick={() => onDownload('docx')}
                  title="Download this message as a Word document"
                  aria-label="Download as Word document"
                >
                  <Download size={13} />
                  <span>Word</span>
                </button>
                <button
                  type="button"
                  className="chat-bubble__download"
                  onClick={() => onDownload('xlsx')}
                  title="Download this message as an Excel workbook (tables become worksheets)"
                  aria-label="Download as Excel workbook"
                >
                  <Download size={13} />
                  <span>Excel</span>
                </button>
              </>
            )}
            {exports.map((exp, i) => (
              <button
                key={`${exp.endpoint}-${i}`}
                type="button"
                className="chat-bubble__download chat-bubble__download--export"
                onClick={() => onExport?.(exp)}
                title={`Generate and download: ${exp.label}`}
                aria-label={`Download ${exp.label}`}
              >
                <Download size={13} />
                <span>{exp.label}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {isUser && <div className="chat-bubble__avatar chat-bubble__avatar--user" aria-hidden="true">U</div>}
    </div>
  )
}

/* Re-render a bubble only when something it actually renders changed. The two
 * handlers (onDownload, onExport) back buttons that appear ONLY on a settled
 * assistant message, and that message re-renders when it settles (streaming
 * flips false), so a fresh closure always lands before the button is clickable —
 * their identity never changes render output. Ignoring it here is what lets the
 * memo skip prior bubbles during streaming: O(1) work per token instead of
 * O(messages), which is the long-conversation freeze fix. */
function propsEqual(prev: Props, next: Props): boolean {
  const a = prev.message
  const b = next.message
  return (
    a.id === b.id &&
    a.role === b.role &&
    a.content === b.content &&
    !!a.streaming === !!b.streaming &&
    !!a.error === !!b.error &&
    a.toolStatus === b.toolStatus &&
    a.exports === b.exports &&
    !!prev.onDownload === !!next.onDownload
  )
}

export default memo(ChatBubble, propsEqual)
