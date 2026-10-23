/* ChatComposer — Quarry design 2026-06-21.
 *
 * Changes vs PR #90:
 *   • The three inline icon buttons (Attach / Photo / Voice) collapse
 *     into a single "+" button that opens a popover with four options:
 *     Attach file · Google Drive · Photo · Voice.
 *   • The send button stays right-aligned with the ArrowUp glyph.
 *   • Optional Clear button still surfaces only when hasHistory.
 *
 * All file upload + voice recording behavior is preserved byte-for-byte;
 * the popover items dispatch to the same handlers as the old buttons.
 * Google Drive item invokes the parent-supplied onOpenDrive callback so
 * ProjectWorkspace can surface its DrivePanel as a modal.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Plus, Paperclip, Camera, Mic, MicOff, RotateCcw, ArrowUp, Cloud, Bot, X,
} from 'lucide-react'
import { getToken } from '../lib/token'
import './ChatComposer.css'

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000'

type UploadLimits = { max_document_bytes: number; allowed_extensions: string[] }

let _limitsCache: UploadLimits | null = null

/** Server-published upload caps, fetched once per session.
 *
 * Returns null if unreachable — an unknown limit must never BLOCK an upload
 * that might succeed, it only forfeits the pre-flight check. */
async function fetchUploadLimits(): Promise<UploadLimits | null> {
  if (_limitsCache) return _limitsCache
  try {
    const res = await fetch(`${API_BASE}/v1/upload-limits`)
    if (!res.ok) return null
    _limitsCache = (await res.json()) as UploadLimits
    return _limitsCache
  } catch {
    return null
  }
}

function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return `${n} B`
  const units = ['B', 'KB', 'MB', 'GB']
  let value = n
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value >= 10 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`
}

export interface AgentOption {
  name: string
  description?: string
  icon?: string
}

/** The pseudo-agent that restores today's automatic routing. */
const AUTO_AGENT: AgentOption = {
  name: 'auto',
  description: 'Automatic — let the assistant route to the right agent.',
}

interface Props {
  /** Returns false if the send was rejected (e.g. a turn is still streaming),
   *  in which case the composer keeps the typed text instead of clearing it. */
  onSend: (text: string) => boolean
  disabled: boolean
  disabledReason?: string
  projectId: string
  onAttached?: (docName: string, docId?: string) => void
  onClear?: () => void
  hasHistory?: boolean
  /** Open the Google Drive picker (parent renders DrivePanel as a modal). */
  onOpenDrive?: () => void
  /** The / agent-picker: available agents (from GET /v1/agents). */
  agents?: AgentOption[]
  /** Currently pinned agent name, or null for automatic routing. */
  pinnedAgent?: string | null
  /** Pin an agent (or null to return to automatic routing). */
  onPinAgent?: (name: string | null) => void
}

export default function ChatComposer({
  onSend, disabled, disabledReason, projectId,
  onAttached, onClear, hasHistory, onOpenDrive,
  agents, pinnedAgent, onPinAgent,
}: Props) {
  const [text, setText] = useState('')
  const [uploading, setUploading] = useState(false)
  const [attachStatus, setAttachStatus] = useState<string | null>(null)
  const [recording, setRecording] = useState(false)
  const [popoverOpen, setPopoverOpen] = useState(false)
  const [slashIdx, setSlashIdx] = useState(0)

  // ── / agent-picker ──────────────────────────────────────────────────────
  // The menu opens when the message is a single "/word" token (no space yet):
  // "/", "/qs", "/quantity". Selecting pins the agent and clears the slash text.
  const slashQuery = /^\/(\S*)$/.exec(text)?.[1]
  const slashMenuItems = useMemo<AgentOption[]>(() => {
    if (slashQuery === undefined) return []
    const q = slashQuery.toLowerCase()
    const pool = [AUTO_AGENT, ...(agents ?? [])]
    return pool.filter(
      (a) => a.name.toLowerCase().includes(q)
        || (a.description ?? '').toLowerCase().includes(q),
    )
  }, [slashQuery, agents])
  const slashOpen = slashQuery !== undefined && slashMenuItems.length > 0

  function pinFromMenu(a: AgentOption) {
    onPinAgent?.(a.name === 'auto' ? null : a.name)
    setText('')
    setSlashIdx(0)
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    textareaRef.current?.focus()
  }

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const cameraInputRef = useRef<HTMLInputElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const popoverRootRef = useRef<HTMLDivElement>(null)

  // Close popover on outside click + Escape
  useEffect(() => {
    if (!popoverOpen) return
    function onDown(ev: MouseEvent) {
      if (popoverRootRef.current && !popoverRootRef.current.contains(ev.target as Node)) {
        setPopoverOpen(false)
      }
    }
    function onKey(ev: KeyboardEvent) {
      if (ev.key === 'Escape') setPopoverOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [popoverOpen])

  function submit() {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    // Clear the box only if the parent ACCEPTED the send. A send rejected
    // because a previous turn is still streaming must keep the user's text so
    // the question isn't silently lost.
    if (onSend(trimmed) === false) return
    setText('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (slashOpen) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSlashIdx((i) => (i + 1) % slashMenuItems.length)
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSlashIdx((i) => (i - 1 + slashMenuItems.length) % slashMenuItems.length)
        return
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault()
        pinFromMenu(slashMenuItems[slashIdx] ?? slashMenuItems[0])
        return
      }
      if (e.key === 'Escape') {
        e.preventDefault()
        setText('')
        return
      }
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setText(e.target.value)
    setSlashIdx(0)
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }

  async function uploadFile(file: File, role = 'other') {
    setUploading(true)
    // Check the file against the server's real limits BEFORE sending it.
    // An oversize upload used to be discovered only by attempting it, and the
    // failure arrived as a bare "Failed to fetch": when the connection dies or
    // the server rejects mid-body, fetch() reports a network error and the HTTP
    // status never reaches JS. That is unactionable, and on a phone it costs
    // minutes of uploading first. A local size check is instant and exact.
    const limits = await fetchUploadLimits()
    if (limits && file.size > limits.max_document_bytes) {
      setAttachStatus(
        `${file.name} is ${formatBytes(file.size)}, over the ` +
          `${formatBytes(limits.max_document_bytes)} limit. Ask an admin to ` +
          `raise MAX_DOC_UPLOAD_SIZE, or split the file.`,
      )
      setUploading(false)
      return
    }
    if (file.size === 0) {
      // A 0-byte file uploads "successfully" and indexes to nothing, which is
      // how documents end up listed but unsearchable.
      setAttachStatus(`${file.name} is empty (0 bytes) — nothing to upload.`)
      setUploading(false)
      return
    }
    setAttachStatus(`Uploading ${file.name} (${formatBytes(file.size)})…`)
    try {
      const token = getToken() || ''
      const fd = new FormData()
      fd.append('file', file)
      // Photos attached via chat are question-context, not corpus
      // material — route them to /v1/chat/analyze-photo so we don't
      // require project ownership (shared admin-approved projects
      // would 404 the documents endpoint).
      const isImage = /\.(jpe?g|png|webp|tiff?|bmp|gif)$/i.test(file.name)
      const endpoint = isImage
        ? `${API_BASE}/v1/chat/analyze-photo`
        : `${API_BASE}/v1/projects/${projectId}/documents`
      if (!isImage) fd.append('role', role)
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: fd,
      })
      if (!res.ok) {
        const errBody = await res.text()
        setAttachStatus(`Upload failed (${res.status}): ${errBody.slice(0, 120)}`)
        return
      }
      const body = await res.json()
      const docName = body?.document?.original_name || body?.filename || file.name
      const docId: string | undefined = body?.document?.id || undefined
      // Safety Observation AI v2 returns tiered OBSERVATIONS, never
      // "violations". Use observations[] verbatim so the LLM also reads
      // them as observations rather than as a verdict.
      const observations = (body?.observations as string[] | undefined) ?? []
      const statusMsg = observations.length
        ? `Attached: ${docName} -- ${observations.join('; ')}`
        : `Attached: ${docName}`
      setAttachStatus(statusMsg)
      onAttached?.(docName, docId)
      const inlineTag = observations.length
        ? `[attached: ${docName} | observations: ${observations.join('; ')}] `
        : `[attached: ${docName}] `
      setText((prev) => (prev ? `${prev}\n` : '') + inlineTag)
      setTimeout(() => setAttachStatus(null), 6000)
    } catch (err) {
      // "Failed to fetch" is what fetch() reports for EVERY transport-level
      // failure — connection reset, timeout, the server closing mid-body, a
      // blocked request. It names none of them, so echoing it verbatim told the
      // user nothing they could act on. Say what it actually means and what to
      // try, and keep the raw text for anyone reading a bug report.
      const raw = (err as Error).message || String(err)
      const transportFailure = /failed to fetch|networkerror|load failed/i.test(raw)
      setAttachStatus(
        transportFailure
          ? `Upload of ${file.name} (${formatBytes(file.size)}) did not reach ` +
            `the server — the connection dropped mid-transfer. Usually the file ` +
            `is too large for the proxy, or the network is too slow to finish. ` +
            `(${raw})`
          : `Upload error: ${raw}`,
      )
    } finally {
      setUploading(false)
    }
  }

  async function startVoiceRecording() {
    if (recording) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr = new MediaRecorder(stream)
      audioChunksRef.current = []
      mr.ondataavailable = (ev) => {
        if (ev.data.size > 0) audioChunksRef.current.push(ev.data)
      }
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        const file = new File([blob], `voice-${Date.now()}.webm`, { type: 'audio/webm' })
        await uploadFile(file, 'other')
      }
      mediaRecorderRef.current = mr
      mr.start()
      setRecording(true)
      setAttachStatus('Recording — click + then Voice again to stop')
    } catch (err) {
      setAttachStatus(`Mic blocked: ${(err as Error).message}`)
    }
  }

  function stopVoiceRecording() {
    mediaRecorderRef.current?.stop()
    mediaRecorderRef.current = null
    setRecording(false)
  }

  function pickAttach() {
    setPopoverOpen(false)
    fileInputRef.current?.click()
  }

  function pickPhoto() {
    setPopoverOpen(false)
    cameraInputRef.current?.click()
  }

  function pickDrive() {
    setPopoverOpen(false)
    onOpenDrive?.()
  }

  function pickVoice() {
    setPopoverOpen(false)
    if (recording) stopVoiceRecording()
    else startVoiceRecording()
  }

  return (
    <div className="chat-composer">
      {attachStatus && (
        <p className="chat-composer__attach-status" aria-live="polite">{attachStatus}</p>
      )}
      <div className="chat-composer__card">
        <input
          ref={fileInputRef}
          type="file"
          multiple
          style={{ display: 'none' }}
          accept=".pdf,.docx,.doc,.xlsx,.xls,.csv,.txt,.md,.png,.jpg,.jpeg,.webp,.tif,.tiff,.dxf,.ifc,.xer,.mp3,.wav,.webm,.mp4"
          onChange={(e) => {
            const files = e.target.files
            if (files) Array.from(files).forEach((f) => uploadFile(f))
            e.target.value = ''
          }}
        />
        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          style={{ display: 'none' }}
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) uploadFile(f)
            e.target.value = ''
          }}
        />

        {pinnedAgent && (
          <div className="chat-composer__pin" title="Pinned agent — the message goes straight to this agent">
            <Bot size={13} />
            <span>{pinnedAgent}</span>
            <button
              type="button"
              className="chat-composer__pin-x"
              aria-label="Unpin agent (back to automatic)"
              onClick={() => onPinAgent?.(null)}
            >
              <X size={12} />
            </button>
          </div>
        )}

        {slashOpen && (
          <div className="chat-composer__slash" role="listbox" aria-label="Pick an agent">
            {slashMenuItems.map((a, i) => (
              <button
                type="button"
                key={a.name}
                role="option"
                aria-selected={i === slashIdx}
                className={
                  'chat-composer__slash-item'
                  + (i === slashIdx ? ' chat-composer__slash-item--active' : '')
                }
                onMouseEnter={() => setSlashIdx(i)}
                onClick={() => pinFromMenu(a)}
              >
                <span className="chat-composer__slash-name">
                  {a.icon ? `${a.icon} ` : ''}{a.name === 'auto' ? 'Auto (default)' : a.name}
                </span>
                {a.description && (
                  <span className="chat-composer__slash-desc">{a.description}</span>
                )}
              </button>
            ))}
          </div>
        )}

        <textarea
          ref={textareaRef}
          className="chat-composer__textarea"
          value={text}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder={pinnedAgent
            ? `Message ${pinnedAgent} directly…  (type / to change)`
            : 'Ask about your project documents...  (type / to pick an agent)'}
          disabled={disabled}
          rows={1}
          aria-label="Chat message"
        />

        <div className="chat-composer__row">
          <div className="chat-composer__plus-wrap" ref={popoverRootRef}>
            <button
              type="button"
              className={`chat-composer__plus${popoverOpen ? ' chat-composer__plus--open' : ''}`}
              title="Attach or record"
              onClick={() => setPopoverOpen((v) => !v)}
              disabled={disabled || uploading}
              aria-haspopup="menu"
              aria-expanded={popoverOpen}
              aria-label="Open attachment menu"
            >
              <Plus size={16} />
            </button>
            {popoverOpen && (
              <div className="chat-composer__popover" role="menu">
                <button
                  type="button"
                  role="menuitem"
                  className="chat-composer__popover-item"
                  onClick={pickAttach}
                >
                  <Paperclip size={14} />
                  <span>Attach file</span>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  className="chat-composer__popover-item"
                  onClick={pickDrive}
                  disabled={!onOpenDrive}
                >
                  <Cloud size={14} />
                  <span>Google Drive</span>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  className="chat-composer__popover-item"
                  onClick={pickPhoto}
                >
                  <Camera size={14} />
                  <span>Photo</span>
                </button>
                <button
                  type="button"
                  role="menuitem"
                  className={
                    'chat-composer__popover-item' +
                    (recording ? ' chat-composer__popover-item--rec' : '')
                  }
                  onClick={pickVoice}
                >
                  {recording ? <MicOff size={14} /> : <Mic size={14} />}
                  <span>{recording ? 'Stop recording' : 'Voice'}</span>
                </button>
              </div>
            )}
          </div>

          {onClear && hasHistory && (
            <button
              type="button"
              className="chat-composer__tool"
              title="Clear chat history (cannot be undone)"
              onClick={() => onClear()}
              disabled={disabled || uploading}
              aria-label="Clear chat history"
            >
              <RotateCcw size={16} />
            </button>
          )}

          <span className="chat-composer__spacer" aria-hidden="true" />

          <span className="chat-composer__hint">Enter to send · Shift+Enter newline</span>

          <button
            type="button"
            className="chat-composer__send"
            onClick={submit}
            disabled={disabled || !text.trim()}
            aria-label="Send message"
            title={disabled && disabledReason ? disabledReason : undefined}
          >
            <ArrowUp size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}
