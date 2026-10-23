import { useState, type CSSProperties } from 'react'
import { getToken } from '../lib/token'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

export interface ActivityRow {
  id: string
  name?: string
  duration_days?: number
  duration?: number
  early_start_day?: number
  early_finish_day?: number
  total_float_days?: number
  critical?: boolean
  wbs_phase?: string
  phase?: string
  predecessors?: string[]
  resources?: string[]
}

interface ScheduleBuilderProps {
  projectId: string
  conversationId?: string
}

type Mode = 'brief' | 'documents' | 'activities'

async function downloadSchedule(
  endpoint: string,
  payload: Record<string, unknown>,
  filenameFallback: string,
): Promise<void> {
  const token = getToken()
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: {
      Authorization: token ? `Bearer ${token}` : '',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Export failed (${res.status}): ${text.slice(0, 200)}`)
  }

  const blob = await res.blob()
  const cd = res.headers.get('Content-Disposition') || ''
  const m = /filename="?([^";]+)"?/.exec(cd)
  const filename = m?.[1] || filenameFallback

  const a = document.createElement('a')
  const url = URL.createObjectURL(blob)
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export default function ScheduleBuilder({ projectId, conversationId }: ScheduleBuilderProps) {
  const [mode, setMode] = useState<Mode>('brief')
  const [brief, setBrief] = useState('')
  const [targetCount, setTargetCount] = useState(200)
  const [projectType, setProjectType] = useState('')
  const [startDate, setStartDate] = useState('')
  const [dayRate, setDayRate] = useState('')
  const [crewPerTrade, setCrewPerTrade] = useState(4)
  const [currency, setCurrency] = useState('SAR')
  const [documentIds, setDocumentIds] = useState('')
  const [activitiesJson, setActivitiesJson] = useState('[]')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const sharedPayload: Record<string, unknown> = {
    currency,
    crew_per_trade: crewPerTrade,
    ...(startDate ? { start_date: startDate } : {}),
    ...(dayRate && !Number.isNaN(parseFloat(dayRate))
      ? { day_rate: parseFloat(dayRate) }
      : {}),
  }

  const generateFromBrief = async (): Promise<void> => {
    setLoading(true)
    setError(null)
    try {
      await downloadSchedule(
        `/v1/projects/${projectId}/export/schedule-from-brief`,
        {
          ...sharedPayload,
          brief,
          target_count: targetCount,
          project_type: projectType || undefined,
          ...(conversationId ? { conversation_id: conversationId } : {}),
        },
        'schedule.xlsx',
      )
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const generateFromDocuments = async (): Promise<void> => {
    setLoading(true)
    setError(null)
    const ids = documentIds
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
    try {
      await downloadSchedule(
        `/v1/projects/${projectId}/export/schedule-from-document`,
        {
          ...sharedPayload,
          brief,
          document_ids: ids,
          target_count: targetCount,
          project_type: projectType || undefined,
        },
        'schedule.xlsx',
      )
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const exportActivities = async (): Promise<void> => {
    setLoading(true)
    setError(null)
    let activities: ActivityRow[]
    try {
      activities = JSON.parse(activitiesJson) as ActivityRow[]
    } catch {
      setError('Activities JSON is invalid')
      setLoading(false)
      return
    }
    if (!Array.isArray(activities) || activities.length === 0) {
      setError('Activities array must be non-empty')
      setLoading(false)
      return
    }
    try {
      await downloadSchedule(
        `/v1/projects/${projectId}/export/cost-schedule`,
        { ...sharedPayload, activities },
        'cost_loaded_schedule.xlsx',
      )
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const tab = (key: Mode, label: string) => (
    <button
      type="button"
      key={key}
      onClick={() => setMode(key)}
      disabled={loading}
      style={{
        ...styles.tab,
        ...(mode === key ? styles.tabActive : {}),
      }}
      aria-pressed={mode === key}
    >
      {label}
    </button>
  )

  return (
    <section style={styles.container} aria-label="Schedule builder">
      <h2 style={styles.heading}>Schedule Builder</h2>

      <div style={styles.tabs} role="tablist">
        {tab('brief', 'From Brief')}
        {tab('documents', 'From Documents')}
        {tab('activities', 'From Activities')}
      </div>

      <div style={styles.form}>
        {mode !== 'activities' && (
          <>
            <label style={styles.label}>
              Project brief
              <textarea
                value={brief}
                onChange={(e) => setBrief(e.target.value)}
                placeholder="Describe the project scope, deliverables, and constraints..."
                rows={4}
                style={styles.input}
              />
            </label>
          </>
        )}

        {mode === 'documents' && (
          <label style={styles.label}>
            Document IDs (comma separated)
            <input
              type="text"
              value={documentIds}
              onChange={(e) => setDocumentIds(e.target.value)}
              placeholder="doc-id-1, doc-id-2"
              style={styles.input}
            />
          </label>
        )}

        {mode === 'activities' && (
          <label style={styles.label}>
            Activities JSON
            <textarea
              value={activitiesJson}
              onChange={(e) => setActivitiesJson(e.target.value)}
              placeholder='[{"id":"A100","name":"Concrete","duration_days":5,"wbs_phase":"Substructure"}]'
              rows={8}
              style={styles.input}
            />
          </label>
        )}

        <div style={styles.row}>
          <label style={styles.label}>
            Target count
            <input
              type="number"
              min={20}
              max={1000}
              value={targetCount}
              onChange={(e) => setTargetCount(parseInt(e.target.value || '0', 10))}
              disabled={mode === 'activities'}
              style={styles.input}
            />
          </label>

          <label style={styles.label}>
            Project type
            <input
              type="text"
              value={projectType}
              onChange={(e) => setProjectType(e.target.value)}
              placeholder="e.g. data-center"
              disabled={mode === 'activities'}
              style={styles.input}
            />
          </label>
        </div>

        <div style={styles.row}>
          <label style={styles.label}>
            Start date
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              style={styles.input}
            />
          </label>

          <label style={styles.label}>
            Day rate (optional)
            <input
              type="number"
              value={dayRate}
              onChange={(e) => setDayRate(e.target.value)}
              placeholder="SAR / day"
              style={styles.input}
            />
          </label>

          <label style={styles.label}>
            Currency
            <input
              type="text"
              value={currency}
              onChange={(e) => setCurrency(e.target.value)}
              style={styles.input}
            />
          </label>
        </div>

        <label style={styles.label}>
          Crew per trade
          <input
            type="number"
            min={1}
            value={crewPerTrade}
            onChange={(e) => setCrewPerTrade(parseInt(e.target.value || '1', 10))}
            style={styles.input}
          />
        </label>

        {error && <div style={styles.error}>{error}</div>}

        <button
          type="button"
          onClick={
            mode === 'brief'
              ? generateFromBrief
              : mode === 'documents'
                ? generateFromDocuments
                : exportActivities
          }
          disabled={loading}
          style={styles.button}
        >
          {loading
            ? 'Working...'
            : mode === 'activities'
              ? 'Export Cost-Loaded Schedule'
              : 'Generate Schedule'}
        </button>
      </div>
    </section>
  )
}

const styles: Record<string, CSSProperties> = {
  container: {
    padding: '1rem',
    border: '1px solid var(--color-border, #d1d5db)',
    borderRadius: '0.5rem',
    backgroundColor: 'var(--color-surface, #ffffff)',
    color: 'var(--color-text, #111827)',
  },
  heading: {
    marginTop: 0,
    marginBottom: '0.75rem',
    fontSize: '1.25rem',
  },
  tabs: {
    display: 'flex',
    gap: '0.5rem',
    marginBottom: '1rem',
  },
  tab: {
    flex: 1,
    padding: '0.5rem',
    border: '1px solid var(--color-border, #d1d5db)',
    borderRadius: '0.25rem',
    backgroundColor: 'var(--color-surface, #ffffff)',
    cursor: 'pointer',
  },
  tabActive: {
    backgroundColor: 'var(--color-primary, #1f3864)',
    color: '#ffffff',
    borderColor: 'var(--color-primary, #1f3864)',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
  },
  row: {
    display: 'flex',
    gap: '0.75rem',
    flexWrap: 'wrap',
  },
  label: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
    fontSize: '0.875rem',
    fontWeight: 500,
    flex: '1 1 200px',
  },
  input: {
    padding: '0.5rem',
    border: '1px solid var(--color-border, #d1d5db)',
    borderRadius: '0.25rem',
    fontSize: '0.875rem',
    fontFamily: 'inherit',
  },
  button: {
    marginTop: '0.5rem',
    padding: '0.625rem 1rem',
    border: 'none',
    borderRadius: '0.25rem',
    backgroundColor: 'var(--color-primary, #1f3864)',
    color: '#ffffff',
    cursor: 'pointer',
    fontSize: '0.875rem',
    fontWeight: 600,
  },
  error: {
    color: '#b91c1c',
    fontSize: '0.875rem',
  },
}
