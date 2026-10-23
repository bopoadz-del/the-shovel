import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
}

/**
 * App-wide error boundary. Without this, any uncaught render exception
 * (e.g. ReactMarkdown choking on malformed LLM output) blanks the whole SPA
 * to a white screen with no recovery. Catches the error, shows a controlled
 * fallback, and offers a reload — it never surfaces the raw error text.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Log for diagnostics; do NOT show raw error/stack to the user.
    console.error('Unhandled render error:', error, info.componentStack)
  }

  handleReload = (): void => {
    this.setState({ hasError: false })
    window.location.reload()
  }

  render(): ReactNode {
    if (!this.state.hasError) return this.props.children
    return (
      <div
        style={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '16px',
          background: 'var(--bg)',
          color: 'var(--text)',
          fontFamily: 'var(--font-sans)',
          padding: '24px',
          textAlign: 'center',
        }}
      >
        <span style={{ fontSize: '15px', fontWeight: 600 }}>
          Something went wrong.
        </span>
        <span
          style={{
            fontSize: '13px',
            color: 'var(--text-muted)',
            maxWidth: '420px',
          }}
        >
          The page hit an unexpected error. Reloading usually fixes it; your work
          on the server is not affected.
        </span>
        <button
          type="button"
          onClick={this.handleReload}
          style={{
            padding: '8px 18px',
            borderRadius: '8px',
            border: '1px solid var(--border)',
            background: 'var(--accent, #c2603f)',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '13px',
          }}
        >
          Reload
        </button>
      </div>
    )
  }
}
