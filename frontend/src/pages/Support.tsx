import { useState, type FormEvent } from 'react'
import { useSupportEmail } from '../lib/support'
import './support.css'

/* Public support form. Posts to the unauthenticated /v1/support endpoint;
 * the app-wide rate limiter guards abuse. */
export default function Support() {
  const supportEmail = useSupportEmail()
  const [message, setMessage] = useState('')
  const [contact, setContact] = useState('')
  const [state, setState] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle')
  const [error, setError] = useState('')

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!message.trim()) return
    setState('sending')
    setError('')
    try {
      const res = await fetch('/v1/support', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, contact_email: contact.trim() || null }),
      })
      if (!res.ok) {
        setState('error')
        setError('Your message could not be sent right now. Please try again shortly.')
        return
      }
      setState('sent')
      setMessage('')
      setContact('')
    } catch {
      setState('error')
      setError('Your message could not be sent right now. Please try again shortly.')
    }
  }

  return (
    <main className="support-page">
      <div className="support-card">
        <h1>Support</h1>
        <p>
          Tell us what you need help with and we will get back to you by email.
        </p>
        {supportEmail ? (
          <p className="support-direct">
            Or email us directly at{' '}
            <a href={`mailto:${supportEmail}`}>{supportEmail}</a>.
          </p>
        ) : null}
        {state === 'sent' ? (
          <p className="support-ok" role="status">
            Sent — thank you. We will reply to the email you gave (or to your
            account email if you left it blank).
          </p>
        ) : (
          <form onSubmit={submit} className="support-form">
            <label htmlFor="support-message">Message</label>
            <textarea
              id="support-message"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              maxLength={5000}
              required
              rows={6}
              placeholder="Describe the problem…"
            />
            <label htmlFor="support-contact">Email (optional)</label>
            <input
              id="support-contact"
              type="email"
              value={contact}
              onChange={(e) => setContact(e.target.value)}
              maxLength={320}
              placeholder="you@example.com"
            />
            {state === 'error' && (
              <p className="support-error" role="alert">
                {error}
              </p>
            )}
            <button type="submit" disabled={state === 'sending' || !message.trim()}>
              {state === 'sending' ? 'Sending…' : 'Send'}
            </button>
          </form>
        )}
      </div>
    </main>
  )
}
