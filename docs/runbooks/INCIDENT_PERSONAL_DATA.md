# Personal-data incident runbook

This is the one-page incident process for a personal-data breach. It implements
the obligation in `docs/legal/PRIVACY_POLICY.md` (section 6). Personal data here
means any information about an identifiable individual processed by the
Service.

## 1. Detect

- Signals that may indicate a breach: failed logins against a single account,
  unexpected exports, a leaked API key or login token, an unintended change to
  access controls, an upstream processor reporting an incident, or a customer
  report.
- Anyone who sees a signal records it immediately in the incident tracker with
  the time, the signal, and the data involved. Do not wait to confirm before
  recording.

## 2. Contain

- If a credential is compromised: rotate or revoke it (account password, API
  key, login token) and end the affected session.
- If access is exposed: restrict the affected scope or take the affected
  component out of the data path until the exposure is closed.
- Preserve logs and evidence; do not delete anything that may be needed for the
  assessment.

## 3. Assess

- Within 24 hours of detection, determine: what personal data was involved,
  how many people are affected, whether the data left the platform, and the
  likely harm (identity, financial, reputational).
- Record the assessment with the evidence and the conclusion.

## 4. Notify

- Notify affected customers without undue delay, in plain language: what
  happened, what data was involved, what we have done, and what they should do.
- Where GDPR (or an equivalent regime) applies, notify the competent
  supervisory authority within 72 hours of becoming aware of the breach.

## 5. Post-mortem

- After the breach is contained and notified, run a post-mortem: root cause,
  what failed, what is being changed so it does not recur, and who owns each
  follow-up action.
- Update this runbook if the incident exposed a gap in it.
