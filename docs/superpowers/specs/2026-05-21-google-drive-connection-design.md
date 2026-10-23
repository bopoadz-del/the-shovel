# Google Drive Connection — Design Spec

**Date:** 2026-05-21
**Goal:** Let the user connect a personal Google Drive and import Drive files
into a project as documents — "Connect Drive" → Google login → browse Drive →
pick files → they appear as project documents.

## Decisions (from brainstorming)

- **OAuth client:** the user supplies `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`
  (already in `.env`). Built and tested end-to-end against a real Drive.
- **Scope of connection:** ONE app-wide Drive connection (not per-project). The
  user connects once; any project can then import from it.
- **File handling:** import a COPY — the file is downloaded once and stored as a
  normal project document; a later Drive edit does not auto-sync.
- **OAuth scope:** `drive.readonly` — read/import only, never write to the Drive.
- **Token storage:** an encrypted file, not the DB.

## Architecture

### New files
- `app/routers/drive.py` — the `/v1/drive/*` HTTP surface.
- `app/core/drive_auth.py` — token persistence + access-token refresh.

### Modified files
- `app/blocks/google_drive.py` — keep `list`/`download`; the broken OOB
  `_oauth_url` and "paste GOOGLE_ACCESS_TOKEN" path are superseded by the router.
- `app/main.py` — mount the drive router.
- `app/routers/projects.py` (or `drive.py`) — the per-project import endpoint.
- `app/static/index.html` — wire the "Connect Drive" modal + a Drive file browser.

## OAuth flow — `app/routers/drive.py`

All routes require `Authorization: Bearer $CEREBRUM_DEV_KEY` like other `/v1/*`,
**except `/v1/drive/callback`** — Google calls that directly and cannot send our
header, so it is protected by the OAuth `state` value instead.

| Route | Behaviour |
|-------|-----------|
| `GET /v1/drive/connect` | Build the Google consent URL — `scope=drive.readonly`, `access_type=offline`, `prompt=consent`, `redirect_uri` = `GOOGLE_REDIRECT_URI`, a random `state` (stored server-side, single-use). 302-redirect the browser to Google. |
| `GET /v1/drive/callback?code=&state=` | Verify `state`. Exchange `code` at `https://oauth2.googleapis.com/token` for `access_token` + `refresh_token` + `expiry`. Persist via `drive_auth`. Fetch the account email (`/oauth2/v2/userinfo` or Drive `about`). 302-redirect back to `/`. |
| `GET /v1/drive/status` | `{connected: bool, email: str|null, configured: bool}` — `configured` is false when `GOOGLE_CLIENT_ID` is unset. |
| `POST /v1/drive/disconnect` | Delete the stored token. |
| `GET /v1/drive/files?q=<search>` | Live (auto-refreshed) token → `GoogleDriveBlock` `list`. Returns the file list. |
| `POST /v1/projects/{project_id}/drive/import` | Body `{file_id}`. `GoogleDriveBlock` `download` → save bytes via `file_crypto.write_document` + `projects.add_document` — i.e. the exact path an upload takes. No analysis runs (Epic 0.3). Returns the created document record. |

## Token storage & refresh — `app/core/drive_auth.py`

- One app-wide token, persisted at `DATA_DIR/google_drive_token.json`, written
  through `file_crypto` so the **refresh token is encrypted at rest**.
- Stored fields: `access_token`, `refresh_token`, `expiry` (epoch), `email`.
- `load_token()` / `save_token()` / `clear_token()`.
- `get_access_token()` — returns a valid access token, transparently refreshing
  against `https://oauth2.googleapis.com/token` (grant `refresh_token`) when
  `expiry` has passed; persists the refreshed token. Raises a typed error if no
  token is stored or the refresh fails (→ caller prompts reconnect).

## Frontend — `app/static/index.html`

- The existing "Connect Drive" modal's Google Drive option opens
  `/v1/drive/connect` (full-page or popup). On `/v1/drive/status.connected`,
  the modal shows "Connected as &lt;email&gt;" + a Disconnect action.
- A Drive browser: a search box → `GET /v1/drive/files?q=` → a file list; each
  row has an "Add to project" action → `POST /v1/projects/{id}/drive/import`
  for the active project. Imported files appear in the project's document list.
- If `status.configured` is false, the option shows "Google Drive not
  configured" instead of a dead button.

## Error handling

| Situation | Behaviour |
|-----------|-----------|
| `GOOGLE_CLIENT_ID` unset | `status.configured=false`; `/connect` returns a clear "not configured" error, no redirect. |
| Not connected, but `/files` or import called | HTTP 409 + a "connect Drive first" message. |
| `state` mismatch on callback | HTTP 400, no token stored. |
| Refresh-token exchange fails | Clear the stored token; tell the user to reconnect. |
| Google API error (list/download) | Surfaced verbatim — never a fabricated result. |

## Setup prerequisites (user side, one-time)

- In Google Cloud Console, the OAuth client must list
  `http://localhost:8000/v1/drive/callback` as an **authorized redirect URI**.
- The OAuth consent screen is in "testing" mode → the user's own Google account
  must be added as a **test user** (the `drive.readonly` sensitive scope is
  allowed for test users without app verification).

## Testing

- `app/core/drive_auth.py` — unit tests: token save/load round-trip is
  encrypted on disk; `get_access_token()` refreshes when expired; missing-token
  and failed-refresh raise the typed errors. Token endpoint mocked.
- `app/routers/drive.py` — router tests with the Google HTTP calls mocked:
  `/connect` redirects, `/callback` stores a token and rejects a bad `state`,
  `/status` reflects state, `/files` and import behave, auth is enforced on the
  gated routes.
- One `@pytest.mark.skipif`-on-`GOOGLE_CLIENT_ID` live test exercising the real
  Google round-trip.
- Full suite stays green.

## Out of scope

- Writing to / modifying the user's Drive (`drive.readonly` only).
- Per-project Drive connections; multiple Google accounts.
- Live re-sync of imported files when the Drive original changes.
- OneDrive / local-drive connectors (separate blocks already exist).
