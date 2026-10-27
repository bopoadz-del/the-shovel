# CI secrets & variables

GitHub **Actions** secrets and variables the-shovel's workflows require.
This file documents **names and formats only** — never a value. Values live
in the repository's Settings → Secrets and variables → Actions, and in the
owner's local backup. A committed value would defeat the secret.

Set a secret:  `gh secret set <NAME> --repo bopoadz-del/the-shovel`
Set a variable: `gh variable set <NAME> --repo bopoadz-del/the-shovel --body <value>`
List (names only): `gh secret list` / `gh variable list`

## Secrets

| Name | Format | Used by | Notes |
|------|--------|---------|-------|
| `SECRET_SCAN_PATTERNS` | newline-separated regexes, one per line; `#` comments and blank lines ignored | `lint.yml` (`integrity`, `client-pattern scan`) via `scripts/scan_secrets.py` | **Fail-closed**: empty/unset ⇒ exit 2, never "clean". Covers confidential project/client identifiers + owner-identity name forms and personal emails. The list is itself sensitive — never commit it. |
| `UI_PHYS_FORBIDDEN` | newline-separated literal strings | live UI / acceptance jobs | Figures and identifiers the UI must never surface. |
| `FORK_API_KEY` | opaque API key | `smoke-live.yml`, `eval-battery.yml`, `browser-nightly.yml` | App API key for the live/acceptance jobs. |
| `FORK_BASE_URL` | URL (`https://theshovel.ai`) | live/acceptance jobs | Base URL the acceptance harness targets. |
| `FORK_EMAIL` | email | live/acceptance jobs | Test-account login. |
| `FORK_PASSWORD` | string | live/acceptance jobs | Test-account password. |
| `FORK_TOKEN` | bearer token | live/acceptance jobs | Test-account session/API token. |
| `AWS_ACCESS_KEY_ID` | AWS key id | `deploy-aws.yml` | ECS/ECR deploy. |
| `AWS_SECRET_ACCESS_KEY` | AWS secret | `deploy-aws.yml` | ECS/ECR deploy. |
| `AWS_DEFAULT_REGION` | region | `deploy-aws.yml` | Deploy region. |

`GITHUB_TOKEN` is provided automatically by Actions; no setup needed.

## Variables

| Name | Format | Used by | Notes |
|------|--------|---------|-------|
| `PUBLIC_BASE_URL` | URL (`https://theshovel.ai`) | `deploy-aws.yml` | Live base URL for the post-deploy `/health` + `build_sha` confirm step. A variable, not a secret (value is public). |

## Release assets (not a secret, but required for `build`)

`data/models/manifest.json` pins model weights to a GitHub **release** of this
repo (`release_tag`, with sha256 + size per file). `scripts/fetch_model.py`
downloads them with `gh release download` during `docker build`. The release
and its assets must exist on this repo:

- Release `safety-world-v2` → `safety_world_v2.onnx`, `safety_world_v2.prompts.json`
