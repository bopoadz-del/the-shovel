# Security Policy

## Supported versions

The Fork is continuously deployed. Security fixes ship on the `main` branch
and on the live service at [theshovel.ai](https://theshovel.ai).

| Track | Supported |
| ----- | --------- |
| `main` / live theshovel.ai | Yes |
| Other branches, historical tags, and unofficial forks | No |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for a vulnerability that could
expose client documents, credentials, authentication bypass, or remote code
execution.

Report privately using one of:

1. A [GitHub private security advisory](https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories/creating-a-repository-security-advisory) on this repository.
2. The contact listed on [theshovel.ai](https://theshovel.ai).

Include the impact, a reproduction, and (if you have one) a suggested fix.
You should receive an acknowledgement within 7 days. We will not discuss
undisclosed reports in public issues.

## Secrets and credentials

Never commit API keys, `.env` files, or dashboard secrets. Production
secrets belong in the host environment (Render dashboard `sync: false`
blueprint values, or an equivalent secret store). Rotate any credential
that has appeared in git history before treating it as live.

## Dependency alerts

Open Dependabot alerts that cannot be cleared by a semver bump are
documented in [`deploy/DEPENDENCY_RISKS.md`](deploy/DEPENDENCY_RISKS.md)
with acceptance rationale.
