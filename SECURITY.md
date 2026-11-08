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

## Rate limits behind a proxy

Anonymous buckets use the socket peer unless both `TRUSTED_PROXY_COUNT` and
`TRUSTED_PROXY_CIDRS` are set. The count is the number of `X-Forwarded-For`
hops the proxy tier appends. The CIDRs are the networks those proxies belong
to; the immediate peer must fall in one of them. Unset, zero, or invalid
configuration ignores forwarded headers (fail closed). `0.0.0.0/0` and `::/0`
are rejected. `X-Real-IP` and `Forwarded` are not consulted.

Credential routes (login, registration, password reset, email verification,
and `/v1/auth`) are denied when the shared Redis limiter cannot answer.
Other routes stay available during a Redis error. Set the two proxy variables
in the service environment to the real proxy tier; they are not secrets, and
this repository does not ship a production value for them.

## Dependency alerts

Open Dependabot alerts that cannot be cleared by a semver bump are
documented in [`deploy/DEPENDENCY_RISKS.md`](deploy/DEPENDENCY_RISKS.md)
with acceptance rationale.
