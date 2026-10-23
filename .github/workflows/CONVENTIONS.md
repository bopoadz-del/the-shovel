# GitHub Actions — Conventions for this repo

Read this before adding or modifying any file in `.github/workflows/`. These rules came out of retrospectives on PRs #10-#14 and #23 and exist to avoid re-discovering the same gotchas.

## 1. Minimal `permissions` at the top of every workflow

Default to read-only and add granular write scopes only where actually needed:

```yaml
permissions:
  contents: read
```

Add the additional scope at the job level when needed, NOT at the top level:

```yaml
jobs:
  publish:
    permissions:
      contents: read
      packages: write    # only for GHCR push
    steps:
      - ...
```

Why: PR #11 fixed exactly the "everything has write by default" CodeQL alert. The minimal-privilege workflow file is the durable form of that fix. Reviewers will flag any workflow that omits the `permissions:` block at the top.

Reference: existing `.github/workflows/docker-publish.yml` and `test.yml` both use this pattern.

## 2. No QEMU-emulated multi-arch builds

Do NOT use this pattern:

```yaml
- uses: docker/build-push-action@v5
  with:
    platforms: linux/amd64,linux/arm64    # ← arm64 here means QEMU
```

QEMU emulation runs **~5-10× slower than native** for the kinds of workloads in this repo (ifcopenshell + numpy/scipy + OCR stack). On hosted runners with default cache settings, the emulated arm64 build routinely overruns:

- The 10-minute step budget on the `test.yml` job.
- The Azure SAS-token validity window for `cache-to: type=gha` (cache writes silently fail after the token expires partway through a slow build).

PR #10 introduced `linux/amd64,linux/arm64` as an aspiration; PR #23 had to revert the arm64 half for exactly this reason. Don't re-introduce it.

**Correct pattern for multi-arch when it's needed**:

- Run amd64 on the default `ubuntu-latest` runner.
- Run arm64 on a **separate job** targeting a native arm64 runner (GitHub's `ubuntu-22.04-arm` large runner, or a self-hosted arm64 runner — including the future Orin once it has GH runner installed).
- Each job pushes its own platform-tagged manifest entry. Combine via a `docker buildx imagetools create` step at the end if you want a single multi-arch tag.

## 3. Cache strategy

- `cache-from`/`cache-to: type=gha` is fine for amd64 native builds — the SAS token comfortably outlives a typical build.
- For arm64 native runners, **prefer `type=registry,ref=ghcr.io/<owner>/<repo>:buildcache`** — no time-bomb, slightly more setup.
- Avoid `cache-to: type=gha,mode=max,ignore-error=true` for new workflows. The `ignore-error=true` silently masks cache failures and leads to progressively slower builds with no alert. PR #23 used it as defense-in-depth against a known Azure SAS issue; that's a fix-the-incident scope, not a default for new workflows.

## 4. Tag scoping

For images pushed to GHCR via `docker/metadata-action`:

```yaml
tags: |
  type=raw,value=latest,enable={{is_default_branch}}
  type=semver,pattern={{version}}
  type=sha,prefix=,format=short
```

This produces `latest` only on `main` (or whatever the repo's default branch is), and a separate `vX.Y.Z` tag for every release tag. PR #10 got this right; future workflows should copy the pattern.

## 5. Trigger discipline

- `on: [push, pull_request]` is overcautious for most workflows — it runs everything twice on PR commits. Default to `on: pull_request` for CI; `on: push` to `main` only for jobs that publish artifacts (docker images, package releases).
- `on: schedule` for nightly things (CodeQL re-scan, dependency audit). Always include `workflow_dispatch:` alongside so the workflow can be re-run manually without waiting for the cron.

## 6. Secrets

- Never reference `secrets.GITHUB_TOKEN` unless you actually need write to the repo from the workflow. The token has wider permissions than most workflows need; the `permissions:` block in section 1 is what scopes it down.
- For external API keys (DeepSeek, Anthropic, etc.), use **repository secrets**, not environment secrets, unless you actually need per-environment values. PR #14 records the policy on `.env` handling; the workflow side is the same — never echo a key, never pass it to a step that logs its env.

---

If a workflow change needs to break any of these, document the exception in the PR body. The reviewer's job is to challenge the exception; the workflow author's job is to justify it.
