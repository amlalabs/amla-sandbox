# Secrets: amla-sandbox mirror release workflow

This document covers credentials consumed by
[`.github/workflows/release.yml`](./.github/workflows/release.yml) in
`amlalabs/amla-sandbox`.

The workflow downloads `amla_sandbox.wasm` from the matching
`amla-sandbox-core` release, verifies its SHA256, bundles it into a Python
wheel, publishes the wheel + sdist to PyPI via OIDC trusted publishing, and
attaches both artifacts to a GitHub Release on this mirror.

**There are no traditional secret values to configure.** PyPI publishing is
token-less (OIDC), and the GitHub Release uses the auto-provided
`GITHUB_TOKEN`. What you do need is a **one-time PyPI Trusted Publisher
configuration**, documented below.

Upstream of this workflow: the monorepo orchestrator pushes the `vX.Y.Z` tag
that triggers it. See [`../../../.github/SECRETS.md`](../../../.github/SECRETS.md)
for the PAT (`MIRROR_PUSH_TOKEN_AMLA_SANDBOX`) used for that push.

## Overview

| Name                       | Type                          | Purpose                                                              |
| -------------------------- | ----------------------------- | -------------------------------------------------------------------- |
| (no secret)                | PyPI trusted publisher        | Publish wheel + sdist to PyPI without a token, via OIDC.             |
| `GITHUB_TOKEN`             | Auto-provided by Actions      | Create the GitHub Release; upload wheel + sdist as release assets. Also used by `gh attestation verify` to fetch the SLSA build-provenance attestation for `amla_sandbox.wasm` from `amla-sandbox-core`. |
| (no secret)                | OIDC `id-token: write`        | The OIDC token PyPI exchanges for a short-lived upload credential.   |

## PyPI Trusted Publishing (one-time setup)

This is the actual gate on publishing. There is no API token stored anywhere.
Instead, the PyPI project is configured to trust an OIDC identity coming from
this exact repo + workflow, and only that identity can publish.

### Configure

1. Sign in to PyPI as an owner of the `amla-sandbox` project.
2. Go to <https://pypi.org/manage/project/amla-sandbox/settings/publishing/>.
3. Under **"Add a new pending publisher"** (or "Add a new publisher" if the
   project already exists), choose **GitHub** and fill in:

   | Field             | Value                                       |
   | ----------------- | ------------------------------------------- |
   | Owner             | `amlalabs`                                  |
   | Repository name   | `amla-sandbox`                              |
   | Workflow name     | `release.yml`                               |
   | Environment name  | *leave blank*                               |

   Note: leave **Environment name** blank unless you also gate the publish
   job behind a GitHub Environment (e.g. `release`). The current workflow
   does not use an environment, so the field must be empty for the claim to
   match. If you later add `environment: release` to the `publish-wheel`
   job, you must come back here and set the same name; OIDC matches all
   four fields exactly.

4. Save. Subsequent pushes of `v*` tags to `amlalabs/amla-sandbox` will
   trigger the workflow and be allowed to publish.

For a brand-new PyPI project (first release), register a **pending
publisher** instead via the same UI before the first publish; PyPI will
create the project on first successful publish.

Reference docs:
<https://docs.pypi.org/trusted-publishers/>
<https://docs.pypi.org/trusted-publishers/adding-a-publisher/>

### What the workflow uses

The publish step is:

```yaml
- name: Publish to PyPI (OIDC trusted publishing)
  uses: pypa/gh-action-pypi-publish@cef221092ed1bacb1cc03d23a2d87d1d172e277b  # v1.14.0
  with:
    packages-dir: dist
```

No `password:` / `api-token:` is set. The action requests an OIDC token (the
job declares `permissions: id-token: write`), POSTs it to PyPI's
`/_/oidc/mint-token` endpoint, gets back a short-lived (15-minute) upload
token tied to this exact run, and uses it to `twine upload`.

### Scope of access

A successful OIDC exchange yields a token that:

- **Can**: upload wheels and sdists to the `amla-sandbox` PyPI project only.
- **Cannot**: yank releases, modify metadata, manage project collaborators,
  publish to any other PyPI project, or be reused after the run ends.

### Failure modes

| Symptom in `publish-wheel` job                                                       | Cause                                                                       |
| ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `Trusted publishing exchange failure: invalid-publisher`                             | PyPI publisher entry doesn't match repo/workflow/environment exactly.       |
| `Trusted publishing exchange failure: not-pending`                                   | First publish but no pending publisher was registered. Add one then retry.  |
| `HTTPError: 403 Forbidden ... user does not have permission to upload`               | Project name in `pyproject.toml` differs from the PyPI project (`amla-sandbox`). Verify `[project] name = "amla-sandbox"` in `pyproject.toml`. |
| `Error: OIDC token retrieval failed`                                                 | `id-token: write` missing from the job, or OIDC disabled at org level.      |
| `400 Bad Request: File already exists`                                               | Re-running the same tag after a successful publish. PyPI immutability. Bump version, do not retry. |

### Rotation

Trusted publishers do not rotate. There is no token to expire. To revoke
publish rights:

1. PyPI UI -> publishing settings -> delete the trusted publisher entry.
   Publishes from this workflow stop immediately.

To migrate to a different repo or workflow file (rare): delete the old
publisher entry, add a new one with the new claims. Plan a brief window
where neither old nor new can publish during the swap.

### Compromise detection

Compromise of a trusted publisher requires compromise of the source repo
itself (push access to `amla-sandbox` or write access to the workflow file).
There is no token to steal.

- Monitor unauthorized pushes to `amlalabs/amla-sandbox` and unauthorized
  edits of `.github/workflows/release.yml` via GitHub's audit log.
- PyPI emails the project owners after every successful publish; an
  unexpected email is the first signal.
- `pip download amla-sandbox==<version>` and verify the wheel matches the
  artifact that was attached to the GitHub Release for the same tag.

## Attestation verification

Before the wheel is built, the workflow runs `gh attestation verify` (step
"Verify amla_sandbox.wasm provenance attestation") against the
`amla_sandbox.wasm` it just downloaded from the
[`amla-sandbox-core`](https://github.com/amlalabs/amla-sandbox-core) release.
This is a supply-chain integrity check that complements the SHA256 check that
runs immediately before it:

- The SHA256 check proves the bytes match what the upstream release page
  publishes.
- The attestation check proves *who built those bytes* — namely, that the
  binary came out of a signed GitHub Actions run in `amlalabs/amla-sandbox-core`
  (via `actions/attest-build-provenance`, signed through Sigstore).

The step uses the auto-provided `GITHUB_TOKEN`. Its default `contents:read`
permission is enough to fetch the public attestation manifest from
`amla-sandbox-core`; no secret to configure.

### Failure modes

| Symptom                                                                       | Cause                                                                                                  |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `gh attestation verify: no matching attestations found`                       | `amla-sandbox-core` CI did not run `attest-build-provenance` for this release (or the run failed). The upstream release is incomplete; do not publish from a release with no provenance. |
| `gh attestation verify: failed to verify` / signature mismatch                | Possible tampering: the bytes match SHA256 but the attestation does not validate. Stop. Inspect the upstream release and the attestation manifest before any further action. |
| `gh: HTTP 401/403` on attestation API                                         | `GITHUB_TOKEN` lost `contents:read` somehow (would only happen if the job's `permissions:` block was edited incorrectly). Restore `contents: read` (or `write`) on the publish job. |

If `amla-sandbox-core`'s release workflow is changed in a way that drops the
`actions/attest-build-provenance` step, this gate will start failing
immediately. That is the desired behavior: it forces the upstream regression
to be fixed before any `amla-sandbox` wheel goes to PyPI.

## `GITHUB_TOKEN`

Auto-provided. The `publish-wheel` job declares:

```yaml
permissions:
  id-token: write   # for PyPI OIDC
  contents: write   # for the GitHub Release at the end
```

The `contents: write` portion is what lets `gh release create` /
`gh release upload` work on this mirror.

### Repo prerequisites

Same as for the core mirror: prefer per-job `permissions:` blocks (already in
the workflow) over flipping the repo-wide setting to "Read and write
permissions". See
[`mirrors/amla-sandbox-core/overlay/SECRETS.md`](../../amla-sandbox-core/overlay/SECRETS.md#repo-prerequisites)
for the same discussion in more depth; the constraints are identical.

### Failure modes

| Symptom                                                                  | Cause                                                       |
| ------------------------------------------------------------------------ | ----------------------------------------------------------- |
| `HTTP 403: Resource not accessible by integration` on `gh release create` | `contents: write` missing from job permissions.             |
| `gh release view ... not found` then `release create` fails              | A previous partial run left orphan state. Delete in UI, re-run. |

## Branch protection setup

The monorepo orchestrator pushes new `main` commits and `vX.Y.Z` tags into
`amlalabs/amla-sandbox` on every release using the
`MIRROR_PUSH_TOKEN_AMLA_SANDBOX` PAT documented in the parent
[`.github/SECRETS.md`](../../../.github/SECRETS.md). The push bypasses code
review by design: this repo is a read-only release artifact and the
authoritative review happens upstream in the monorepo PR.

If branch protection is enabled on `main` (or tag protection on `v*`), the
PAT identity must be allowed to bypass the relevant checks or the
orchestrator push will fail.

Two equivalent ways to make this work:

1. **Recommended for a true release mirror**: leave `main` and `v*` with
   no branch protection. No human ever pushes to this repo directly, so
   protection adds no value, only operational friction.

2. **If branch protection is required by org policy**: in
   **Settings -> Branches -> Branch protection rules -> `main`**, add the
   bot user that owns `MIRROR_PUSH_TOKEN_AMLA_SANDBOX` to
   **"Allow specified actors to bypass required pull requests"**. If you
   also use **Settings -> Tags -> Tag protection rules** with a `v*`
   pattern, add the same bot user there too.

Symptom of misconfiguration: orchestrator logs show

```
remote: error: GH006: Protected branch update failed for refs/heads/main.
```

or, for tag-protection,

```
remote: error: GH013: Tag protection rule violated for v1.2.3.
```

Fix by adding the PAT identity to the bypass list, or by removing the
protection rule entirely.
