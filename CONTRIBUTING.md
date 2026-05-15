# Contributing

Thanks for your interest in contributing to amla-sandbox.

## This is a release mirror

This repository is the public release source for the
[amla-sandbox](https://pypi.org/project/amla-sandbox/) Python package.
Development happens upstream in
[amlalabs/monorepo](https://github.com/amlalabs/monorepo). On each release the
package source is re-extracted from the monorepo and force-pushed here,
together with a pin (`.mirror-deps.json`) to the matching tag of
[amla-sandbox-core](https://github.com/amlalabs/amla-sandbox-core) for the
bundled WASM artifact.

That means:

- Pull requests opened against this repository will be clobbered on the next
  release. We will not silently lose your work; if a PR has merit, a
  maintainer will copy it into the monorepo and credit you in the resulting
  commit message. But please assume the surface area here is read-only.
- Issues are welcome here. Bug reports, feature requests, framework
  integration questions, and questions about the published wheel all belong
  here.
- Code changes are welcome as PRs against the monorepo. Link to the relevant
  monorepo path (e.g. `src/python/packages/amla-sandbox/...`) in your
  description.

## Reporting issues

Please include the `amla-sandbox` version, your Python version, your
operating system and architecture, and a minimal reproducer if possible.

## Development checks

If you are opening a PR against the upstream monorepo (the normal route for
code changes, as described above), the monorepo uses a `pre-commit`
configuration that runs a set of hooks on every commit, including
`actionlint` for GitHub Actions workflow files. Please ensure all hooks
pass before opening the PR.

From the monorepo root:

```bash
pre-commit install            # one-time, wires the git hook
pre-commit run --all-files    # run every hook across the whole tree
```

CI runs the same set on every PR. Anything that fails locally will fail
there too. External contributors who only see this mirror repository do
not need to install pre-commit; the hooks live with the source.

## License

This repository ships two components under different licenses; please read
both before contributing so you know what you are agreeing to.

- The Python package source in this repository is licensed under the MIT
  License (see `LICENSE`). By contributing code to the Python package, you
  agree that your contribution will be licensed under MIT.
- The bundled WebAssembly artifact (`src/amla_sandbox/_wasm/amla_sandbox.wasm`)
  is built from the Rust runtime in
  [amla-sandbox-core](https://github.com/amlalabs/amla-sandbox-core) and is
  licensed under `AGPL-3.0-or-later OR BUSL-1.1`. It is not part of the MIT
  source in this repository; its source and license terms live in the
  amla-sandbox-core mirror.

If your change touches the Rust runtime, please open the PR against the
monorepo and reference the relevant `src/rust/crates/amla-sandbox/...` path;
those contributions are governed by the runtime's license, not MIT.
