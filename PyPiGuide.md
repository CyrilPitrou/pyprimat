# PyPiGuide.md — Publishing `primat` to PyPI

Companion to `PRIMAT.md` §6 (the plan) — this is the step-by-step
checklist for actually doing it, with every irreversible action flagged
and a way to test up to (but not past) each one.

Current repo state: `.github/workflows/wheels.yml` exists (built per
`PRIMAT.md` §6.1) and builds cleanly locally (`make` in `primat-c/`,
`pip install -e .` for the Python extension). Nothing has been uploaded
anywhere yet — steps 1–6 below are all still ahead of you.

## Legend

- 🟢 **Reversible** — redo it, undo it, no lasting effect outside your
  own machine/repo.
- 🟡 **Hard to undo** — affects a shared system (GitHub) but can be
  cleaned up; mistakes are recoverable with effort.
- 🔴 **Irreversible** — affects PyPI's permanent, public, append-only
  index. Cannot be undone. Do these last, and only once you've verified
  everything beforehand.

---

## Step 1 — 🟢 Build and check locally, no network upload

Before touching any external service, verify the artifacts you'd
eventually publish actually work:

```bash
# sdist + wheel for your current platform only
python -m build

# Validate package metadata (long_description renders, no malformed
# classifiers, etc.) without uploading anywhere
pip install twine
twine check dist/*

# Sanity-install into a throwaway venv from the built wheel, not -e .
python -m venv /tmp/primat-check
source /tmp/primat-check/bin/activate
pip install dist/primat-*.whl
python -c "from primat import PRIMAT; print(PRIMAT({}).solve()['DoH'])"
deactivate
```

You can repeat this as many times as you want. Nothing here touches
PyPI, TestPyPI, or GitHub.

### Optional: exercise the multi-platform wheel matrix locally

`cibuildwheel` can run on your own machine before any of it goes near
GitHub Actions:

```bash
pip install cibuildwheel
cibuildwheel --platform macos   # builds the matrix for the OS you're on
```

This is the closest you can get to rehearsing `wheels.yml`'s
`build_wheels` job with zero network publish step and zero GitHub
involvement. It won't catch the MSVC-specific risk (you're not on
Windows) or the `aarch64` QEMU cross-build, but it does validate the
`setup.py`/`pyproject.toml` packaging metadata and the `primat-c`
extension build flags.

---

## Step 2 — 🟢 Trigger the GitHub Actions workflow manually, targeting **nothing real**

Push `.github/workflows/wheels.yml` to GitHub (a normal commit/push —
reversible, this repo already has commits going to `origin`). Once it's
on GitHub, `workflow_dispatch` lets you run the `build_wheels` +
`build_sdist` jobs on every OS in the matrix (including real Windows
and real aarch64 emulation) **without** running the `publish` job at
all, by triggering it from the Actions tab and just inspecting the
uploaded build artifacts (the `actions/upload-artifact` step) rather
than wiring up trusted publishing yet.

This is how you get the empirical Windows/MSVC check that `PRIMAT.md`
§6.1 flags as a real risk: trigger the workflow, look at whether the
`windows-latest` job's `primat._primat_c` extension actually compiles,
or silently fails over to the pure-Python fallback (check the build log
for the `optional_build_ext` warning). The complex-arithmetic rewrite
already done in `primat-c/src/weak_rates.c` (replacing `<complex.h>`
with a hand-rolled `(re, im)` struct) was specifically to remove one
known MSVC incompatibility ahead of this check; this step is where you
confirm there's nothing else lurking.

Nothing here uploads anything anywhere. You can re-run it indefinitely.

---

## Step 3 — 🟡 Push the workflow + register on TestPyPI

Register a separate account at **test.pypi.org** (distinct from
pypi.org — different login, different API). TestPyPI is an explicit
sandbox: project names there are not reserved on real PyPI, uploads can
be (and routinely are) overwritten/deleted, and nobody treats it as a
real index. Treat everything in this step as repeatable.

1. Create the TestPyPI account.
2. On TestPyPI, under your future `primat` project (it doesn't need to
   exist yet — first upload creates it) → set up **Trusted Publishing**
   pointing at this GitHub repo + `wheels.yml` + a `testpypi` GitHub
   environment (separate from the `pypi` environment used for the real
   index, so a misconfiguration can't accidentally hit the real index).
3. Temporarily point `wheels.yml`'s `publish` job at TestPyPI for this
   dry run — `gh-action-pypi-publish` takes a `repository-url` input:
   ```yaml
   - uses: pypa/gh-action-pypi-publish@release/v1
     with:
       repository-url: https://test.pypi.org/legacy/
   ```
   Keep this as a second job (or a workflow-dispatch input toggle)
   rather than overwriting the real-PyPI `publish` job, so you don't
   have to remember to revert it later.
4. Trigger via `workflow_dispatch` → wheels + sdist get built and
   uploaded to **test**.pypi.org.
5. Verify end-to-end: `pip install -i https://test.pypi.org/simple/ primat`
   into a clean venv, run the validation script (`PyPRIMAT_run.py`
   equivalent), and confirm the result matches the documented tolerances
   in `CLAUDE.md`.

Why 🟡 and not 🟢: pushing the workflow file change is a normal commit
(undo with another commit), but a TestPyPI upload, while low-stakes, is
itself a public action on a shared service — re-uploading the exact
same version+filename is still rejected by TestPyPI's index just like
real PyPI (each `(name, version)` is append-only there too). You just
don't care, because nobody depends on it and you can bump to `0.3.0rc1`,
`rc2`, etc. for as many dry runs as you need.

---

## Step 4 — 🔴 Claim the name on real PyPI

`PRIMAT.md` §6.2 step 2 calls this out explicitly: a manual
`twine upload` of an sdist-only `0.3.0rc0` build, run from your laptop,
claims the `primat` name on PyPI. **This is the first genuinely
irreversible action in the whole process.**

Why it's irreversible:
- PyPI project names are first-come-first-served, global, and permanent
  in practice. You can delete a project, but you cannot guarantee you
  (or anyone) can re-register the exact same name afterward — and an
  abandoned/deleted name sits in a "do not re-register easily" limbo
  PyPI maintains specifically to prevent supply-chain hijacking of
  formerly-used names.
- Once any `(name, version)` pair is uploaded, that exact version's
  files can never be re-uploaded — even after deletion. So once
  `primat-0.3.0rc0.tar.gz` exists on PyPI, that filename is burned
  forever; a botched upload means bumping to `0.3.0rc1` rather than
  trying again with `rc0`.

What to verify *before* this step (everything above, plus):
- The package name `primat` is not already taken by checking
  `https://pypi.org/project/primat/` yourself in a browser first — if
  it's taken, this whole plan needs a rename, which you want to know
  *before* burning a name on a placeholder.
- `twine check dist/*` (Step 1) passes clean.
- The TestPyPI dry run (Step 3) reproduced a working `pip install` end
  to end.
- You're certain `0.3.0rc0` is the version string you want to spend on
  this claim (small naming/version mistakes here are permanent).

This step needs your real PyPI account credentials — I can't perform it
for you, and you should run the `twine upload` command yourself rather
than have an agent run it from this machine, since it's a one-shot,
unrecoverable action tied to your personal PyPI identity.

---

## Step 5 — 🟡 Wire up Trusted Publishing on real PyPI

On pypi.org, under the now-claimed `primat` project → "Publishing" →
"Add a new publisher" → register this GitHub repo, the `wheels.yml`
workflow filename, and the `pypi` GitHub environment name. This grants
the `id-token: write` OIDC permission in the workflow the ability to
publish — no long-lived API token to manage or leak.

This is 🟡, not 🔴: misconfiguring or revoking a trusted publisher is
fully reversible from the PyPI UI (remove it, add a corrected one) as
long as you haven't yet triggered a publish through it. It only becomes
irreversible the moment a `release: published` event actually fires the
`publish` job successfully (→ Step 6).

---

## Step 6 — 🔴 Tag `v0.3.0`, publish the GitHub release, let `wheels.yml` upload to real PyPI

This is `PRIMAT.md` §9 Phase I. The GitHub release's `published` event
triggers `wheels.yml`'s full pipeline against the real `pypi` index.

Irreversible because:
- Once `primat-0.3.0` (wheels + sdist) lands on real PyPI, that exact
  version's files can never be replaced — only deleted (hiding it from
  new installs, but not erasing it from anyone who already resolved it,
  and not freeing the version number for reuse with different content).
- Anyone in the world can `pip install primat==0.3.0` from the moment
  the first wheel finishes uploading. There is no "private" undo.

Pre-flight checklist (everything above must already be true):
- [ ] Step 1: local build + `twine check` clean.
- [ ] Step 2: `workflow_dispatch` build-only run green on every OS in
      the matrix, including a real look at the Windows job's build log.
- [ ] Step 3: full TestPyPI install-and-run dry run matches `CLAUDE.md`
      tolerances.
- [ ] Step 4: name claimed, `0.3.0rc0` placeholder visible on
      pypi.org/project/primat/.
- [ ] Step 5: Trusted Publisher registered and pointing at the `pypi`
      environment + `wheels.yml`.
- [ ] The version string in `pyproject.toml` (`version = "0.3.0"`) is
      exactly what you intend to ship — this is your last chance to
      change it before it's permanent.
- [ ] You (not an agent) create the `v0.3.0` git tag and the GitHub
      release, since this is the action that actually fires the
      irreversible publish — treat the "Publish release" button on
      GitHub as the point of no return.

After this, the only thing left to verify is `pip install primat` from
a clean machine/venv with no `-i test.pypi.org` flag, confirming the
real index serves what you expect.
