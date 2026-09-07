# DO NOT USE THIS PIPELINE IN PRODUCTION

**NOSHIP_AGPL__yoloe__moge2__sonnet5**

This pipeline depends on `yoloe`, which **cannot be shipped**:

- **yoloe** — AGPL-3.0 or paid Ultralytics Enterprise

## What that means in practice

AGPL-3.0 requires publishing the source of the ENTIRE work that uses it, and
it reaches network deployment, so a hosted API does not avoid it. Ultralytics
sells an Enterprise Licence as the alternative.

## What this folder IS for

Measuring what staying licence-clean costs us. If `NOSHIP_AGPL__yoloe__moge2__sonnet5` scores materially better
than a shippable pipeline, that number is an argument for buying a licence or for
finding another approach — it is **not** permission to ship this.

Every run requires `--allow-noncommercial`, and the results are stamped
`shippable: false`. That flag is a deliberate speed bump, not a formality.

## Before running this at all

Ultralytics' own published position is that **any** use of their models — including
internal research, commercial or not — requires either releasing your entire project
under AGPL-3.0 or buying an Enterprise Licence. AGPL also reaches SaaS deployment, so
wrapping it in an API is not an escape.

So this is a legal question, not a technical one. **Get BJIT/NX sign-off first.**

See `MODELS.md` for the licence reasoning and `GAPS.md` gap B9.
