# Source-only publication policy

## Allowed in the current public tree

- Research implementation and configuration templates.
- Tests using explicitly synthetic inputs.
- Concise usage, provenance and methodological documentation.

The exact permitted file set is recorded in `PUBLIC_FILES.txt`. New files require deliberate review and an explicit allowlist update. Run `python scripts/check_public_release.py` before committing or pushing.

## Keep private

Never commit patient records or derivatives, even if direct identifiers have been removed. This includes cohort memberships, split assignments, timestamps and clinical narratives; row-level labels and predictions; extracted sequences, arrays and embeddings; private fitted vocabularies and identifier mappings; trained weights, checkpoints and serialized estimators; and individual-case explanations or plots.

Also keep database files and exports, local logs and notebooks with outputs, credentials, institution-specific configuration, internal development notes and submission correspondence outside the public release. Archives are not a workaround: their contents can disclose the same information.

Training and analysis code may create these files locally. Their exclusion from Git is intentional. Model architecture code is not a trained checkpoint, and no trained model is distributed here.

## Limits of the checks

The allowlist and `.gitignore` reduce accidental inclusion; they are not a formal privacy guarantee. Text files can contain sensitive values and require content review. Aggregate statistics can also require disclosure review, especially for small groups. Do not bypass the checks with force-adds.

The core-release cleanup changes the current tree only, not Git history. If sensitive content is discovered in any historical commit or external artifact, stop publication and follow a coordinated remediation process. A normal deletion commit does not erase prior copies.
