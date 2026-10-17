# Green SM Full Corpus Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this runbook task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialise, validate, preserve, and hand off the immutable full Green SM v1 paired corpus for full Foundation pretraining and the Food/Ads downstream experiments.

**Architecture:** This is an operational plan, not a change to the simulator. The existing deterministic simulator reads `configs/green_sm_sim_full.json`, creates two roots from one target world (`linked` and `paired_control`), and validates both roots before returning success. The resulting artifacts are immutable: a non-empty output directory is rejected rather than overwritten.

**Tech Stack:** Python 3, PyArrow/Parquet, existing `src.data.green_sm` CLI, PowerShell, Kaggle Dataset upload.

**Spec:** `docs/superpowers/specs/2026-10-15-green-sm-simulated-corpus-design.md`

## Global Constraints

- Run on the local machine's **CPU**; generation does not use a GPU. Reserve Kaggle GPU time for pretraining and downstream training only.
- Use exactly `configs/green_sm_sim_full.json`: `green_sm_sim_full_v1`, seed `20261015`, 12,000 users, 42 days, `max_tokens=256`, and full fraction `1.0`.
- Do not edit the JSON configuration while a run is in progress. Its digest is stored in the emitted reports and is part of the corpus identity.
- Do not delete, reuse, or overwrite `data/processed/green_sm_sim_full_v1`. If it is non-empty, stop and investigate rather than generating a mixed corpus.
- Keep `data/processed/**`, generated logs, and archives out of Git. The dataset is immutable data, not source code.
- Treat `linked` as the training corpus and retain `paired_control` unchanged for the matched-control experiment. Both must be uploaded together.
- Do not use downstream labels in Foundation pretraining. Food and Ads labels are for their own trainer/evaluator only.

## Expected Deliverable

```text
data/processed/green_sm_sim_full_v1/
  green_sm_sim_linked_v1/
    foundation/
    food_opportunities/
    ads_opportunities/
    reports/{data_quality_report,causal_audit_report}.json
  green_sm_sim_control_paired_v1/
    foundation/
    food_opportunities/
    ads_opportunities/
    reports/{data_quality_report,causal_audit_report}.json
```

The full configuration requires 144,000 Foundation snapshots, 24,000 Food requests x 20 candidates, and 32,000 Ads requests x 10 candidates **in each paired root**. From the actual 6.69 MiB pilot, a linear lower-bound estimate for the two full roots is about 335 MiB. Require at least **5 GiB free disk** before starting to allow for Parquet writes, reports, logs, and a compressed upload copy.

## Review Focus

- A stale or partially populated full root must be rejected; never mix output from two invocations.
- The configuration must still declare full mode, exactly 12,000 users, 144,000 snapshots, and the v1 target path before running.
- Both roots must report `readiness_passed: true`; a successful Python exit code alone is insufficient evidence.
- Linked and control logical Food/Ads requests must remain identical after removing only their Foundation-manifest linkage.
- Kaggle must receive the generated data roots and reports, not an accidental repository copy, pilot corpus, checkpoint archive, or run artifacts.

---

### Task 1: Freeze the input identity and verify local capacity

**Files:**
- Read: `configs/green_sm_sim_full.json`
- Read: `docs/superpowers/specs/2026-10-15-green-sm-simulated-corpus-design.md`
- Target (must be absent/empty): `data/processed/green_sm_sim_full_v1/`
- Create: `artifacts/full-corpus-logs/` (run logs only; ignored by Git)

**Consumes:** The audited simulator and the approved full configuration.

**Produces:** One recorded configuration digest and a go/no-go check for an immutable run.

- [ ] **Step 1: Inspect the Git working tree and full-output target without changing either.**

  Run from the repository root:

  ```powershell
  git status --short
  Test-Path data/processed/green_sm_sim_full_v1
  if (Test-Path data/processed/green_sm_sim_full_v1) {
      Get-ChildItem data/processed/green_sm_sim_full_v1 -Force | Select-Object -First 20
  }
  ```

  Expected: unrelated source changes may exist, but the full output root is absent or empty. A non-empty root is a stop condition: preserve it and decide whether it is a complete earlier corpus before any later action.

- [ ] **Step 2: Parse the configuration through the production config loader and record its identity.**

  ```powershell
  python -c "from src.data.green_sm.config import GreenSmConfig; c=GreenSmConfig.from_json('configs/green_sm_sim_full.json'); assert c.mode == 'full' and c.pilot_fraction == 1.0 and c.user_count == 12000 and c.max_tokens == 256; assert sum(c.snapshot_counts.values()) == 144000; print({'data_version': c.data_version, 'output_root': str(c.output_root), 'config_digest': c.config_digest, 'snapshots': dict(c.snapshot_counts), 'food_requests': dict(c.food_request_counts), 'ads_requests': dict(c.ads_request_counts)})"
  ```

  Expected: `data_version` is `green_sm_sim_full_v1`, target root is `data/processed/green_sm_sim_full_v1`, and no assertion fails.

- [ ] **Step 3: Verify enough free space using Windows Explorer or PowerShell.**

  ```powershell
  Get-PSDrive -Name C | Select-Object Name, Used, Free
  ```

  Expected: at least 5 GiB free on the drive containing the repository. If lower, free space or choose another local workspace before generating; do not start a run that may be interrupted by disk exhaustion.

- [ ] **Step 4: Create a dated log directory and capture the preflight evidence.**

  ```powershell
  New-Item -ItemType Directory -Force artifacts/full-corpus-logs | Out-Null
  Get-Date -Format o | Set-Content artifacts/full-corpus-logs/full-v1-preflight.txt
  ```

  Expected: only an ignored log directory is created; no simulator data has been written.

### Task 2: Generate the two immutable full corpus roots

**Files:**
- Read: `configs/green_sm_sim_full.json`
- Execute: `src/data/green_sm/__main__.py`
- Create: `data/processed/green_sm_sim_full_v1/**`
- Create: `artifacts/full-corpus-logs/full-v1-generation.log`

**Consumes:** The completed Task 1 preflight.

**Produces:** Linked and paired-control Parquet datasets with their manifests, preprocessing configuration, and reports.

- [ ] **Step 1: Start one foreground CPU generation process and retain its complete log.**

  ```powershell
  python -m src.data.green_sm --config configs/green_sm_sim_full.json *>&1 |
    Tee-Object -FilePath artifacts/full-corpus-logs/full-v1-generation.log
  if ($LASTEXITCODE -ne 0) { throw "Green SM full generation failed with exit code $LASTEXITCODE" }
  ```

  The command builds both roots and invokes `validate_outputs()` before it exits. It can take substantially longer than the 2% pilot because it writes 50 times as many users, snapshots, and requests. Leave this terminal and the laptop awake; no GPU is required.

- [ ] **Step 2: Confirm the CLI returned its final JSON result rather than only intermediate output.**

  ```powershell
  Get-Content artifacts/full-corpus-logs/full-v1-generation.log -Tail 5
  ```

  Expected: the final JSON includes `foundation_roots`, `paired_logical_rows_equal: true`, `target_world_fingerprint`, `config_digest`, and a `validation` object. Any exception, missing root, or `false` paired equality is a failure; keep the log and do not upload the corpus.

### Task 3: Verify full-scale counts, quality gates, and lineage independently

**Files:**
- Read: `data/processed/green_sm_sim_full_v1/green_sm_sim_linked_v1/**`
- Read: `data/processed/green_sm_sim_full_v1/green_sm_sim_control_paired_v1/**`
- Create: `artifacts/full-corpus-logs/full-v1-handoff-summary.json`

**Consumes:** The two completed roots from Task 2.

**Produces:** A compact, reproducible handoff summary whose values are taken from the persisted reports/manifests, not console impressions.

- [ ] **Step 1: Read both quality reports and assert every root passed its readiness gates.**

  ```powershell
  python -c "import json; from pathlib import Path; base=Path('data/processed/green_sm_sim_full_v1'); roots=['green_sm_sim_linked_v1','green_sm_sim_control_paired_v1']; reports={name: json.loads((base/name/'reports'/'data_quality_report.json').read_text()) for name in roots}; assert all(r['readiness_passed'] for r in reports.values()); print(json.dumps({n:{'ready':r['readiness_passed'],'snapshots':r['snapshot_counts'],'food':r['food'],'ads':r['ads'],'oov':r['closed_vocab_oov_rate'],'geo_rate':r['valid_geo_rate']} for n,r in reports.items()}, indent=2))"
  ```

  Expected for each root: `readiness_passed` is true; snapshot counts are train 96,000 / validation 24,000 / test 24,000; OOV is no greater than 0.01; geo rate is at least 0.60; Food and Ads metrics meet the declared thresholds.

- [ ] **Step 2: Check exact downstream candidate-row counts from the persisted task manifests.**

  ```powershell
  @'
  import json
  from pathlib import Path

  base = Path("data/processed/green_sm_sim_full_v1")
  expected = {
      "food": {"train": 360000, "validation": 60000, "test": 60000},
      "ads": {"train": 240000, "validation": 40000, "test": 40000},
  }
  for root in ("green_sm_sim_linked_v1", "green_sm_sim_control_paired_v1"):
      for task, counts in expected.items():
          manifest = json.loads((base / root / f"{task}_opportunities" / "manifest.json").read_text())
          observed = {split: manifest["files"][split]["rows"] for split in counts}
          assert observed == counts, (root, task, observed)
          print(root, task, observed)
  '@ | python -
  ```

  Expected: Food has 360,000 / 60,000 / 60,000 candidate rows and Ads has 240,000 / 40,000 / 40,000 candidate rows for train / validation / test in both roots.

- [ ] **Step 3: Check the control audit and capture the immutable handoff fields.**

  ```powershell
  python -c "import json; from pathlib import Path; base=Path('data/processed/green_sm_sim_full_v1'); linked=json.loads((base/'green_sm_sim_linked_v1'/'reports'/'data_quality_report.json').read_text()); control=json.loads((base/'green_sm_sim_control_paired_v1'/'reports'/'data_quality_report.json').read_text()); audit=json.loads((base/'green_sm_sim_control_paired_v1'/'reports'/'causal_audit_report.json').read_text()); assert linked['target_world_fingerprint']==control['target_world_fingerprint']; assert linked['generator_config_digest']==control['generator_config_digest']; assert audit['paired_logical_rows_equal'] is True; summary={'data_version':'green_sm_sim_full_v1','config_digest':linked['generator_config_digest'],'target_world_fingerprint':linked['target_world_fingerprint'],'linked_foundation_manifest_sha256':linked['foundation_manifest_sha256'],'control_foundation_manifest_sha256':control['foundation_manifest_sha256'],'linked_preprocessing_config_sha256':linked['preprocessing_config_sha256'],'control_preprocessing_config_sha256':control['preprocessing_config_sha256']}; Path('artifacts/full-corpus-logs/full-v1-handoff-summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))"
  ```

  Expected: target-world and generator-config identities match; Foundation manifest checksums differ because roots differ; the paired control audit is true. The summary is the source of truth for later pretraining and downstream handoff.

- [ ] **Step 4: Inspect the final size before choosing an upload mechanism.**

  ```powershell
  Get-ChildItem data/processed/green_sm_sim_full_v1 -File -Recurse |
    Measure-Object -Property Length -Sum |
    Select-Object Count, @{Name='SizeGiB';Expression={[math]::Round($_.Sum / 1GB, 3)}}
  ```

  Expected: record the actual count and size in the experiment notes. Do not guess from pilot size when configuring Kaggle upload limits.

### Task 4: Hand off the immutable corpus to Kaggle without leaking unrelated artifacts

**Files:**
- Upload source: `data/processed/green_sm_sim_full_v1/`
- Upload source: `artifacts/full-corpus-logs/full-v1-handoff-summary.json`
- Do not upload: repository source, pilot data, `artifacts/` checkpoints, `__pycache__/`, `.git/`, or any private local files.

**Consumes:** The Task 3 passed reports and summary.

**Produces:** A versioned Kaggle Dataset containing the full paired corpus and evidence required to identify it.

- [ ] **Step 1: Make a staging directory containing only corpus data plus the handoff summary.**

  Use Windows Explorer to copy `data/processed/green_sm_sim_full_v1` into an empty staging directory outside the repository, then place a copy of `full-v1-handoff-summary.json` beside it. This avoids accidentally uploading source code or training artifacts.

  Expected staged layout:

  ```text
  green-sm-sim-full-v1/
    green_sm_sim_full_v1/
      green_sm_sim_linked_v1/
      green_sm_sim_control_paired_v1/
    full-v1-handoff-summary.json
  ```

- [ ] **Step 2: Create a new private Kaggle Dataset version from that staging directory.**

  Name it `green-sm-sim-full-v1`. Keep the data private while the work is experimental. Its description must state: synthetic data; linked root is training; paired-control root is for control/ablation; dataset identity is the recorded config digest and target-world fingerprint.

  Expected: Kaggle displays both roots and the two report files under each root after upload.

- [ ] **Step 3: Perform a Kaggle read-only intake check before launching full pretraining.**

  In a fresh Kaggle notebook with the dataset attached, run:

  ```python
  from pathlib import Path
  import json

  matches = list(Path('/kaggle/input').rglob('green_sm_sim_full_v1'))
  assert len(matches) == 1, matches
  root = matches[0]
  report = json.loads((root / 'green_sm_sim_linked_v1' / 'reports' / 'data_quality_report.json').read_text())
  assert report['readiness_passed'] is True
  assert report['snapshot_counts'] == {'train': 96000, 'validation': 24000, 'test': 24000}
  print(root, report['foundation_manifest_sha256'])
  ```

  Expected: one unambiguous dataset root, passed readiness report, and the same linked Foundation manifest SHA recorded in the handoff summary.

### Task 5: Bind later training to the new corpus version

**Files:**
- Read: `configs/pretrain_green_sm_full.json`
- Later modify at Kaggle runtime only: full-pretraining configuration paths
- Later modify after pretraining: Food/Ads runtime configuration checkpoint path and SHA

**Consumes:** A verified Kaggle Dataset from Task 4.

**Produces:** Clear next steps without silently reusing the pilot checkpoint or pilot opportunity files.

- [ ] **Step 1: Before full pretraining, point the runtime config only at the uploaded linked v1 `foundation/` and paired-control `foundation/` roots.**

  `configs/pretrain_green_sm_full.json` already declares local equivalents of:

  ```text
  .../green_sm_sim_full_v1/green_sm_sim_linked_v1/foundation
  .../green_sm_sim_full_v1/green_sm_sim_control_paired_v1/foundation
  ```

  In Kaggle, replace these only in a copied runtime config with the mounted `/kaggle/input/...` paths. Preserve the original config in Git and retain the full corpus manifests with the run artifacts.

- [ ] **Step 2: Do not run Food or Ads against the pilot Foundation checkpoint after this handoff.**

  Full Food/Ads training begins only after full Foundation pretraining selects a verified `best.pt` with `best_manifest.json` and `vocabulary.json`. Then bind its SHA and the corresponding full opportunity root into each downstream runtime configuration.

- [ ] **Step 3: Record the first full experiment as a distinct run.**

  Use new run IDs such as `foundation-full-v1-r1`, `food-full-v1-r1`, and `ads-full-v1-r1`; never place their checkpoints in `pilot-v2-*` directories. This preserves the interpretability of pilot smoke results and prevents artifact collisions.

## Execution Handoff

This plan changes no source code; it uses the existing deterministic full simulator. The critical decision is operational: generate locally on CPU, then upload the immutable output to Kaggle for GPU training. After you review the plan, the next action is Task 1 and then the one full generation command in Task 2.
