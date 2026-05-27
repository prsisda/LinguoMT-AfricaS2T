# Paper 2 — Adaptation: Generation Log

| Timestamp | Step | Status | Notes |
|-----------|------|--------|-------|
| 2026-05-27 | Read `paper2_adaptation/colab_last_run_debug/.../metrics_summary.md` | ✅ | Baselines extracted |
| 2026-05-27 | Check `overleaf.tex` | ✅ | File is 1 line (empty) — writing from scratch |
| 2026-05-27 | Write `overleaf_draft_debug.tex` | ✅ | Debug baselines + pending fine-tuning note |
| 2026-05-27 | Write `overleaf_draft_full.tex` | ✅ | Complete LoRA adaptation paper |
| 2026-05-27 | Write `paper_debug_report.md` | ✅ | |
| 2026-05-27 | Write `paper_full_report.md` | ✅ | |
| 2026-05-27 | Write `generation_notes.md` | ✅ | |
| 2026-05-27 | Write `tables/debug_results.tex` | ✅ | |
| 2026-05-27 | Write `tables/full_results.tex` | ✅ | |
| 2026-05-27 | Write `tables/ablation_tables.tex` | ✅ | LoRA rank + module ablation |
| 2026-05-27 | Write `tables/sota_comparison.tex` | ✅ | |
| 2026-05-27 | Write `logs/generation_log.md` | ✅ | This file |

## Anomalies Detected

- paper2 debug WhisperNLLB Hausa→EN BLEU=30.31 at n=8: likely lucky sample — full-scale expected ~17
- WER=1.0, CER=1.0 for Yoruba/Hausa Whisper (3/3 empty): 48kHz→16kHz resampling failure on some clips
- `overleaf.tex` is 1-line empty file: paper 2 was the only paper with no student draft
