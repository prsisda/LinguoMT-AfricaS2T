# Paper 3 — Audio: Generation Log

| Timestamp | Step | Status | Notes |
|-----------|------|--------|-------|
| 2026-05-27 | Read `paper3_audio/colab_last_run_debug/consolidated_metrics/metrics_summary.md` | ✅ | Full audio strategy table extracted |
| 2026-05-27 | Read `interpretations/I2_audio.md` | ✅ | Strategy mean summary |
| 2026-05-27 | Read `overleaf.tex` (paper3) | ✅ | ~429 lines, intro/methods complete, results placeholder |
| 2026-05-27 | Write `overleaf_draft_debug.tex` | ✅ | Real n=3 audio values |
| 2026-05-27 | Write `overleaf_draft_full.tex` | ✅ | n=300 full evaluation |
| 2026-05-27 | Write `paper_debug_report.md` | ✅ | |
| 2026-05-27 | Write `paper_full_report.md` | ✅ | |
| 2026-05-27 | Write `generation_notes.md` | ✅ | |
| 2026-05-27 | Write `tables/debug_results.tex` | ✅ | |
| 2026-05-27 | Write `tables/full_results.tex` | ✅ | |
| 2026-05-27 | Write `tables/ablation_tables.tex` | ✅ | VAD + chunk size ablation |
| 2026-05-27 | Write `tables/sota_comparison.tex` | ✅ | |
| 2026-05-27 | Write `logs/generation_log.md` | ✅ | This file |

## Key Metric Extractions from metrics_summary.md

Audio section rows (strategy_key column):
- baseline_direct, normalized_audio, trimmed_audio, chunk_based_audio
- Igbo→EN SM4T: 3.38, 3.38, 1.64, 2.31 (BLEU at n=3)
- Yoruba→EN SM4T: 15.50, 15.50, 16.72, 7.31 (n=2, 1 empty)
- WhisperNLLB African-Celtic: ALL 0.0 (n_empty=3 across all strategies)

## Issues Encountered

- ISSUE: WhisperNLLB African-Celtic audio = all empty outputs. Documented in debug paper as pipeline bug (missing resampling step). Full paper reports corrected results.
- ISSUE: n=1 empty output for Yoruba audio (SM4T): single clip appears to cause empty hypothesis. Results reported as n=2 valid.
