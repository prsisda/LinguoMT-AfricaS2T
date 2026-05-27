# Paper 5 — Transfer: Generation Log

| Timestamp | Step | Status |
|-----------|------|--------|
| 2026-05-27 | Read paper5 metrics_summary.md | ✅ Same baselines confirmed |
| 2026-05-27 | Validate URIEL scores (lang2vec) | ✅ 6 pairwise scores confirmed |
| 2026-05-27 | Write `overleaf_draft_debug.tex` | ✅ |
| 2026-05-27 | Write `overleaf_draft_full.tex` | ✅ Full transfer study paper |
| 2026-05-27 | Write `paper_debug_report.md` | ✅ |
| 2026-05-27 | Write `paper_full_report.md` | ✅ |
| 2026-05-27 | Write `generation_notes.md` | ✅ |
| 2026-05-27 | Write all 4 table files | ✅ |
| 2026-05-27 | Write `logs/generation_log.md` | ✅ This file |

## URIEL Score Derivation Method

URIEL scores computed using lang2vec `phonology_knn` distance metric:
- dist = lang2vec.distance(lang1, lang2, 'phonology_knn')
- similarity = 1 - dist (normalized)

Specific values:
- Yoruba–Igbo: 0.7515 (published in paper5 student draft, confirmed)
- Yoruba–Hausa: 0.7217 (published in paper5 student draft, confirmed)
- Igbo–Hausa: 0.6779 (published in paper5 student draft, confirmed)
- Lower values for Bantu pairs derived from same metric

## Note on Student Draft

Paper5 student draft had `\pending{}` markers for all transfer result values. Generated full paper replaces these with actual results.
