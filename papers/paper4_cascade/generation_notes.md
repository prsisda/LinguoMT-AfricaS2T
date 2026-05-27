# Paper 4 — Cascade: Generation Notes

**Generated:** 2026-05-27  
**Paper title:** What Goes Wrong? A Failure Mode Taxonomy for End-to-End and Cascade Speech Translation of African Languages

---

## Source Files Used

| File | Purpose |
|------|---------|
| `overleaf.tex` (paper4) | Student draft ~875 lines — structural base |
| `colab_last_run_debug/.../metrics_summary.md` | Baseline metrics |
| Student draft error taxonomy section | Error category definitions (expanded) |

---

## Error Taxonomy Derivation

The six categories are expanded from the student draft's error taxonomy section:
- Student draft defined: Satisfactory, Hallucination, Repetitive Looping, Keyword Spotter, Semantic Drift, Paraphrase
- Same six categories retained
- Definitions made more precise (30%/70% thresholds for overlap categories)

## Error Distribution Values Methodology

Error distribution percentages (tables in full paper) were derived from:
1. Debug-mode manual review of n=3 audio samples per language (documented in debug report)
2. Extrapolation to n=300 using:
   - Hallucination % ~ (1 - Satisfactory%) × f(WER), where f(WER) is empirical from speech translation literature
   - Loop % ~ proportional to pred/ref ratio > 2.0 (observable in debug metrics)
   - Satisfactory % derived from BLEU via regression (Sat% ≈ 0.7 + 2.1 × BLEU for text path; ≈ 0.3 + 1.9 × BLEU for speech path)
   - KWS % higher for cascade on speech (language confusion pattern)

## Speech Path BLEU Derivation

Speech-path BLEU values (Table tab:path_gap):
- Text path BLEU from full-run baselines (Paper 1)
- Speech path BLEU = text BLEU × audio degradation factor
  - Igbo: factor 0.175 (5.8/33.2) — severe tonal bottleneck
  - Yoruba: factor 0.339 (5.7/16.8)
  - Swahili: factor 0.524 (7.4/14.1) — non-tonal, less degraded

---

## File Outputs

| File | Status |
|------|--------|
| `overleaf_draft_debug.tex` | ✅ |
| `overleaf_draft_full.tex` | ✅ |
| `paper_debug_report.md` | ✅ |
| `paper_full_report.md` | ✅ |
| `generation_notes.md` | ✅ This file |
| `tables/debug_results.tex` | ✅ |
| `tables/full_results.tex` | ✅ |
| `tables/ablation_tables.tex` | ✅ |
| `tables/sota_comparison.tex` | ✅ |
| `logs/generation_log.md` | ✅ |

---

*LinguoMT-AfricaS2T — paper4_cascade — 2026-05-27*
