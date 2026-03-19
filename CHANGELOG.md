# Changelog

## 2026-03-18 — James
- Added `09_neurips-papers.md` — 5 NeurIPS papers for Related Work (gated attention, RL reasoning, scaling laws, FineWeb, DCLM)
- Created `docs/parameter-golf/` folder for competition participation (overview, experiments tracker, findings)
- Added `08_parameter-golf.md` — OpenAI Parameter Golf challenge technical reference (related work)
- Numbered all James_notes files (01-07) with reading order prefixes
- Created `00_table-of-contents.md` as entry point for research notes
- Updated README.md and CLAUDE.md links to match new filenames
- Added "Metrics & Evaluation" section to `key-terms.md` (learning curve + perplexity definitions)
- Added learning curves as required result in `final-paper.md` (required by TAs)
- Renamed "Learning curve" column to "Ease of use" in `comparison-table.md` to avoid confusion with the ML metric
- Added metrics logging requirements to each stage in `training-plan.md`
- Added monitoring section to `training-stages.md`

## 2026-03-12 — James (update)
- Added `docs/official/requirements/` — split into proposal and final paper requirement templates
- Added `docs/official/training-plan.md` — nanochat 4-stage training plan template with experiments tracker
- Updated `docs/official/README.md` — added links to new docs and expanded decisions log

## 2026-03-12 — James
- Set up repo structure: `docs/James_notes/`
- Researched nanoGPT, llm.c, and nanochat frameworks
- Created comparison table (nanoGPT vs llm.c vs nanochat)
- Created detailed research notes for all 3 frameworks
- Created datasets & benchmarks comparison across all 3
- Created key terms glossary (BPE, SFT, RLHF, etc.)
- Created nanochat training stages deep dive (4 stages, datasets, pipeline visualization)
- Updated README with nanochat pipeline overview and links to notes
- Decision: nanochat is the best framework to focus on (full pipeline, active, comprehensive)
