This repository does not include raw data, intermediate results, or model weights.

Please prepare the following structure under this directory before running the pipeline:

- reports_batch_TCGA/      PRISM-generated WSI-level draft reports, one `<slide_id>.txt` per slide
- tcga_stad_reports/       (optional) cached patch-level reports
- features_conch_v1/       CONCH patch features, `<slide_id>.h5`
- patches/                 patch coordinates, `<slide_id>_patches.h5`
- TCGA-STAD/          original whole-slide images, `<slide_id>.svs`
- tcga_reports_cleaned.csv evaluation ground truth (columns: slide_id, slide_reports; from TITAN for TCGA-STAD)

How to produce these inputs:
- CONCH features/coordinates: run the official CONCH feature-extraction pipeline.
- PRISM draft reports: run the official PRISM WSI-level model per slide.
- These upstream tools are not bundled in this repository.

Notes:
- Do not commit patient data.
- Do not commit model checkpoints or large binary artifacts.
