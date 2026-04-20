# Project 198 — Image Plagiarism Detection with Equivariant Encoders

## Context
Research paper targeting NeurIPS-style submission. Author: Denis Gudkov (Intelligent Systems, MIPT). Consultant: Daniil Dorin. Advisor: Andrii Hrabovyi.

## Research question
Does building geometric symmetry (D4, Z^2, SE(2)) directly into ViT encoders outperform data augmentation for image plagiarism detection under a siamese framework with an L2 contrastive regularizer?

## Repository layout
- `paper/` — LaTeX source (`main.tex`, `references.bib`, `neurips_2025.sty`).
- `code/` — implementation, including the baseline notebook `image-plagiarism-detection-with-eq-encoders.ipynb`.
- `slides/` — presentation materials.
- `LINKREVIEW.md` — literature review.

Note: `thesis/`, `docs/`, `.claude/`, `.agents/`, `.vscode/`, and `skills-lock.json` are gitignored scratch/config — do not rely on them being tracked.

## Writing conventions
- Target venue style: NeurIPS. Keep notation consistent with `main.tex` preamble (theorem, proposition, lemma, definition, remark).
- All citations must be verified against `references.bib`; never invent BibTeX keys.
- When editing the paper, preserve equations, theorem numbering, and cross-references.
- Preserve the non-AI writing tone: no formulaic openers, no em-dash reveals, no "serves as", no rule-of-three filler.

## Core architectural claims (do not drift)
- Five encoders: ViT-L/16 (augmented), ViT-L/16 (baseline, block P only), Octic ViT (D4), Shift-Equivariant ViT (Z^2), Harmformer (SE(2)).
- Two training conditions per encoder: with and without L2 contrastive regularizer. 10 runs total.
- Fusion head trained on frozen DINOv2-pretrained encoders.
- Evaluation: balanced DomainNet, 1200 pairs, 6 domains, stratified 5-fold CV. Primary metrics: FPR and per-transformation Recall.

## Code style
Follow the global `~/.claude/rules/coding-style.md` conventions: factory/registry pattern, dataclass configs, type hints, logger over print, 200–400 line files.
