# Paper Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `paper/main.tex` from scratch so that each section satisfies the corresponding m1p.org course deliverable (W1–W6, excluding O* and R), using the approved spec at `.claude/specs/2026-04-12-paper-rewrite-design.md`.

**Architecture:** Single LaTeX file (`paper/main.tex`) using NeurIPS 2025 template (`neurips_2025.sty`). All references in `paper/references.bib`. Content driven by `thesis/tesis.ipynb` (Introduction source), the approved spec (experiment design), and `code/image-plagiarism-detection-with-eq-encoders.ipynb` (training pipeline details). Use `superpowers:scientific-writing` skill for prose generation in each task.

**Tech Stack:** LaTeX (pdflatex + bibtex), NeurIPS 2025 template, amsmath/amssymb/amsthm, booktabs, graphicx, hyperref.

---

## File Map

- **Modify:** `paper/main.tex` — complete rewrite (all content)
- **Modify:** `paper/references.bib` — add missing entries, clean up unused ones
- **Keep as-is:** `paper/neurips_2025.sty`

---

### Task 1: Preamble and Document Skeleton

**Files:**
- Modify: `paper/main.tex:1-475` (full rewrite)

- [ ] **Step 1: Write the LaTeX preamble**

Keep existing packages. Remove `algorithm`, `algorithmicx`, `algpseudocode`, `listings` (no pseudocode needed). Keep `amsmath`, `amssymb`, `amsthm`, `booktabs`, `graphicx`, `hyperref`, `subcaption`, `multirow`, `microtype`, `xcolor`. Keep theorem environments (`theorem`, `proposition`, `lemma`, `definition`, `remark`). Update `\title` to: `Image Plagiarism Detection with Equivariant Encoders`.

```latex
\documentclass{article}
\PassOptionsToPackage{numbers, sort, compress}{natbib}
\usepackage[main,final]{neurips_2025}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{subcaption}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}

\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}{Lemma}
\newtheorem{definition}{Definition}
\newtheorem{remark}{Remark}

\title{Image Plagiarism Detection with Equivariant Encoders}

\author{
  Denis Gudkov\\
  Intelligent Systems\\
  \texttt{email@gmail.com}\\
}

\begin{document}
\maketitle
```

- [ ] **Step 2: Write the section skeleton**

After `\maketitle`, lay out empty sections as placeholders to be filled in subsequent tasks:

```latex
\begin{abstract}
% Task 2
\end{abstract}

\section{Introduction}\label{sec:intro}
% Task 3

\section{Related Work}\label{sec:rw}
% Task 4

\section{Problem Statement}\label{sec:problem}
% Task 5

\section{Preliminaries: Equivariance}\label{sec:prelim}
% Task 6

\section{Method}\label{sec:method}
% Task 7

\section{Experiments}\label{sec:exp}
% Task 8

\section{Conclusion}\label{sec:concl}
% Task 9

\bibliographystyle{unsrtnat}
\bibliography{references}

\end{document}
```

- [ ] **Step 3: Verify skeleton compiles**

Run: `cd /Users/den1shh/Documents/2026-Project-198/paper && pdflatex -interaction=nonstopmode main.tex`
Expected: Compiles with no errors (warnings about empty references are OK at this stage).

- [ ] **Step 4: Commit**

```bash
git add paper/main.tex
git commit -m "refactor: replace paper/main.tex with clean skeleton for rewrite"
```

---

### Task 2: Abstract (W2.A)

**Files:**
- Modify: `paper/main.tex` — replace abstract placeholder

**Requirements (m1p.org W2.A):** ≤600 characters, 5 elements: problem field, narrow focus, method features, novelty, application.

- [ ] **Step 1: Write abstract**

Use `superpowers:scientific-writing` skill. Replace the abstract placeholder with flowing prose covering all 5 elements. Key content:

1. **Problem field:** Image plagiarism detection requires robustness to geometric transformations.
2. **Narrow focus:** Whether geometric inductive bias in the encoder outperforms data augmentation.
3. **Method features:** Siamese network with equivariant ViT encoders (D₄, ℤ², SE(2)) and L² contrastive regularizer.
4. **Novelty:** First controlled comparison of three symmetry groups inside one siamese framework, evaluated with FPR and Recall on DomainNet.
5. **Application:** Plagiarism screening in journalism, academia, design.

Constraint: ≤600 characters total. Scientific style, no filler.

- [ ] **Step 2: Verify character count**

Run: `cd /Users/den1shh/Documents/2026-Project-198/paper && sed -n '/\\begin{abstract}/,/\\end{abstract}/p' main.tex | grep -v '\\begin\|\\end' | wc -c`
Expected: ≤600.

- [ ] **Step 3: Compile check**

Run: `cd /Users/den1shh/Documents/2026-Project-198/paper && pdflatex -interaction=nonstopmode main.tex`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add paper/main.tex
git commit -m "feat: add abstract (W2.A, ≤600 chars, 5 elements)"
```

---

### Task 3: Introduction (W3.I)

**Files:**
- Modify: `paper/main.tex` — replace §1 placeholder

**Requirements (m1p.org W3.I):** 8 numbered elements, Chief Editor formula, Table 1 (strengths/weaknesses).

- [ ] **Step 1: Write Introduction prose**

Use `superpowers:scientific-writing` skill. Translate and expand `thesis/tesis.ipynb` into English. Structure as flowing paragraphs that naturally cover all 8 elements (do NOT number them explicitly in text — they should emerge from the narrative):

1. **Goal:** Evaluate whether built-in G-invariance improves plagiarism detection over augmentation.
2. **Object:** Pairs of natural images under D₄/ℤ²/SE(2) + photometric perturbations.
3. **Challenge:** ViTs lack geometric invariance; augmentation only approximates robustness; scaling doesn't help (cite chen2024spatialvlm).
4. **State of the art:** Octic ViT (D₄, Nordström et al.), Shift-Equivariant ViT (ℤ², Rojas-Gomez et al.), Harmformer (SE(2), Karella et al.), DINOv2, siamese detectors (Dorin et al.).
5. **Project tasks:** (i) five encoder configurations in one siamese framework; (ii) DINOv2 pre-training; (iii) each ±L² contrastive; (iv) FPR+Recall on DomainNet.
6. **Chief Editor formula:** Embed naturally: *"We propose a siamese plagiarism detector with G-equivariant ViT encoders, providing exact invariance under D₄, ℤ² or SE(2) without geometric augmentation, distinguished from prior work by an L² contrastive regularizer and a controlled ablation of geometric augmentation vs equivariant inductive bias."*
7. **Table 1: Strengths and weaknesses.**
8. **Experimental goal:** 10 runs = 5 encoders × {±L²}; primary FPR + Recall.

- [ ] **Step 2: Add Table 1**

```latex
\begin{table}[t]
\centering
\caption{Comparison of recent approaches to geometric robustness in image matching.}
\label{tab:comparison}
\begin{tabular}{@{}lllll@{}}
\toprule
Method & Group & Exactness & Strengths & Weaknesses \\
\midrule
Dorin et al. & None & Approximate & Augmentation-based; flexible & No invariance guarantee \\
Octic ViT & $D_4$ & Exact & 40\% FLOPs reduction & Discrete; $90^\circ$ only \\
Shift-Eq.\ ViT & $\mathbb{Z}^2$ & Exact & Exact shift invariance & No rotation equivariance \\
Harmformer & $SE(2)$ & Exact & Arbitrary angles + shifts & Higher compute cost \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 3: Add outline paragraph at end**

End with a brief paragraph: "Section 2 reviews... Section 3 formalizes... etc."

- [ ] **Step 4: Compile check**

Run: `cd /Users/den1shh/Documents/2026-Project-198/paper && pdflatex -interaction=nonstopmode main.tex`
Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add paper/main.tex
git commit -m "feat: add Introduction (W3.I, 8 elements, Table 1, Chief Editor formula)"
```

---

### Task 4: Related Work (W2.L)

**Files:**
- Modify: `paper/main.tex` — replace §2 placeholder
- Modify: `paper/references.bib` — add any missing entries

**Requirements (m1p.org W2.L):** Thematic grouping, ≥9 key works. Based on `LINKREVIEW.md`.

- [ ] **Step 1: Write Related Work**

Use `superpowers:scientific-writing` skill. Four thematic subsections as flowing paragraphs (use `\paragraph{}` or `\textbf{}` for sub-headings):

**Siamese detectors for image matching and plagiarism.** Bromley et al. (1993) introduced siamese networks. Dorin et al. (2024) applied siamese ViT + contrastive loss to plagiarism detection on DomainNet; their evaluation used FPR as primary metric. Ke et al. (2004) on classical near-duplicate detection.

**Equivariant and invariant architectures.** Cohen & Welling (2016) G-CNNs. Weiler & Cesa (2019) steerable E(2)-equivariant CNNs. Bronstein et al. (2021) unifying geometric deep learning framework. These target classification/segmentation — we apply them to pairwise similarity.

**Equivariant Vision Transformers.** Nordström et al. (2025) Octic ViT with D₄-equivariant layers. Rojas-Gomez et al. (2024) shift-equivariant ViT via adaptive polyphase sampling. Karella et al. (2024) Harmformer for continuous roto-translation equivariance. No prior work compares these on pairwise similarity.

**Self-supervised ViT pre-training.** DINOv2 (Oquab et al., 2024) produces strong features via self-supervised learning on ImageNet-1K. We use it as a common pre-training recipe for fair comparison.

- [ ] **Step 2: Update references.bib**

Verify all cited keys exist in `paper/references.bib`. Add missing entries:
- `hadsell2006dimensionality` (L² contrastive loss)
- `peng2019domainnet` (DomainNet dataset)
- `chopra2005learning` (contrastive loss)

Standardize existing citation keys if needed (e.g., `nordstrom2025octicvisiontransformersquicker` → `nordstrom2025octic` for consistency, updating all `\cite{}` calls).

```bibtex
@inproceedings{hadsell2006dimensionality,
  title     = {Dimensionality Reduction by Learning an Invariant Mapping},
  author    = {Hadsell, Raia and Chopra, Sumit and LeCun, Yann},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1735--1742},
  year      = {2006},
}

@inproceedings{peng2019domainnet,
  title     = {Moment Matching for Multi-Source Domain Adaptation},
  author    = {Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {1406--1415},
  year      = {2019},
}

@inproceedings{chopra2005learning,
  title     = {Learning a Similarity Metric Discriminatively, with Application to Face Verification},
  author    = {Chopra, Sumit and Hadsell, Raia and LeCun, Yann},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {539--546},
  year      = {2005},
}
```

- [ ] **Step 3: Compile check with bibtex**

Run: `cd /Users/den1shh/Documents/2026-Project-198/paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex`
Expected: No errors; all citations resolved.

- [ ] **Step 4: Commit**

```bash
git add paper/main.tex paper/references.bib
git commit -m "feat: add Related Work (W2.L, 4 thematic groups, ≥9 references)"
```

---

### Task 5: Problem Statement (W3.P)

**Files:**
- Modify: `paper/main.tex` — replace §3 placeholder

**Requirements (m1p.org W3.P):** 12 explicitly numbered elements.

- [ ] **Step 1: Write Problem Statement**

Use `superpowers:scientific-writing` skill. Write as numbered items (this section benefits from explicit numbering per m1p.org). Each item is a short paragraph.

Content from the approved spec §3:

1. **General problem.** Binary: is $x_2$ a plagiarised derivative of $x_1$?
2. **Sample set.** $\mathcal{D}=\{(x_1^{(i)},x_2^{(i)},y^{(i)})\}_{i=1}^N$, $y\in\{0,1\}$.
3. **Statistical hypothesis.** $p_+(x_1,x_2)=\int p(x)\,p_T(x_2\mid x_1)\,dx$, $p_T$ mixture over $G\cup G_\text{photo}$.
4. **Measurement conditions.** DomainNet, 6 domains, $224\times224$ 8-bit RGB.
5. **Sample restrictions.** Positive: source + transformed copy. Negative: distinct sources, same domain.
6. **Model.** $f_\theta(x_1,x_2)=\sigma(h_\psi(\varphi_\theta(x_1),\varphi_\theta(x_2)))$.
7. **Class restrictions.** $\varphi_\theta$ G-equivariant with G-invariant pooling; $h_\psi$ symmetric.
8. **Loss.** $\mathcal{L}=\text{BCE}+\lambda\,\mathcal{L}_\text{contr}^{L^2}$ ($\lambda=0$ in variant without regularization).
9. **CV procedure.** Stratified by DomainNet domain, five-fold.
10. **Solution restrictions.** Single GPU ≤48 GB; ≤30 epochs.
11. **External quality criteria.** Primary: FPR, Recall (stratified by transformation). Secondary: F1, t-SNE.
12. **Argmin.** $\hat\theta=\arg\min_\theta\,\mathbb{E}\,\mathcal{L}(\theta)$.

Use consistent notation: $x$ for images (not $\mathbf{I}$), $z$ for embeddings (not $\mathbf{z}$), $\varphi_\theta$ for encoder (not $g_\omega$).

- [ ] **Step 2: Compile check**

Run: `cd /Users/den1shh/Documents/2026-Project-198/paper && pdflatex -interaction=nonstopmode main.tex`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add paper/main.tex
git commit -m "feat: add Problem Statement (W3.P, 12 elements)"
```

---

### Task 6: Preliminaries — Equivariance (W6, theory)

**Files:**
- Modify: `paper/main.tex` — replace §4 placeholder

**Requirements (m1p.org W6):** Theory section with formal definitions. This section covers equivariance as a narrow concept (user's explicit request).

- [ ] **Step 1: Write subsection 4.1 — Group actions and representations**

Definitions of group action on a set, left action on image space, group representation $\rho: G \to GL(V)$.

```latex
\begin{definition}[Group action]
A \emph{left action} of a group $G$ on a set $\mathcal{X}$ is a map $G \times \mathcal{X} \to \mathcal{X}$, written $(g, x) \mapsto g \cdot x$, satisfying $e \cdot x = x$ and $(gh) \cdot x = g \cdot (h \cdot x)$ for all $g, h \in G$, $x \in \mathcal{X}$.
\end{definition}

\begin{definition}[Group representation]
A \emph{representation} of $G$ on a vector space $V$ is a group homomorphism $\rho \colon G \to GL(V)$.
\end{definition}
```

- [ ] **Step 2: Write subsection 4.2 — Equivariance and invariance**

```latex
\begin{definition}[Equivariance]
A map $\varphi \colon \mathcal{X} \to \mathcal{Y}$ is \emph{$G$-equivariant} with respect to representations $\rho_\mathcal{X}$ and $\rho_\mathcal{Y}$ if $\varphi(\rho_\mathcal{X}(g) \cdot x) = \rho_\mathcal{Y}(g) \cdot \varphi(x)$ for all $g \in G$, $x \in \mathcal{X}$.
\end{definition}

\begin{definition}[Invariance]
$\varphi$ is \emph{$G$-invariant} if $\rho_\mathcal{Y}$ is the trivial representation: $\varphi(\rho_\mathcal{X}(g) \cdot x) = \varphi(x)$ for all $g \in G$.
\end{definition}
```

Add 1–2 sentences of prose: invariance is a special case of equivariance. For plagiarism detection, a G-invariant encoder guarantees identical embeddings regardless of which $g \in G$ was applied.

- [ ] **Step 3: Write subsection 4.3 — Irreducible representations and Schur's lemma**

State Schur's lemma (without proof — it's classical). Explain its consequence: equivariant linear maps between representations of a finite group decompose into block-diagonal form w.r.t. irreducible subspaces. This motivates the architecture of Octic ViT.

```latex
\begin{lemma}[Schur]
Let $\rho_1 \colon G \to GL(V_1)$ and $\rho_2 \colon G \to GL(V_2)$ be irreducible representations. Any $G$-equivariant linear map $\varphi \colon V_1 \to V_2$ is either zero or an isomorphism.
\end{lemma}
```

One paragraph of consequence text.

- [ ] **Step 4: Write subsection 4.4 — Groups considered**

Three paragraphs, one per group:

**$D_4$:** Dihedral group of order 8, generated by $r$ (90° rotation) and $s$ (reflection), with $r^4 = s^2 = e$, $srs = r^{-1}$. Five irreducible representations: four 1-dimensional ($A_1, A_2, B_1, B_2$) and one 2-dimensional ($E$). Acts on images by rotating/reflecting spatial axes.

**$\mathbb{Z}^2$:** Countable group of integer translations acting by cyclic pixel shifts: $(t_{a,b} \cdot x)[i,j,c] = x[(i+a) \bmod H, (j+b) \bmod W, c]$. Relevant when plagiarists crop and re-center images.

**$SE(2)$:** Special Euclidean group of roto-translations, $SE(2) = SO(2) \ltimes \mathbb{R}^2$. An element $(R_\alpha, t) \in SE(2)$ acts by rotating coordinates by angle $\alpha$ and translating by $t$. Continuous group; handles arbitrary rotation angles unlike $D_4$.

- [ ] **Step 5: Compile check**

Run: `cd /Users/den1shh/Documents/2026-Project-198/paper && pdflatex -interaction=nonstopmode main.tex`
Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add paper/main.tex
git commit -m "feat: add Preliminaries on equivariance (W6, definitions, Schur, 3 groups)"
```

---

### Task 7: Method (W6, theory of the solution)

**Files:**
- Modify: `paper/main.tex` — replace §5 placeholder

**Requirements (m1p.org W6):** Theory of the solution + properties. Subsections 5.1–5.6 per spec.

- [ ] **Step 1: Write 5.1 — Siamese framework**

One paragraph: two branches with shared weights $\varphi_\theta$, fusion head $h_\psi$, sigmoid output. Cite Bromley et al. (1993). Use notation from §3: $f_\theta(x_1, x_2) = \sigma(h_\psi(\varphi_\theta(x_1), \varphi_\theta(x_2)))$.

- [ ] **Step 2: Write 5.2 — Encoders**

Five encoder configurations, each as a `\paragraph{}`:

**ViT-L/16 (augmented).** Baseline with block P + G augmentations. Standard ViT-L/16 (Dosovitskiy et al., 2021): 196 patches of 16×16, 24 transformer blocks, [CLS] token → $z \in \mathbb{R}^{1024}$. No equivariance guarantee. Receives both photometric (block P) and geometric (block G) augmentations during training.

**ViT-L/16 (baseline).** Same architecture, only block P augmentations (no geometric). Serves as ablation: tests whether geometric augmentations are necessary without equivariant inductive bias.

**Octic ViT ($D_4$-equivariant).** Nordström et al. (2025). By Schur's lemma (§4.3), equivariant linear maps decompose as block-diagonal w.r.t. the 5 irreps of $D_4$: $\mathbb{R}^d \cong (\mathbb{R}^{d/8})_{A_1} \oplus (\mathbb{R}^{d/8})_{A_2} \oplus (\mathbb{R}^{d/8})_{B_1} \oplus (\mathbb{R}^{d/8})_{B_2} \oplus (\mathbb{R}^{d/4})_E$. Invariant pooling projects onto $A_1$, yielding $D_4$-invariant output. Only block P augmentations.

**Shift-Equivariant ViT ($\mathbb{Z}^2$-equivariant).** Rojas-Gomez et al. (2024). Replaces four ViT modules: A-token (polyphase offset selection), A-WSA (adaptive window partition), A-PMerge (polyphase downsampling), A-RPE (circular relative position encoding). Only block P augmentations.

**Harmformer ($SE(2)$-equivariant).** Karella et al. (2024). Combines circular harmonic filters with transformer blocks, achieving equivariance to continuous rotations and translations. Unlike D₄-restricted models, handles arbitrary angles. Only block P augmentations.

All encoders pre-trained on ImageNet-1K with DINOv2 (Oquab et al., 2024).

- [ ] **Step 3: Write 5.3 — Fusion module**

Equation: $h_\psi(z_1, z_2) = W \, \text{ReLU}[|z_1 - z_2| \| z_1 \odot z_2]$. Symmetric by construction.

- [ ] **Step 4: Write 5.4 — L² contrastive regularizer**

Equation: $\mathcal{L}_\text{contr}^{L^2} = y \|z_1 - z_2\|_2^2 + (1-y) \max(0, m - \|z_1 - z_2\|_2)^2$.

One paragraph explaining the choice of L² over cosine (cite Hadsell et al. 2006): L² distance is isometry-invariant, making it naturally compatible with G-invariant encoders. Cosine similarity depends on vector norms, which may break the alignment with equivariant geometry.

For each encoder, we compare λ > 0 (with L² contrastive) vs λ = 0 (without).

- [ ] **Step 5: Write 5.5 — Augmentation blocks**

**Block P (photometric):** Applied to ALL encoders. ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), RandomGrayscale(p=0.05), GaussianBlur(kernel=3, p=0.1), ImageNet normalization.

**Block G (geometric):** Applied ONLY to ViT-L/16 (augmented). RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5), RandomChoice{0°, 90°, 180°, 270°}, RandomResizedCrop(scale=0.7–1.0, ratio=0.9–1.1).

Present as a compact table or enumeration. Explain rationale: equivariant encoders handle geometric transformations by construction, so block G is redundant for them — this is the core hypothesis.

- [ ] **Step 6: Write 5.6 — Theoretical properties**

Three propositions with short proofs:

**Proposition 1 (Symmetry of fusion).** $h_\psi(z_1, z_2) = h_\psi(z_2, z_1)$.
Proof: $|z_1 - z_2| = |z_2 - z_1|$ and $z_1 \odot z_2 = z_2 \odot z_1$. QED.

**Proposition 2 (Invariance propagation).** If $\varphi_\theta$ is $G$-invariant, then $f_\theta(g \cdot x_1, g \cdot x_2) = f_\theta(x_1, x_2)$ for all $g \in G$.
Proof: $G$-invariance gives $\varphi_\theta(g \cdot x) = \varphi_\theta(x)$. The fusion and head are deterministic functions of embeddings. QED.

**Proposition 3 (Isometry invariance of L² contrastive).** $\mathcal{L}_\text{contr}^{L^2}$ is invariant under isometries of the embedding space: if $U$ is an orthogonal map, replacing $z_i$ with $Uz_i$ leaves the loss unchanged.
Proof: $\|Uz_1 - Uz_2\|_2 = \|U(z_1 - z_2)\|_2 = \|z_1 - z_2\|_2$ since $U$ preserves norms. QED.

- [ ] **Step 7: Compile check**

Run: `cd /Users/den1shh/Documents/2026-Project-198/paper && pdflatex -interaction=nonstopmode main.tex`
Expected: No errors.

- [ ] **Step 8: Commit**

```bash
git add paper/main.tex
git commit -m "feat: add Method section (W6, 5 encoders, fusion, L² contrastive, 3 propositions)"
```

---

### Task 8: Experiments (W4.X + W5.C + W5.V)

**Files:**
- Modify: `paper/main.tex` — replace §6 placeholder

**Requirements:** W4.X (dataset, object/feature counts, workflow, figure/table lists with axes), W5.C (code organization), W5.V (planned visualizations with format specs). No pseudocode (W4.B excluded by user).

- [ ] **Step 1: Write 6.1 — Dataset and pre-training**

**Pre-training:** All encoders use DINOv2 on ImageNet-1K.

**Evaluation dataset:** DomainNet (Peng et al., 2019), 6 domains (real, painting, clipart, quickdraw, infograph, sketch). Balanced test set: 1200 pairs (100 positive + 100 negative per domain). Feature count per image: $3 \times 224 \times 224 = 150{,}528$ scalar features.

- [ ] **Step 2: Write 6.2 — Training pipeline**

Describe in prose (no pseudocode). Pipeline: load base images → form pairs via Dorin-style cross-batch construction (batch of $B$ images → $B^2$ pairs, positives on diagonal) → apply augmentation blocks (P for all, P+G for ViT augmented) → forward through siamese network → compute BCE + λ · L²-contrastive → AdamW step.

**10 runs:** 5 encoders × {λ = 0, λ > 0}.

Hyperparameters table (from notebook):

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | $10^{-3}$ |
| Weight decay | $10^{-2}$ |
| Scheduler | MultiStepLR, milestones [5, 10], γ = 0.5 |
| BCE weights | $w_+ = 0.3$, $w_- = 0.7$ |
| Epochs | 15 |
| Contrastive margin $m$ | 1.0 |
| Contrastive weight $\lambda$ | 0.1 (when enabled) |

- [ ] **Step 3: Write 6.3 — Evaluation protocol**

Workflow: for each (encoder, ±L²) → train → select threshold on validation → compute metrics on test set.

Primary metrics: **FPR** (for comparability with Dorin et al.) and **Recall** stratified by transformation class (CJ, GN, GS, R90, R180, R270, WM, combinations).

Secondary: F1, t-SNE visualization of embeddings across 6 DomainNet domains.

- [ ] **Step 4: Write 6.4 — Code organization (W5.C)**

Target structure:
- `code/main.py` — single entry point
- `code/encoders/` — encoder wrappers
- `code/siamese/` — siamese network + fusion module
- `code/data/` — dataset loading, augmentation blocks P and G
- `code/train/` — training loop
- `code/eval/` — evaluation, metrics, t-SNE
- `code/config.yaml` — centralized hyperparameters
- Current baseline: `code/image-plagiarism-detection-with-eq-encoders.ipynb`

- [ ] **Step 5: Write 6.5 — Planned figures and tables (W5.V)**

List with format specs (.pdf/.eps, 300 DPI, 12–16 pt fonts, 2 pt lines):

- **Fig. 1:** Siamese architecture schema.
- **Fig. 2:** Diagrams of $D_4$, $\mathbb{Z}^2$, $SE(2)$ with example actions on an image.
- **Fig. 3** (source data): Distribution of DomainNet classes. x: domain, y: image count.
- **Fig. 4** (main message): FPR by encoder and regularization. x: encoder, y: FPR, grouped bars {λ=0, λ>0}.
- **Fig. 5** (main message): Recall by transformation class. x: transformation, y: Recall, bars per encoder.
- **Fig. 6:** t-SNE embeddings. 5 subplots (one per encoder) × 6 colors (domains).

Tables:
- **Tab. 1:** Strengths/weaknesses (already in §1).
- **Tab. 2:** Hyperparameters (already in §6.2).
- **Tab. 3:** Main result — FPR × (encoder, ±L²).
- **Tab. 4:** Recall stratified by transformation × (encoder, ±L²).

- [ ] **Step 6: Write 6.6 — Results placeholder**

```latex
\subsection{Results}\label{sec:results}

% TODO: Fill after experiment runs (subproject C).
% Expected content: Tab. 3 (FPR), Tab. 4 (Recall by transformation),
% Fig. 4 (FPR bars), Fig. 5 (Recall bars), Fig. 6 (t-SNE),
% and 2-3 paragraphs of analysis.
```

- [ ] **Step 7: Compile check**

Run: `cd /Users/den1shh/Documents/2026-Project-198/paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex`
Expected: No errors.

- [ ] **Step 8: Commit**

```bash
git add paper/main.tex
git commit -m "feat: add Experiments section (W4.X, W5.C, W5.V, results placeholder)"
```

---

### Task 9: Conclusion

**Files:**
- Modify: `paper/main.tex` — replace §7 placeholder

- [ ] **Step 1: Write Conclusion**

Use `superpowers:scientific-writing` skill. Two paragraphs:

1. **Summary of contribution.** We presented a siamese plagiarism detector with five encoder configurations spanning three symmetry groups (D₄, ℤ², SE(2)). The framework isolates the effect of geometric inductive bias from data augmentation by using identical DINOv2 pre-training and comparing each encoder with and without L² contrastive regularization. We provided theoretical guarantees (fusion symmetry, invariance propagation, isometry invariance of L²).

2. **Planned next steps.** Complete the 10 experimental runs on DomainNet, report FPR and Recall, and assess whether built-in invariance eliminates the need for geometric augmentation. Extend evaluation to additional transformation combinations and domain shifts.

- [ ] **Step 2: Compile check**

Run: `cd /Users/den1shh/Documents/2026-Project-198/paper && pdflatex -interaction=nonstopmode main.tex`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add paper/main.tex
git commit -m "feat: add Conclusion"
```

---

### Task 10: Final Compilation and Cleanup

**Files:**
- Modify: `paper/main.tex` — minor fixes if needed
- Modify: `paper/references.bib` — remove unused entries

- [ ] **Step 1: Remove unused bib entries**

Compare `\cite{}` keys in `main.tex` against entries in `references.bib`. Remove entries not cited anywhere (e.g., `power2022grokking`, `nanda2023progress`, `gromov2023grokking`, `cheng2024spatialrgpt`, `he2016deep`, `tan2019efficientnet`, `radford2021learning`, `zbontar2021barlow` — unless they ended up cited in Related Work).

- [ ] **Step 2: Standardize citation keys**

Ensure all `\cite{}` calls in `main.tex` match the keys in `references.bib`. In particular:
- `nordstrom2025octicvisiontransformersquicker` → rename to `nordstrom2025octic` (shorter, consistent with spec)
- `rojasgomez2023makingvisiontransformerstruly` → rename to `rojasgomez2024shift`
- `karella2024harmformerharmonicnetworksmeet` → rename to `karella2024harmformer`
- `dorin2024pairwise` → keep or rename to `dorin2024imagematching`

Update all corresponding `\cite{}` in `main.tex`.

- [ ] **Step 3: Full compilation**

Run: `cd /Users/den1shh/Documents/2026-Project-198/paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex`
Expected: No errors, no unresolved citations, no undefined references.

- [ ] **Step 4: Verify abstract length**

Run: `cd /Users/den1shh/Documents/2026-Project-198/paper && sed -n '/\\begin{abstract}/,/\\end{abstract}/p' main.tex | grep -v '\\begin\|\\end' | wc -c`
Expected: ≤600.

- [ ] **Step 5: Verify notation consistency**

Grep for old notation that should not appear:
- `\mathbf{I}` → should be `x` (images)
- `\mathbf{z}` → should be `z` (embeddings)
- `g_\omega` → should be `\varphi_\theta` (encoder)
- `q_\phi` → should be `h_\psi` (fusion)
- `cosine` or `\cos` in loss context → should be L² only

Run: `grep -n 'mathbf{I}\|mathbf{z}\|g_\\omega\|q_\\phi\|\\\\cos(' paper/main.tex`
Expected: No matches (or only in intentional contexts like Schur's lemma).

- [ ] **Step 6: Commit**

```bash
git add paper/main.tex paper/references.bib
git commit -m "chore: clean up references, standardize citation keys, verify compilation"
```

---

## Post-Plan Checklist

- [ ] Every m1p.org deliverable (W1, W2.A, W2.L, W3.I, W3.P, W4.X, W5.C, W5.V, W6) is covered by at least one task
- [ ] §6.6 Results is a TODO placeholder — no fabricated numbers
- [ ] Notation is consistent throughout: $x$, $z$, $\varphi_\theta$, $h_\psi$, $f_\theta$, $G$, $\rho$
- [ ] SE(2) used for Harmformer (not SO(2))
- [ ] FPR + Recall as primary metrics everywhere
- [ ] 5 encoders × {±L²} = 10 runs
- [ ] No pseudocode blocks
- [ ] No mention of davnords/octic-vits weights
- [ ] L² contrastive only (no cosine in loss)
