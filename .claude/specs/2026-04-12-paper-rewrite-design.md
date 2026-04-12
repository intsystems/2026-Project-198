# Design Spec: Rewrite of `paper/main.tex`

**Date:** 2026-04-12
**Scope:** Subproject A of 3 (A = paper rewrite, B = ISP RAS thesis, C = experiment code)
**Status:** Approved v5 — FPR+Recall primary; 5 encoders × {±L² contrastive}; P/G augmentation blocks; no pseudocode; Harmformer = SE(2) (roto-translations).

---

## 1. Цель

Полностью переписать `paper/main.tex` на английском языке так, чтобы:

1. текст соответствовал исследованию, описанному пользователем и в `thesis/tesis.ipynb`;
2. каждая секция закрывала соответствующие задания курса m1p.org за недели W1–W6 (кроме O* и R);
3. стиль был научным, читаемым, с минимумом воды;
4. правятся только `paper/main.tex` и `paper/references.bib`.

Результаты эксперимента в §6.6 — placeholder (TODO); реальные числа после подпроекта C.

---

## 2. Источники правды

| Файл | Роль |
|---|---|
| `thesis/tesis.ipynb` | Русский драфт Introduction; основа §1 (перевод + расширение до 8 элементов W3.I). |
| Первое сообщение пользователя | Описание эксперимента: энкодеры, метрики, L² contrastive, fusion. Основа §6. |
| `code/image-plagiarism-detection-with-eq-encoders.ipynb` | Рабочий пайплайн; источник гиперпараметров для §6.2. |
| `Image_Matching (1).pdf` (Dorin et al.) | Базовая сиамская архитектура и DomainNet-протокол. |
| `LINKREVIEW.md` | Текущий W2.L — расширяем на английском в §2. |
| `paper/neurips_2025.sty` | Шаблон — **не трогать**. |

---

## 3. Scope

**In scope:** `paper/main.tex`, `paper/references.bib`.
**Out of scope:** экспериментальные числа в §6.6, код в `code/`, русские тезисы, O*, R.

---

## 4. Структура статьи

```
Abstract                              ← W2.A
1. Introduction                       ← W3.I
2. Related Work                       ← W2.L
3. Problem Statement                  ← W3.P
4. Preliminaries: Equivariance        ← W6 (теория)
5. Method                             ← W6 (теория решения)
6. Experiments                        ← W4.X + W5.C + W5.V
7. Conclusion
```

---

## 5. Содержание по секциям

### Abstract (W2.A — ≤600 символов, 5 элементов)
1. Problem field: image plagiarism detection.
2. Narrow focus: inductive bias vs augmentation as routes to geometric robustness.
3. Method features: siamese network with equivariant ViT encoders (D₄, ℤ², SE(2)) + L² contrastive regularizer.
4. Novelty: head-to-head comparison of three symmetry groups in one siamese framework, evaluated with FPR and Recall on DomainNet.
5. Application: screening of plagiarism in journalism, academia, design.

### §1 Introduction (W3.I — 8 элементов + Chief Editor formula + Table 1)
1. **Goal** — evaluate whether built-in G-invariance improves plagiarism detection over augmentation.
2. **Object of study** — pairs of natural images related by D₄/ℤ²/SE(2) + photometric perturbations.
3. **Central challenge** — ViTs are not geometrically invariant by construction; augmentation only approximates robustness.
4. **State of the art** — Octic ViT (D₄), Shift-Equivariant ViT (ℤ²), Harmformer (SE(2)), DINOv2 pre-training, siamese detectors (Dorin et al.).
5. **Project tasks** — (i) implement five encoder configurations in one siamese framework; (ii) DINOv2 pre-training on ImageNet-1K; (iii) fine-tune each ± L² contrastive regularization; (iv) evaluate with FPR and Recall on DomainNet.
6. **Proposed solution / novelty / advantages (Chief-Editor formula).** *"A siamese plagiarism detector with G-equivariant ViT encoders, providing exact invariance under D₄, ℤ² or SE(2) without geometric augmentation, distinguished from Dorin et al. by an L² contrastive regularizer and a controlled ablation of geometric augmentation vs equivariant inductive bias."*
7. **Strengths and weaknesses of recent work — Table 1.** Rows: Dorin et al., Octic ViT, Shift-Equiv ViT, Harmformer. Columns: group coverage, exactness (exact/approximate), scope (supervised/SSL), weakness.
8. **Experimental goal and set-up** — 10 runs = 5 encoders × {±L² contrastive} on DomainNet; primary metrics FPR and Recall (stratified by transformation); secondary F1.

### §2 Related Work (W2.L)
Перенос `LINKREVIEW.md` на английский, тематические подсекции:
- Siamese detectors for image matching and plagiarism;
- Equivariant and invariant architectures (CNN → ViT);
- Self-supervised ViT pre-training (DINOv2);
- Contrastive objectives in metric learning.

### §3 Problem Statement (W3.P — явно пронумерованные 12 пунктов)

1. **General problem.** Binary decision: is $x_2$ a plagiarised derivative of $x_1$?
2. **Sample set.** $\mathcal{D}=\{(x_1^{(i)},x_2^{(i)},y^{(i)})\}_{i=1}^N$, $y\in\{0,1\}$.
3. **Statistical hypothesis.** Positive pairs drawn from $p_+(x_1,x_2)=\int p(x)\,p_T(x_2\mid x_1)\,dx$, $p_T$ a mixture over admissible transformations in $G\cup G_\text{photo}$.
4. **Measurement conditions.** DomainNet, six domains, $224{\times}224$ 8-bit RGB.
5. **Sample restrictions.** Positive — source + its $(g,\pi)$-transformed copy; negative — distinct source images from the same domain.
6. **Model.** $f_\theta(x_1,x_2)=\sigma\!\left(h_\psi(\varphi_\theta(x_1),\varphi_\theta(x_2))\right)$, $\varphi_\theta$ a G-equivariant ViT, $h_\psi$ the fusion head.
7. **Class restrictions.** $\varphi_\theta$ G-equivariant with G-invariant pooling; $h_\psi$ symmetric.
8. **Loss / criterion.** $\mathcal{L}=\text{BCE}(f_\theta(x_1,x_2),y)+\lambda\,\mathcal{L}_\text{contr}^{L^2}$ (λ=0 в варианте без регуляризации).
9. **Cross-validation procedure.** Stratified split by DomainNet domain, five-fold CV.
10. **Solution restrictions.** Single GPU ≤48 GB VRAM; ≤30 fine-tuning epochs.
11. **External quality criteria.** **Primary: FPR and Recall** (Recall stratified by transformation class). Secondary: F1, t-SNE separability.
12. **Argmin.** $\hat\theta=\arg\min_\theta\,\mathbb{E}_{(x_1,x_2,y)\sim\mathcal{D}}\,\mathcal{L}(\theta)$.

### §4 Preliminaries: Equivariance (W6 — theory)

- **4.1 Group actions & representations.** Определение $G$-действия на $\mathcal{X}$; определение представления $\rho$.
- **4.2 Equivariance and invariance.** $\varphi(\rho_X(g)\cdot x)=\rho_Y(g)\cdot\varphi(x)$; инвариантность как частный случай ($\rho_Y\equiv\text{id}$).
- **4.3 Irreducible representations & Schur's lemma.** Мотивирует блочно-диагональную структуру Octic ViT.
- **4.4 Groups considered:**
  - $D_4$ — конечная, $|D_4|=8$, irreps $\{1,1',1'',1''',2\}$;
  - $\mathbb{Z}^2$ — счётная, действует циклическим сдвигом на patch-сетке;
  - $SE(2)$ — непрерывная группа рототрансляций плоскости ($SO(2) \ltimes \mathbb{R}^2$).

### §5 Method (W6 — theory of the solution)

- **5.1 Siamese framework.** Две ветви с общими весами, fusion, сигмоида.
- **5.2 Encoders.** Пять конфигураций:
  - **ViT-L/16 (augmented)** — baseline с геометрическими аугментациями (блок G + блок P);
  - **ViT-L/16 (baseline)** — только фотометрические аугментации (блок P);
  - **Octic ViT** — $D_4$-эквивариантный, без геометрических аугментаций (блок P);
  - **Shift-Equivariant ViT** — $\mathbb{Z}^2$-эквивариантный, без геом. аугментаций (блок P);
  - **Harmformer** — $SE(2)$-эквивариантный, без геом. аугментаций (блок P).
- **5.3 Fusion module.** $h_\psi(z_1,z_2)=W\,\text{ReLU}\!\left[\,|z_1-z_2|\,\Vert\,z_1\odot z_2\,\right]$. Симметричен по построению.
- **5.4 L² contrastive regularizer.** $\mathcal{L}_\text{contr}^{L^2}=y\,\|z_1-z_2\|_2^2+(1-y)\,\max(0,m-\|z_1-z_2\|_2)^2$. Для каждого энкодера сравниваются версии с и без ($\lambda=0$).
- **5.5 Augmentation blocks.**
  - **Block P (photometric)** — все энкодеры: ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05), RandomGrayscale(p=0.05), GaussianBlur(kernel=3, p=0.1), ImageNet normalization.
  - **Block G (geometric)** — только ViT (augmented): RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5), RandomChoice{0°, 90°, 180°, 270°}, RandomResizedCrop(scale=0.7–1.0, ratio=0.9–1.1).
- **5.6 Theoretical properties.**
  - **Prop. 1 (Symmetry of fusion).** $h_\psi(z_1,z_2)=h_\psi(z_2,z_1)$.
  - **Prop. 2 (Invariance propagation).** Если $\varphi_\theta$ $G$-инвариантен, то $f_\theta(g\cdot x_1,g\cdot x_2)=f_\theta(x_1,x_2)$.
  - **Prop. 3 (Isometry invariance of L² contrastive).** $\mathcal{L}_\text{contr}^{L^2}$ инвариантен относительно изометрий пространства эмбеддингов.

Каждое утверждение с коротким доказательством (3–5 строк).

### §6 Experiments

- **6.1 Dataset and pre-training (W4.X).**
  - **Pre-training:** ImageNet-1K, DINOv2 self-supervised.
  - **Fine-tuning / evaluation:** DomainNet (Peng et al., 2019), 6 доменов (real, painting, clipart, quickdraw, infograph, sketch). Object count: 1200 пар (100 pos + 100 neg × 6 доменов). Feature count per image: $3{\times}224{\times}224=150{,}528$.
- **6.2 Training pipeline.**
  - 10 runs = 5 encoders × {без L² contrastive (λ=0), с L² contrastive (λ>0)}.
  - Гиперпараметры из ноутбука: AdamW (lr $10^{-3}$, wd $10^{-2}$), MultiStepLR (milestones $[5,10]$, $\gamma=0.5$), BCE weights $w_+=0.3$, $w_-=0.7$, 15 эпох, Dorin-style cross-batch pairing.
  - Описание пайплайна текстом (без псевдокода): загрузка данных → формирование пар → аугментация (P для всех, P+G для ViT augmented) → forward через сиамскую сеть → BCE + λ·L²-contrastive → AdamW step.
- **6.3 Evaluation protocol (W4.X).**
  - Workflow: для каждого encoder × {±L²} — обучение → выбор порога на валидации → метрики на тест-сете.
  - Сбалансированный тест-сет 1200 пар на DomainNet.
  - **Primary metrics: FPR and Recall** (Recall стратифицирован по классам трансформаций: CJ, GN, GS, R90/180/270, WM, combos).
  - **Secondary:** F1; t-SNE.
- **6.4 Code organization (W5.C).**
  - `code/main.py` — единая точка входа;
  - модули: `encoders/`, `siamese/`, `data/`, `train/`, `eval/`;
  - центральный `config.yaml`;
  - ноутбук — текущий entry-point baseline.
- **6.5 Planned figures and tables (W5.V).**
  - **Figures** (.pdf / .eps, 300 DPI, 12–16 pt, 2 pt lines):
    - Fig. 1: siamese architecture schema.
    - Fig. 2: diagrams of $D_4$, $\mathbb{Z}^2$, $SE(2)$ с примерами действий.
    - Fig. 3 (source data): распределение DomainNet — x: domain, y: image count.
    - Fig. 4 (main message): FPR × (encoder, ±L²) — x: encoder, y: FPR, grouped bars {без L², с L²}.
    - Fig. 5 (main message): Recall by transformation — x: transformation, y: Recall, bars per encoder.
    - Fig. 6: t-SNE — 5 subplots (encoders) × 6 цветов (domains).
  - **Tables:**
    - Tab. 1: strengths/weaknesses recent work (из §1).
    - Tab. 2: hyperparameters.
    - Tab. 3: main result — FPR × (encoder, ±L²).
    - Tab. 4: Recall stratified by transformation × (encoder, ±L²).
- **6.6 Results.** **Placeholder с TODO.** Числа после подпроекта C.

### §7 Conclusion
Резюме вклада + запланированные шаги.

---

## 6. Обновления `paper/references.bib`

Добавить/проверить: `dosovitskiy2021image`, `chen2024spatialvlm`, `cohen2016group`, `bronstein2021geometric`, `bromley1993signature`, `nordstrom2025octic`, `rojasgomez2024shift`, `karella2024harmformer`, `oquab2024dinov2`, `dorin2024imagematching`, `chopra2005learning`, `hadsell2006dimensionality`, `peng2019domainnet`.

Компиляция: `pdflatex → bibtex → pdflatex → pdflatex`.

---

## 7. Writing style rules

- Scientific, terse, no filler.
- Активный залог; настоящее время для метода.
- Обозначения: $x$ — изображение, $z$ — эмбеддинг, $\varphi_\theta$ — энкодер, $h_\psi$ — fusion, $f_\theta$ — модель, $G$ — группа, $\rho$ — представление.
- Аббревиатуры раскрываются при первом упоминании.

---

## 8. Skills / plugins / MCPs

- **superpowers:scientific-writing** — посекционная перезапись.
- **superpowers:writing-plans** — implementation plan после утверждения spec'а.
- **Context7 MCP** — документация пакетов NeurIPS при необходимости.
- **Read/Edit/Write** — `paper/main.tex`, `paper/references.bib`.

---

## 9. Non-goals

- Не переструктурируем репозиторий.
- Не запускаем эксперименты, не обновляем §6.6.
- Не пишем `code/` — это подпроект C.
- Не включаем O* и R.

---

## 10. Success criteria — per-section compliance vs m1p.org

| # | Задание | Секция | Элементы рекомендаций | Покрыто? |
|---|---|---|---|---|
| W1 | LaTeX / BibTeX / GitHub / env | вся статья + bib | компилируется; bib корректен; git | ✅ |
| W2.A | Abstract ≤600 символов | Abstract | 5 элементов | ✅ |
| W2.L | Literature review | §2 | LINKREVIEW → English, ≥9 ключевых работ, тематическая группировка | ✅ |
| W3.I | Introduction 8 элементов | §1 | 8 элементов + Chief Editor formula + Table 1 | ✅ |
| W3.P | Problem Statement 12 элементов | §3 | 12 пронумерованных пунктов | ✅ |
| W4.X | Experiment plan | §6.1, §6.3, §6.5 | dataset, object/feature count, workflow, figure/table lists с осями | ✅ |
| W5.C | Code organization | §6.4 | single main, modules, centralized params | ✅ |
| W5.V | Visualizations | §6.5 | source data plot, main message plot, formats, font/line specs | ✅ |
| W6 | Theory | §4 + §5.6 | определения + 3 propositions с доказательствами | ✅ |

**Примечание:** W4.B (pseudocode) исключён по запросу пользователя. Пайплайн обучения описан текстом в §6.2.

Финальный критерий: статья компилируется без ошибок; §6.6 — TODO.

---

## 11. Open questions

1. **Margin $m$ в L² contrastive.** Предварительно $m=1.0$; окончательный подбор в подпроекте C.
2. **DomainNet sub-sampling.** Default: 6 доменов целиком, 100 pos + 100 neg на домен.
3. **λ для L² contrastive.** Предварительно $\lambda=0.1$; финальный подбор в C.

---

## 12. Flow после утверждения

1. Пользователь подтверждает / правит spec.
2. **superpowers:writing-plans** — пошаговый implementation plan.
3. Посекционно переписываю `paper/main.tex` через **superpowers:scientific-writing**; обновляю `references.bib`; проверяю компиляцию.
4. Подпроект B (ISP RAS), затем C (эксперимент).
