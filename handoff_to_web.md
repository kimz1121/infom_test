# Handoff — VLA Task Self-Awareness Proposal

**To:** the Claude instance continuing this work on the web.
**Purpose:** finish a research **proposal** for an English academic-writing course
(weekly chapters → final combined submission). This note tells you the *current,
agreed direction* and what is still open. Where the older draft and this note
disagree, **this note wins** — parts of the existing draft are 3–4 weeks old and
predate the experiments that now define the direction.

---

## 1. What this proposal is about

**Title (working):** *Task Self-Awareness for Vision-Language-Action Models: An
Explicit, Verifiable Task-Embedding Approach.*

**One-sentence thesis:** current VLAs fail because task identity is never made an
explicit, separable, verifiable representation — it is either diluted by
linguistic ambiguity (hierarchical planners) or scattered across a reasoning
trace (CoT-VLA) — so we propose a dedicated module that supplies task identity to
the action expert as an explicit, learned embedding, and we *verify* that the
embedding genuinely discriminates tasks.

**Three contributions / three experiments (keep this spine):**
1. **Quantify task uncertainty** in SOTA VLAs (the LIBERO-Plus bottleneck, made
   into a measured number).
2. **Show policy-performance gains** from supplying an explicit task embedding,
   vs. recent VLA baselines, on long-horizon / compositional tasks.
3. **Verify the task-discriminative quality** of the learned embedding directly
   (independently of end-task success).

---

## 2. Files you are working with

- **`proposal_single.tex`** — the single-file proposal (abstract, intro, related
  work, method *placeholder*, experiments, embedded bibliography). Compiles with
  pdflatex. The **Method section is an empty placeholder** — its body was lost and
  must be written. The **Abstract still commits to a convex-cone / set-membership
  formulation** — that framing is now **deprecated** (see §3).
- **`findings.tex`** — a *Research Findings* chapter I just wrote (hybrid:
  conceptual findings from the literature, then a preliminary empirical study
  grounded in real experiments). It is written to be pasted into the proposal
  before "Proposed Experiments," with two new bibitems to merge. **Read this
  first** — it carries the current direction and the real numbers.

---

## 3. The agreed direction (read carefully — this is the ask)

**Keep:** the thesis, the three-contribution spine, the framing of existing
remedies as routing task identity through *lossy channels* (language /
reasoning trace), and "explicit + verifiable" as the differentiator.

**Drop:** the **convex-cone / set-membership geometry**. It was a secondary idea
and is now out of scope. Do **not** build the method around a hand-specified
embedding geometry or an "abstain on out-of-set inputs" mechanism. The abstract's
convex-cone paragraph should be rewritten to match the rest.

**Adopt instead — the method core to be written:** learn the task representation
with **representation-alignment / contrastive objectives in the successor-feature
spirit**, conditioned **multimodally** (vision + proprioception + language). This
is grounded in:
- *Temporal Representation Alignment* (successor features → emergent
  compositionality in instruction following),
- contrastive task-discriminative embeddings (RS-CL),
- and our own preliminary instance: an **intention-conditioned occupancy
  embedding** `z = q(z | s, a)` — a temporally-grounded "what task am I doing"
  vector that summarizes the discounted future trajectory.

So the embedding is **not** a fixed geometric set; it is a *learned, temporally
grounded representation*, and the proposal's distinctive move is that we can
**measure** its quality (see §4, R2).

---

## 4. The empirical grounding (so you have the facts without external memory)

These are real preliminary results (Robocasa-65 atomic tasks + LIBERO-goal-10;
multimodal = frozen ResNet-34 image feature ⊕ proprio state, optional sentence
embedding of the instruction). Use them; they're already written up in
`findings.tex`. Summary:

- **R1 — Attainable & discriminative.** Explicit embedding separates tasks far
  above chance: Robocasa-65 kNN task acc ~43% (chance ~1.5%), Fisher 0.19–0.21;
  LIBERO-10 Fisher 0.343 with clean per-instruction clusters.
- **R2 — Verifiable independent of policy success (the key methodological win).**
  A label-free diagnostic correlates embedding distance with the geometric DTW
  distance of the underlying trajectories. Good models: within-task ρ ≈ 0.5–0.6;
  a bad ablation (z forced to also reconstruct image features) collapses to
  ρ ≈ 0.18 with negative action correlation. ⇒ **Experiment 3 already has a
  working verification method.**
- **R3 — Language resolves exactly the ambiguous cases.** Adding the instruction
  raises per-demo kNN task-agreement on all 10 LIBERO tasks (mean 0.857→0.898),
  biggest gains on the confusable bowl-placement family (same pick motion,
  different destination: ~0.65–0.74 → toward ~0.93–1.0). It helps *task identity*,
  not trajectory fit — i.e., it attacks the ambiguity bottleneck directly.
- **R4 — Fidelity is structured.** Constrained motions embed well (articulated
  open/close/slide ρ≈0.53, knob ρ≈0.52); free-space embeds poorly (pick-place
  ρ≈0.39, navigate ρ≈0.25). ⇒ the hard, compositional, free-space regime is where
  the module must prove itself — sharpens task selection for Experiment 2.
- **R5 — Composition leaves a trace.** Composite-task demos traverse the atomic
  clusters (qualitatively right, low confidence ~0.21 for 65-way + frozen
  backbone) → motivates moving frozen→end-to-end visual encoder next.

Caveats to keep honest: separability is *flat across training* with frozen
features (limited headroom), and composite confidence is low — both are real and
are presented as motivation, not hidden.

---

## 5. Concrete TODOs to finish the proposal

1. **Rewrite the Abstract** to drop convex-cone; reframe around an *explicit,
   learned, verifiable* task representation (alignment/successor-feature spirit,
   multimodal conditioning).
2. **Write the Method section** (`proposal_single.tex` §Proposed Approach is a
   placeholder). It should specify: (a) module architecture — how the task
   embedding is produced and where it is injected into the action expert; (b) the
   learning objective — representation-alignment / contrastive, temporally
   grounded (cite TRA, RS-CL, InFOM); (c) how it stays lightweight vs. hierarchical
   / CoT alternatives. Do **not** reintroduce convex-cone.
3. **Merge `findings.tex`** in before "Proposed Experiments," and move its two new
   bibitems into the bibliography.
4. **Fill the two bibitems** — `tra` (*Temporal Representation Alignment…*) and
   `infom` (*Intention-Conditioned Flow Occupancy Models*). They are left as
   commented TODOs because the exact authors/arXiv ids were not verified — **look
   these up and fill them, do not invent ids.**
5. **Consistency pass:** make Intro/Related Work language about "embedding
   geometry left open" agree with the now-committed alignment-based direction; the
   three experiments already match — keep them.
6. Fill the author block (currently `Author Names`, `TODO`).

---

## 6. Tone / constraints

- English academic-writing course deliverable: clear, hedged where appropriate,
  no overclaiming (the inFOM results are *preliminary* and on Robocasa/LIBERO, not
  a full VLA integration — say so).
- The inFOM embedding is presented as *one concrete instance* of the explicit task
  representation, evidence the general approach is feasible — not as the final
  proposed system. Keep that distinction.
