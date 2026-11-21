#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch causality main.tex: append Amendment 7 after Amendment 6."""
from pathlib import Path

p = Path(r"C:\Users\Lance\WorkBuddy\2026-10-25-21-15-17\github-repo\paper\causality\main.tex")
src = p.read_text(encoding="utf-8")

anchor = r"""\S\ref{sec:e4} are unchanged; the
caliber note has been added to the abstract, the limitation list,
and the App.~29 discussion."""
assert anchor in src, "Amendment 6 tail anchor missing"

block = r"""

\paragraph{Amendment 7 (label-scope extensions: smoothness, compression,
criticality, and hierarchical feedback).}
Four preregistered follow-up programs (ADR-0022--0028, all frozen
before their runs) extend the label-not-knob scope along independent
axes; all four close negative and none touches a headline claim.
(1) \emph{Smoothness} (ADR-0025): three frozen manifold-smoothness
metrics on the same 68-checkpoint corpus are all weaker markers than
$\dfx$ ($|r| = 0.51$--$0.77$ vs $0.89$) and carry mostly its
information; on the forcing dose axis they co-move with the
intervention but decouple from behavior (ID accuracy flat), the
same label signature.
(2) \emph{Compression} (ADR-0026): spectral and non-spectral
code-length proxies alike collapse into the dimension family
($r(L_{\mathrm{knn}}, \dfx) = +0.92$; partial $+0.66$) --- compression
as an independent generalization axis fails at this scale.
(3) \emph{Criticality} (ADR-0027): the capacity--class ratio
$\Phi = \Psi\bar D/P$ spans $0.002$--$0.035$ across the corpus,
two orders of magnitude below the critical band --- representations
are deeply subcritical, and the one quantity that survives
controlling for $\dfx$ (the GLUE utility, partial $r = -0.61$)
reads \emph{under-utilized representations generalize better},
consistent with that picture.
(4) \emph{Hierarchical feedback} (ADR-0022--0024): top-down
feedback---modulated, bottlenecked, or belief-gated---yields zero
or negative gain in four independent implementations including a
preregistered gate rejection (A0: the belief state's variance
responds \emph{inverted} to input occlusion, $\rho = -1.0$, and
the module is net-harmful, $-9$ to $-23$\,pp), and hierarchical
prompt structure carries no benefit over flat lists on the LLM
side (three saturated batches, descriptive-only). Each program
is archived in full with its frozen criteria; together they delimit
the scope of the label claim without modifying it: every additional
elegance-family quantity we tested behaves as a response variable
of the recipe/forcing, not as an independent causal axis.
"""

src = src.replace(anchor, anchor + block, 1)
p.write_text(src, encoding="utf-8")
print("Amendment 7 appended; new length:", len(src))
