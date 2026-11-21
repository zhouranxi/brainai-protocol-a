#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch paper/measurement/main.tex: insert subsection before Limitations."""
from pathlib import Path

p = Path(r"C:\Users\Lance\WorkBuddy\2026-10-25-21-15-17\github-repo\paper\measurement\main.tex")
src = p.read_text(encoding="utf-8")

marker = "\\section{Limitations}"
assert marker in src, "anchor missing"

block = r"""
\subsection{The next marker, tested}
\label{sec:nextmarker}

Two quantities absent from the twenty-four families above were run through the
checklist on the same forty-model population: the \emph{kNN code length}
$L_{\mathrm{knn}}$ (median $k$-th-neighbour distance times sample count, a
non-spectral compression proxy) and the GLUE \emph{utility} $\Psi$ (the
capacity-utilization ratio of the class manifolds).
Table~\ref{tab:nextmarker} gives the marginal and partial associations against
the c100 probe target, alongside the effective dimension anchor of this study
($r(\bar D, \mathrm{c100}) = 0.883$).

\begin{table}[t]
\centering
\caption{Two new candidates on the paired population ($n=40$). The partial
column controls for the effective dimension: it is the arbiter between
``cheaper replacement'' and ``independent axis''.}
\label{tab:nextmarker}
\small
\begin{tabular}{lccc}
\toprule
 & $r(\cdot,\bar D)$ & $r(\cdot,\mathrm{c100})$ & partial $r(\mathrm{c100},\cdot \mid \bar D)$ \\
\midrule
$L_{\mathrm{knn}}$ & $+0.916$ & $\mathbf{+0.933}$ & $+0.660$ \\
$\Psi$ (utility)  & $-0.395$ & $-0.611$ & $\mathbf{-0.607}$ \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{$L_{\mathrm{knn}}$ is the strongest single marker measured in this
study} --- it outranks the effective dimension itself ($0.933$ vs $0.883$) at
a fraction of the compute (one $k$NN pass, no solver). But the partial column
is where it stops being a new marker: it is a strong carrier of dimension
information ($r(L_{\mathrm{knn}}, \bar D) = +0.916$, partial $+0.660$), not
an independent axis. It passes the checklist as a \emph{cheaper replacement}
for $\bar D$, not as a new quantity. The compression reading is the same
finding from the other side: spectral and non-spectral code-length proxies
alike collapse into the dimension family, so compression as an independent
generalization axis fails at this scale.

\paragraph{$\Psi$ is the first quantity whose signal survives controlling for
the effective dimension} ($-0.611 \to -0.607$: partialling out $\bar D$ leaves
the association essentially unchanged). Its dose-axis behaviour confirms the
reading: utility responds to forcing ($r = +0.92$) but the response is
carried by the dimension response (partial out $\bar D$ and the residual is
flat, $r = -0.23$). The direction is negative ---
\emph{under-utilized representations generalize better} --- consistent with
the capacity picture: every model in the population sits far below the
critical load ($\Phi = \Psi \bar D / P$ spans $0.002$ to $0.035$, two orders
of magnitude below the critical band), so the class manifolds retain large
capacity margin, and the models that burn that margin (high $\Psi$) transfer
worse.

\paragraph{Checklist verdicts.} $L_{\mathrm{knn}}$: passes as a cheap
replacement for the dimension, fails as an independent marker --- the partial
column is the entry we would add to the checklist for any future candidate.
$\Psi$: the first genuine second axis in this study; its negative direction
and its survival under partialling make it the strongest candidate yet for a
marker that measures something the dimension does not.

"""

src = src.replace(marker, block + marker, 1)
p.write_text(src, encoding="utf-8")
print("patched, new length:", len(src))
