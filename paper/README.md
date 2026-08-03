# Manuscript package

`manuscript.tex` is the formula-driven manuscript prepared for Knowledge-Based Systems. Its title introduces Feature-Synthesis Variance Amplification (FSVA) as the paper’s mechanistic contribution, and the theory section gives FSVA a compact defining equation. It uses the 22-dataset primary scope, contains a sub-250-word abstract, defines task-level and data-set-balanced estimands, derives error-propagation and bias--variance conditions, reports condition-wise results, and links only to the self-contained PDF figures in `figures/`. The workflow artwork is an internal evaluation-protocol diagram; coverage exceptions remain documented in the manuscript rather than being mixed into the method graphic.

Files:

- `manuscript.tex` — editable Elsevier `elsarticle` source.
- `references.bib` — bibliography for the methods and software cited in the manuscript.
- `highlights.txt` — five short submission highlights.
- `cover_letter.txt` — submission-ready cover letter naming Md. Shoaib Uddin Chanda as corresponding author.
- `figures/` — self-contained PDF copies of the workflow plus regenerated Figures 2–10. The manuscript uses only these PDF assets and does not depend on paths outside `paper/`.

The source code and manuscript package are maintained at <https://github.com/mdshoaibuddinchanda/AutoFE-ShiftBench>. The local environment used for this audit does not contain `pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `tectonic`, so the source has not been compiled here. Compile from this directory with an Elsevier-compatible LaTeX installation, for example:

```text
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Generated result and report artifacts are intentionally excluded from version control; the local result and table archives are preserved separately from this paper package.
