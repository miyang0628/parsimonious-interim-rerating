# Manuscript source

- `main.tex` — final submitted LaTeX source (Springer Nature `sn-jnl` class, author-year).
- `references.bib` — bibliography.
- `main.pdf` — compiled PDF (for convenience).

To rebuild `main.pdf`, you also need the Springer Nature class files (`sn-jnl.cls`,
`sn-mathphys-ay.bst`) and the `figures/` directory. These ship in the separate journal-submission
bundle (`EmpiricalEcon_submission.zip`); copy them next to `main.tex`, then:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```
