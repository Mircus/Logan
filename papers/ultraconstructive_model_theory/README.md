# Ultraconstructive Model Theory (UCMT) — paper source

*Ultraconstructive Model Theory via Bounded Adversarial Finite Structures.*

## Files
- `main.tex` — the paper source.
- `refs.bib` — bibliography.

## Build

```bash
latexmk -pdf main.tex
```

or, without `latexmk`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

This produces `main.pdf`. The PDF is not committed; build it locally (or on
Overleaf) from `main.tex` + `refs.bib`.
