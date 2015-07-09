# I'm unclear on how to setup the deps for TeX and Bib files, so just mark it as .PHONY for now:
.PHONY: heromsi.pdf
heromsi.pdf:
	pdflatex heromsi
	bibtex heromsi
	pdflatex heromsi
	pdflatex heromsi

