# I'm unclear on how to setup the deps for TeX and Bib files, so just mark it as .PHONY for now:
all: pasc2016.tex 
	pdflatex pasc2016.tex
	bibtex pasc2016
	pdflatex pasc2016.tex
	pdflatex pasc2016.tex

clean:
	rm -f *.log *.bbl *.blg *.aux *.out

