all:
	panzer ---quiet writeup.md -o writeup.tex

watch:
	panzer ---quiet writeup.md -o writeup.tex
	fswatch -o writeup.md | while read f; do panzer ---quiet writeup.md -o writeup.tex ; done