# run pytest watch command

pytest-watch:
	poetry run python -m ptw --onpass "say passed" --onfail "say failed"

watch:
	nodemon --exec poetry run python ./formula_finder/__init__.py