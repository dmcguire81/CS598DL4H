

format:
	env/bin/isort HLAN tests
	env/bin/black HLAN tests

test:
	env/bin/pytest --disable-warnings tests/
