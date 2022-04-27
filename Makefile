
format:
	env/bin/isort HLAN tests
	env/bin/black HLAN tests

test:
	env/bin/pytest --disable-warnings -m "not slow" tests/

test-all:
	env/bin/pytest --disable-warnings tests/
