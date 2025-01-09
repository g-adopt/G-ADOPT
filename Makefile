.PHONY: lint test longtest longtest_output convert_demos

lint:
	@echo "Linting module code"
	@if pip show -qq ruff; then python3 -m ruff check gadopt; else python3 -m flake8 gadopt; fi
	@echo "Linting demos and tests"
	@if pip show -qq ruff; then python3 -m ruff check demos tests; else python3 -m flake8 demos tests; fi

test:
	$(MAKE) -C demos & $(MAKE) -C tests & wait

longtest:
	$(MAKE) -C tests longtest

longtest_output:
	$(MAKE) -C tests longtest_output

# convert demo Python scripts to executed notebooks
convert_demos:
	$(MAKE) -C demos convert_demos

clean:
	$(MAKE) -C demos clean & $(MAKE) -C tests clean & wait

check:
	python -m pytest -m 'not longtest'
