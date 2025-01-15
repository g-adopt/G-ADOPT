.PHONY: lint test longtest longtest_output convert_demos

lint:
	@echo "Linting module code"
	@python3 -m ruff check gadopt
	@echo "Linting demos and tests"
	@python3 -m ruff check demos tests

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
