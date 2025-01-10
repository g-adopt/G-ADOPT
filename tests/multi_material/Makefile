cases := $(wildcard benchmarks/*)
archive := output_0_reference.npz
outputs := $(foreach case,$(cases),$(case)/${archive})
ncpus := 4

all: $(outputs)

%/${archive}:
	$(info running $(notdir $(realpath $(dir $@))) multi-material benchmark)
	tsp -N $(ncpus) -f mpiexec -np $(ncpus) python3 run_benchmark.py $(dir $@)

clean:
	rm -rf __pycache__
	@$(foreach case,$(cases),rm -rf $(case)/__pycache__;)
	@$(foreach case,$(cases),rm -rf $(case)/outputs;)
	@$(foreach case,$(cases),rm -f $(case)/*.pdf;)
	@$(foreach case,$(cases),rm -f $(case)/*.npz;)
	@$(foreach case,$(cases),rm -f $(case)/mesh/*.msh;)

check: $(cases)
	python3 -m pytest
