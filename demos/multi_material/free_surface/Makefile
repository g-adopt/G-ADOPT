all: params.log

demo_type := $(notdir $(realpath $(dir $(CURDIR))))
demo_name := $(notdir $(CURDIR))
ncpus := 4

params.log: $(demo_name).py
	$(info running $<)
	/usr/bin/time --format="$< finished in %E" tsp -L multi_material -N $(ncpus) -f mpiexec -np $(ncpus) python3 $<

clean:
	rm -f *.pvd *.vtu *.pvtu *.h5 params.log
	rm -rf output

check: params.log
	python3 -m pytest ../../test_all.py -k $(demo_type)/$(demo_name)
