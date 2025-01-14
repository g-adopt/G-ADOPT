cases := analytical_comparisons multi_material optimisation_checkpointing scalar_advection scalar_advection_diffusion Drucker-Prager_rheology adjoint 2d_cylindrical_TALA_DG viscoplastic_case_DG free_surface viscoelastic base_gmsh
long_cases := analytical_comparisons parallel_scaling

.PHONY: all longtest longtest_output $(cases) $(long_cases) clean check longcheck

all: $(cases)

longtest: $(long_cases)

longtest_output: $(long_cases)

# sort to remove duplicates which define both short and long tests
$(sort $(cases) $(long_cases)):
	$(MAKE) -C $@ $(MAKECMDGOALS)

clean: $(cases)
	rm -rf __pycache__

check:
	python -m pytest . ../demos -m 'not demo and not longtest'

longcheck:
	python -m pytest -m longtest
