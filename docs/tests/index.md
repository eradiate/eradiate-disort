# Validation tests

These notebooks compare the DISORT backend output against Mitsuba for a
series of scenes of increasing complexity.  Each notebook is authored as a
percent-format Python script under `tests/notebooks/` and paired here as an
executed `.ipynb` file via jupytext.

To regenerate the notebook outputs locally:

```bash
pixi run nb-execute
```

```{toctree}
:maxdepth: 1

test_01_noatmo
test_02_single_layer
test_03_two_layers
test_04_molecular_atmosphere
test_05_aerosols
test_06_full_atmo
```
