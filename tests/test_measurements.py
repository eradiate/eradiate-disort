import eradiate

import eradiate_disort as ed


class TestDisortRadianceMeasure:
    def test_construct(self, mode_mono):
        # Instantiation with default settings
        print(ed.DisortRadianceMeasure())

    def test_factory(self, mode_mono):
        # Instantiation with default settings
        print(eradiate.scenes.measure.measure_factory.create("disoradiance"))


class TestDisortIrradianceMeasure:
    def test_construct(self, mode_mono):
        # Instantiation with default settings
        print(ed.DisortIrradianceMeasure())

    def test_factory(self, mode_mono):
        # Instantiation with default settings
        print(eradiate.scenes.measure.measure_factory.create("disoflux"))
