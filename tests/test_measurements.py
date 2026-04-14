import eradiate

import eradiate_disort as ed


class TestDisortMeasure:
    def test_construct(self, mode_mono):
        # Instantiation with default settings
        print(ed.DisortMeasure())

    def test_factory(self, mode_mono):
        # Instantiation with default settings
        print(eradiate.scenes.measure.measure_factory.create("disort"))
