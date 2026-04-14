import attrs
from eradiate.scenes.measure import Measure, measure_factory


@attrs.define(eq=False, slots=False)
class DisortMeasure(Measure):
    """
    DISORT measurement [``disort``]

    A flexible measurement definition for the DISORT backend.
    """

    @property
    def film_resolution(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    def kernel_type(self) -> str:
        raise NotImplementedError

    @property
    def sensor_id(self) -> str:
        raise NotImplementedError

    @property
    def template(self) -> dict:
        raise NotImplementedError


measure_factory.register(DisortMeasure, type_id="disort")
