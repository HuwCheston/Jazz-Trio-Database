import numpy as np


class SimulationManager:
    """
    Examples:
        >>> piano = {
        >>>     'self_coupling': float | list[float],
        >>>     'coupling_piano': np.nan | list[np.nan],
        >>>     'coupling_bass': float | list[float],
        >>>     'coupling_drums': float | list[float],
        >>>     'intercept': float,
        >>>     'resid_std': float
        >>> }
        >>> bass = {
        >>>     'self_coupling': ...,
        >>>     ...
        >>> }
        >>> drums = {...}

    """
    def __init__(
            self,
            piano_coupling: dict,
            bass_coupling: dict,
            drums_coupling: dict,
            tempo: int = 120
    ):
        # Do a quick check to ensure that input dictionaries all look correct
        self._verify_input(piano_coupling, bass_coupling, drums_coupling)
        # Unpack coupling coefficient dictionaries into dictionary
        self.coupling = {
            'piano': piano_coupling,
            'bass': bass_coupling,
            'drums': drums_coupling
        }
        self.tempo = tempo

    @staticmethod
    def _verify_input(*args: dict) -> None:
        """Checks on input dictionaries to ensure correct formatting

        Args:
            *args (dict): arbitrary number of input dictionaries

        Raises:
            AssertionError: when an incorrectly-formatted dictionary is passed
        """
        # The correct structure for an input dictionary
        dict_struct = {
            'self_coupling': float | list[float],
            'coupling_piano': np.nan | list[np.nan],
            'coupling_bass': float | list[float],
            'coupling_drums': float | list[float],
            'intercept': float,
            'resid_std': float
        }
        # Do we have the correct keys in all of our dictionaries?
        assert all([dic.keys() == dict_struct.keys() for dic in args])
        # If any of our dictionary values are lists, do they all have the same length?
        assert len(set(len(v) for dic in args for v in dic.values() if isinstance(v, list))) == 1