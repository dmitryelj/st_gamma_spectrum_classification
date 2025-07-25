""" Data models """


from dataclasses import dataclass
from radiacode import Spectrum


@dataclass
class SpectrumData:
    spectrum: Spectrum

    def get_duration(self) -> float:
        """ Get spectrum duration """
        return self.spectrum.duration.total_seconds()
    
    def get_data(self) -> Spectrum:
        """ Get spectrum data """
        return self.spectrum
