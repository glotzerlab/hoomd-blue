import hoomd
from hoomd.md import _md
from hoomd.operation import Compute
from hoomd.logging import log

class FEPPair(Compute):
    pass

    @log(category="particle", requires_run=True)
    def energies(self):
        """(*N_particles*, ) `numpy.ndarray` of ``float``: Energy \
        contribution :math:`U_i` from each particle :math:`[\\mathrm{energy}]`.

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `energies` is `None` on ranks >= 1.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getEnergies()

def LJ(FEPPair):
    pass