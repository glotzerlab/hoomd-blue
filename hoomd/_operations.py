import hoomd._integrator


class Operations:
    def __init__(self, simulation=None):
        self.simulation = None
        self._auto_schedule = False
        self._compute = list()

    def add(self, op):
        if isinstance(op, hoomd.integrate._integrator):
            self._set_integrator(op)
        else:
            raise ValueError("Operation is not of the correct type to add to"
                             " Operations.")
        # if self._auto_schedule:
        #     op.initialize()

    def _set_integrator(self, integrator):
        self._integrator = integrator

    @property
    def _sys_init(self):
        if self.simulation is None or self.simulation.state is None:
            return False
        else:
            return True

    def schedule(self):
        if not self._sys_init:
            raise RuntimeError("System not initialized yet")
        if hasattr(self, '_integrator'):
            sys_def = self.simulation.state._cpp_sys_def
            new_objs = self._integrator._attach(sys_def)
            self.simulation._cpp_sys.setIntegrator(
                    self._integrator.cpp_integrator)
            if new_objs is not None:
                self._compute.extend(new_objs)
