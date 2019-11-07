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

    def _set_integrator(self, integrator):
        self._integrator = integrator

    @property
    def _operations(self):
        op = list()
        if hasattr(self, '_integrator'):
            op.append(self._integrator)
        return op

    @property
    def _sys_init(self):
        if self.simulation is None or self.simulation.state is None:
            return False
        else:
            return True

    def schedule(self):
        if not self._sys_init:
            raise RuntimeError("System not initialized yet")
        sim = self.simulation
        for op in self._operations:
            new_objs = op.attach(sim)
            if isinstance(op, hoomd.integrate._integrator):
                sim._cpp_sys.setIntegrator(op._cpp_obj)
            if new_objs is not None:
                self._compute.extend(new_objs)

    def _store_reader(self, reader):
        # TODO
        pass
