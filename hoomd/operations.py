import hoomd.integrate


class Operations:
    def __init__(self, simulation=None):
        self.simulation = simulation
        self._compute = list()
        self._auto_schedule = False
        self._scheduled = False

    def add(self, op):
        if isinstance(op, hoomd.integrate._integrator):
            self._integrator = op
        else:
            raise ValueError("Operation is not of the correct type to add to"
                             " Operations.")

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
        self._scheduled = True

    def _store_reader(self, reader):
        # TODO
        pass

    @property
    def scheduled(self):
        return self._scheduled

    @property
    def integrator(self):
        try:
            return self._integrator
        except AttributeError:
            return None
