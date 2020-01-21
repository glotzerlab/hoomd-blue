import hoomd.integrate
import hoomd.meta


class Operations:
    def __init__(self, simulation=None):
        self.simulation = simulation
        self._compute = list()
        self._auto_schedule = False
        self._scheduled = False
        self._updaters = list()
        self._analyzers = list()

    def add(self, op):
        if op in self:
            return None
        if isinstance(op, hoomd.integrate._BaseIntegrator):
            self.integrator = op
            return None
        elif isinstance(op, hoomd.meta._Updater):
            self._updaters.append(op)
        elif isinstance(op, hoomd.meta._Analyzer):
            self._analyzers.append(op)
        else:
            raise ValueError("Operation is not of the correct type to add to"
                             " Operations.")
        if self._auto_schedule:
            self._schedule([op])

    @property
    def _operations(self):
        op = list()
        if hasattr(self, '_integrator'):
            op.append(self._integrator)
        op.extend(self._updaters)
        op.extend(self._analyzers)
        return op

    @property
    def _sys_init(self):
        if self.simulation is None or self.simulation.state is None:
            return False
        else:
            return True

    def schedule(self):
        self._schedule()

    def _schedule(self, ops=None):
        if not self._sys_init:
            raise RuntimeError("System not initialized yet")
        sim = self.simulation
        ops = self._operations if ops is None else ops
        for op in ops:
            if op.is_attached:
                continue
            new_objs = op.attach(sim)
            if isinstance(op, hoomd.integrate._integrator):
                sim._cpp_sys.setIntegrator(op._cpp_obj)
            if new_objs is not None:
                self._compute.extend(new_objs)
        self._scheduled = True
        self._auto_schedule = True

    def _store_reader(self, reader):
        # TODO
        pass

    def __contains__(self, obj):
        return any([op is obj for op in self._operations])

    @property
    def scheduled(self):
        return self._scheduled

    @property
    def integrator(self):
        try:
            return self._integrator
        except AttributeError:
            return None

    @integrator.setter
    def integrator(self, op):
        if not isinstance(op, hoomd.integrate._integrator):
            raise TypeError("Cannot set integrator to a type not derived "
                            "from hoomd.integrator._integrator")
        old_ref = self.integrator
        self._integrator = op
        if self._auto_schedule:
            self._schedule([op])
            if old_ref is not None:
                new_objs = old_ref.notify_detach(self.simulation)
                old_ref.detach()
            if new_objs is not None:
                self._compute.extend(new_objs)

    @property
    def updaters(self):
        return self._updaters

    @property
    def analyzers(self):
        return self._analyzers
