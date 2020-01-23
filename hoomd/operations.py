import hoomd.integrate
import hoomd.meta
from hoomd.syncedlist import SyncedList
from hoomd.meta import _Analyzer, _Updater


def list_validation(type_):
    def validate(value):
        if not isinstance(value, type_):
            raise ValueError("Value {} is of type {}. Excepted instance of "
                             "{}".format(value, type(value), type_))
        else:
            return True
    return validate


def triggered_op_conversion(value):
    return (value._cpp_obj, value.trigger)


class Operations:
    def __init__(self, simulation=None):
        self._simulation = simulation
        self._compute = list()
        self._auto_schedule = False
        self._scheduled = False
        self._updaters = SyncedList(list_validation(_Updater),
                                       triggered_op_conversion)
        self._analyzers = SyncedList(list_validation(_Analyzer),
                                        triggered_op_conversion)
        self._integrator = None

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
        if self._simulation is None or self._simulation.state is None:
            return False
        else:
            return True

    def schedule(self):
        if not self._sys_init:
            raise RuntimeError("System not initialized yet")
        sim = self._simulation
        if self.integrator is not None and not self.integrator.is_attached:
            self.integrator.attach(sim)
        if not self.updaters.is_attached:
            self.updaters.attach(sim, sim._cpp_sys.updaters)
        if not self.analyzers.is_attached:
            self.analyzers.attach(sim, sim._cpp_sys.analyzers)
        self._scheduled = True

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
        return self._integrator

    @integrator.setter
    def integrator(self, op):
        if not isinstance(op, hoomd.integrate._BaseIntegrator):
            raise TypeError("Cannot set integrator to a type not derived "
                            "from hoomd.integrator._integrator")
        old_ref = self.integrator
        self._integrator = op
        if self._auto_schedule:
            new_objs = op.attach(self._simulation)
            if old_ref is not None:
                old_ref.notify_detach(self._simulation)
                old_ref.detach()
            if new_objs is not None:
                self._compute.extend(new_objs)

    @property
    def updaters(self):
        return self._updaters

    @property
    def analyzers(self):
        return self._analyzers
