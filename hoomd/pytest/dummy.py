from hoomd.meta import _Operation
from hoomd.parameterdicts import TypeParameterDict, RequiredArg
from hoomd.typeparam import TypeParameter


class DummySimulation:
    def __init__(self):
        self.state = DummyState()
        self.operations = DummyOperations()


class DummyState:
    def __init__(self):
        pass

    @property
    def particle_types(self):
        return ['A', 'B']


class DummyOperations:
    pass


class DummyCppObj:
    def __init__(self):
        self._dict = dict()

    def setTypeParam(self, type_, value):
        self._dict[type_] = value

    def getTypeParam(self, type_):
        return self._dict[type_]

    @property
    def param1(self):
        return self._param1

    @param1.setter
    def param1(self, value):
        self._param1 = value

    @property
    def param2(self):
        return self._param2

    @param2.setter
    def param2(self, value):
        self._param2 = value


class DummyOperation(_Operation):
    def __init__(self):
        super().__init__(self)
        self._param_dict = dict(param1=1, param2=2)
        tp1 = TypeParameter(name='type_param', type_kind='particle_types',
                            param_dict=TypeParameterDict(foo=1,
                                                         bar=RequiredArg,
                                                         baz='hello')
                            )
        self._add_typeparam(tp1)

    def attach(self, sim):
        self._cpp_obj = DummyCppObj()
        self._apply_param_dict()
        self._apply_typeparam_dict(self._cpp_obj, sim)
