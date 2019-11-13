import hoomd._hoomd as _hoomd


class ParticleSelector:

    def __init__(self, *args, **kwargs):
        args_str = ''.join([str(arg) for arg in args])
        kwargs_str = ''.join([str(value)for value in kwargs.values()])
        self.args_str = args_str
        self.kwargs_str = kwargs_str
        _id = hash(self.__class__.__name__ + args_str + kwargs_str)
        self._id = _id

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self._id == other._id

    def __call__(self, state):
        raise NotImplementedError

    @property
    def _selector(self):
        raise NotImplementedError


class All(ParticleSelector):
    def __init__(self):
        super().__init__()

    @property
    def _selector(self):
        return _hoomd.ParticleSelectorAll()
