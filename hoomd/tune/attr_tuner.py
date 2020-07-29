from abc import ABCMeta, abstractmethod


class _TuneDefinition(metaclass=ABCMeta):
    def __init__(self, target, domain=None):
        self.domain = domain
        self._target = target

    def in_domain(self, value):
        if self.domain is None:
            return True
        else:
            lower_bound, upper_bound = self.domain
            return ((lower_bound is None or lower_bound <= value)
                    and (upper_bound is None or value <= upper_bound))

    def wrap_into_domain(self, value):
        if self._domain is None:
            return value
        else:
            lower_bound, upper_bound = self.domain
            if lower_bound is not None and value < lower_bound:
                return lower_bound
            elif upper_bound is not None and value > upper_bound:
                return upper_bound
            else:
                return value

    @property
    def x(self):
        return self._get_x()

    @property
    def max_x(self):
        if self.domain is None:
            return None
        else:
            return self.domain[1]

    @property
    def min_x(self):
        if self.domain is None:
            return None
        else:
            return self.domain[0]

    @x.setter
    def x(self, value):
        return self._set_x(self.wrap_into_domain(value))

    @property
    def y(self):
        return self._get_y()

    @property
    def target(self):
        return self._get_target()

    @target.setter
    def target(self, value):
        self._set_target(value)

    @abstractmethod
    def _get_x(self):
        pass

    @abstractmethod
    def _set_x(self):
        pass

    @abstractmethod
    def _get_y(self):
        pass

    def _get_target(self):
        return self._target

    def _set_target(self, value):
        self._target = value

    @property
    def domain(self):
        if self._domain is not None:
            return tuple(self._domain)
        else:
            return None

    @domain.setter
    def domain(self, value):
        if value is not None and not len(value) == 2:
            raise ValueError("domain must be a sequence of length two.")
        self._domain = value

    def __hash__(self):
        raise NotImplementedError("This object is not hashable.")

    def __eq__(self, other):
        raise NotImplementedError("This object is not equatable.")


class ManualTuneDefinition(metaclass=ABCMeta):
    def __init__(self, get_y, target_y, get_x, set_x, domain=None):
        self.__get_x = get_x
        self.__set_x = set_x
        self.__get_y = get_y
        self._target = target_y
        if domain is not None and not len(domain) == 2:
            raise ValueError("domain must be a sequence of length two.")
        self._domain = domain

    def _get_x(self):
        return self.__get_x()

    def _set_x(self, value):
        return self.__set_x(value)

    def _get_y(self):
        return self.__get_y()

    def _get_target(self):
        return self._target

    def _set_target(self, value):
        self._target = value

    def __hash__(self):
        return hash((self.__get_x, self.__set_x, self.__get_y, self._target))

    def __eq__(self, other):
        return (self.__get_x == other.__get_x
                and self.__set_x == other.__set_x
                and self.__get_y == other.__get_y
                and self._target == other._target)


class Solver(metaclass=ABCMeta):
    @abstractmethod
    def _solve_one(self, tunable):
        pass

    def solve(self, tunables):
        return all(self._solve_one(tunable) for tunable in tunables)


class ScaleSolver(Solver):
    def __init__(self, max_scale=2.0, gamma=2.0, tol=1e-5):
        self.max_scale = max_scale
        self.gamma = gamma
        self.tol = tol

    def _solve_one(self, tunable):
        x, y, target = tunable.x, tunable.y, tunable.target
        if abs(y - target) <= self.tol:
            return True

        if y > 0:
            scale = ((1.0 + self.gamma) * y) / (target + (self.gamma * y))
        else:
            # y was zero. Try a value an order of magnitude smaller
            scale = 0.1
        if (scale > self.max_scale):
            scale = self.max_scale
        # Ensures we stay within the tunable's domain (i.e. we don't take on
        # values to high or low).
        tunable.x = tunable.wrap_into_domain(scale * x)
        return False


class SecantSolver(Solver):
    def __init__(self, gamma=0.9, tol=1e-5):
        self.gamma = gamma
        self._previous_pair = dict()
        self.tol = tol

    def _solve_one(self, tunable):
        x, y, target = tunable.x, tunable.y, tunable.target
        if abs(y - target) <= self.tol:
            return True

        if tunable not in self._previous_pair:
            # We must perturb x some to get a second point to find the correct
            # root.
            new_x = tunable.wrap_into_domain(x * 1.1)
            if new_x == x:
                new_x = tunable.wrap_into_domain(x * 0.9)
                if new_x == x:
                    raise RuntimeError("Unable to perturb x for secant solver.")
        else:
            # standard secant formula. A brief note, we use f(x) = y - target
            # since this is the root we are searching for.
            old_x, old_f_x = self._previous_pair[tunable]
            self._previous_pair[tunable] = (x, y - target)
            f_x = y - target
            dxdf = (x - old_x) / (f_x - old_f_x)
            new_x = x - (self.gamma * f_x * dxdf)

        self._previous_pair[tunable] = (x, y - target)
        tunable.x = new_x
        return False
