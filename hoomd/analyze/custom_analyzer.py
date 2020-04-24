from hoomd.custom_operation import _CustomOperation, _InternalCustomOperation
from hoomd.custom_action import _CustomAction, _InternalCustomAction


class _CustomAnalyzerAction(_CustomAction):
    def analyze(self, timestep):
        return self.act(timestep)


class _InternalCustomAnalyzerAction(_InternalCustomAction):
    def analyze(self, timestep):
        return self.act(timestep)


class _CustomAnalyzer(_CustomOperation):
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'
    _cpp_action = 'analyze'

    @property
    def analyzer(self):
        return self._action

    @analyzer.setter
    def analyzer(self, analyzer):
        if isinstance(analyzer, _CustomAction):
            self._action = analyzer
        else:
            raise ValueError("analyzer must be an instance of _CustomAction")


class _InternalCustomAnalyzer(_InternalCustomOperation):
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'
    _cpp_action = 'analyze'

    @property
    def analyzer(self):
        return self._action

    @analyzer.setter
    def analyzer(self, analyzer):
        if isinstance(analyzer, _CustomAction):
            self._action = analyzer
        else:
            raise ValueError("analyzer must be an instance of _CustomAction")
