from hoomd.custom_operation import _CustomOperation, _InternalCustomOperation
from hoomd.custom_action import _CustomAction, _InternalCustomAction


class _AnalyzeMethod:
    def analyze(self, timestep):
        return self.act(timestep)


class _CustomAnalyzerAction(_CustomAction, _AnalyzeMethod):
    pass


class _InternalCustomAnalyzerAction(_InternalCustomAction, _AnalyzeMethod):
    pass


class _AnalyzerProperty:
    @property
    def analyzer(self):
        return self._action

    @analyzer.setter
    def analyzer(self, analyzer):
        if isinstance(analyzer, _CustomAction):
            self._action = analyzer
        else:
            raise ValueError("analyzer must be an instance of _CustomAction")


class _CustomAnalyzer(_CustomOperation, _AnalyzerProperty):
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'


class _InternalCustomAnalyzer(_InternalCustomOperation, _AnalyzerProperty):
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'
