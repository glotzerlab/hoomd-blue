from hoomd.python_action import _PythonAction, _InternalPythonAction
from hoomd.custom_action import _CustomAction


class CustomAnalyzer(_CustomAction):
    pass


class _PythonAnalyzer(_PythonAction):
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


class _InternalPythonAnalyzer(_InternalPythonAction):
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
