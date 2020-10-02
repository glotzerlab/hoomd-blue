from hoomd.custom import (
    _CustomOperation, _InternalCustomOperation, Action)
from hoomd.operation import _Analyzer


class _WriterProperty:
    @property
    def analyzer(self):
        return self._action

    @analyzer.setter
    def analyzer(self, analyzer):
        if isinstance(analyzer, Action):
            self._action = analyzer
        else:
            raise ValueError(
                "analyzer must be an instance of hoomd.custom.Action")


class CustomWriter(_CustomOperation, _WriterProperty, _Analyzer):
    """Analyzer wrapper for `hoomd.custom.Action` objects.

    For usage see `hoomd.custom._CustomOperation`.
    """
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'


class _InternalCustomAnalyzer(
        _InternalCustomOperation, _WriterProperty, _Analyzer):
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'
