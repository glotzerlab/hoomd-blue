from hoomd.custom import (CustomOperation, _InternalCustomOperation, Action)
from hoomd.operation import Writer


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


class CustomWriter(CustomOperation, _WriterProperty, Writer):
    """Writer wrapper for `hoomd.custom.Action` objects.

    For usage see `hoomd.custom.CustomOperation`.
    """
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'


class _InternalCustomWriter(_InternalCustomOperation, _WriterProperty, Writer):
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'
