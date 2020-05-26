from hoomd.custom_operation import _CustomOperation, _InternalCustomOperation
from hoomd.custom_action import CustomAction


class _AnalyzerProperty:
    @property
    def analyzer(self):
        return self._action

    @analyzer.setter
    def analyzer(self, analyzer):
        if isinstance(analyzer, CustomAction):
            self._action = analyzer
        else:
            raise ValueError("analyzer must be an instance of CustomAction")


class CustomAnalyzer(_CustomOperation, _AnalyzerProperty):
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'


class _InternalCustomAnalyzer(_InternalCustomOperation, _AnalyzerProperty):
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'
