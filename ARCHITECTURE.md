# HOOMD-blue code architecture

ARCHITECTURE files are a way to diffuse the organization and mental picture or
30,000 foot view of a code base as well as provide guidance for developers.  For
more information regarding the concept see
https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html.

**Note**: Details that are reported here are API or "stable".  That is the
purpose of this document is not to show how to use the package, for that see
our [API documentation][hoomd_documentation] on readthedocs.

[hoomd_documentation]: https://hoomd-blue.readthedocs.io/en/latest/

## Testing

### Continuous integration

[Azure Pipelines][azp_docs] performs continuous integration testing on
HOOMD-blue. Azure Pipelines compiles HOOMD-blue, runs the unit, validation, and
style tests and reports the status to GitHub pull requests. A number of parallel
builds test a variety of compiler and build configurations, including:

* The 2 most recent **CUDA** toolkit versions
* **gcc** and **clang** versions including the most recent releases back to the
  defaults provided by the oldest maintained Ubuntu LTS release.

Visit the [glotzerlab/hoomd-blue][hoomd_builds] pipelines page to find recent
builds. The pipeline configuration files are in [.azp/](.azp/) which reference
templates in [.azp/templates/](.azp/templates/).

[azp_docs]: https://docs.microsoft.com/en-us/azure/devops/pipelines
[hoomd_builds]: https://dev.azure.com/glotzerlab/hoomd-blue/_build

## Python

The Python code in HOOMD-blue is mostly written to wrap core functionality
written in C++/HIP.  Priority is given to ease of use for users in Python even
at the cost of code complexity (within reason).

_TODO_: Add the basic inheritance and features of HOOMD-blue Python
objects/operations.

### Pickling

By default all Python subclasses of `hoomd.operation._HOOMDBaseObject` support
pickling. This is to facilitate restartability and reproducibility of
simulations. For understanding what *pickling* and Python's supported magic
methods regarding is see https://docs.python.org/3/library/pickle.html. In
general we prefer using `__getstate__` and `__setstate__` if possible to make
class's picklable.  For the implementation of the default pickling support for
`hoomd.operation._HOOMDBaseObject` see the class's `__getstate__` method.
*Notice* that we do not implement a generic `__setstate__`. We rely on Python's
default generally which is somewhat equivalent to `self.__dict__ =
self.__getstate__()`. Added a custom `__setstate__` method is fine if necessary
(see [hoomd/write/table.py](hoomd/write/table.py)).  However, using `__reduce__`
is an appropriate alternative if is significantly reduces code complexity or has
demonstrable advantages; see [hoomd/filter/set\_.py](hoomd/filter/set_.py) for
an example of this approach.  _Note_ that `__reduce__` requires that a function
be able to fully recreate the current state of the object (this means that often
times the constructor will not work).

**Testing**

To test the pickling of objects see the helper methods in
[hoomd/confest.py](hoomd/conftest.py), `pickling_check` and
`operation_pickling_check`. All objects that are expected to be picklable and
this is most objects in HOOMD-blue should have a pickling test.

**Pybind11 Pickling**

For some simple objects like variants or triggers which have very thin Python
wrappers, supporting pickling using pybind11 (see
https://pybind11.readthedocs.io/en/stable/advanced/classes.html#pickling-support)
is acceptable as well. Care just needs to be made that users are not exposed to
C++ classes that "slip" out of their Python subclasses which can happen if no
reference in Python remains to a unpickled object. See
[hoomd/Trigger.cc](hoomd/Trigger.cc) for examples of using pybind11.

**Supporting Class Changes**

Supporting pickling with semantic versioning leads to the need to add support
for objects pickled in version 3.x to work with 3.y, y > x. If new parameters
are added in version "y", then a `__setstate__` method needs to be added if
`__getstate__` and `__setstate__` is the pickling method used for the object.
This `__setstate__` needs to add a default attribute value if one is not
provided in the `dict` given to `__setstate__`. If `__reduce__` was used for
pickling, the any new arguments to constructor must have defaults via semantic
versioning and no changes should be needed to support pickling. The removal of
internal attributes should not cause problems as well.
