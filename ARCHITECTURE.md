# HOOMD-blue code architecture

ARCHITECTURE files are a way to diffuse the organization and mental picture or
30,000 foot view of a code base as well as provide guidance for developers.  For
more information regarding the concept see
https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html.

**Note**: Details that are reported here are API or "stable".  That is the
purpose of this document is not to show how to use the package, for that see
our [API documentation][hoomd_documentation] on readthedocs.

[hoomd_documentation]: https://hoomd-blue.readthedocs.io/en/latest/

## Hardware and software support

Each minor and major release of HOOMD-blue at a minimum supports:

* x86_64 CPUs released in the four prior years.
* NVIDIA GPUs released in the four prior years.
* Compilers and software dependencies available on the oldest maintained Ubuntu LTS release.
* The two most recent major **CUDA** toolkit versions

## Testing

### Continuous integration

[Github Actions] performs continuous integration testing on HOOMD-blue. GitHub
Actions compiles HOOMD-blue, runs the unit and validation, and and reports the
status to GitHub pull requests. A number of parallel builds test a variety of
compiler and build configurations as defined above.

Visit the [workflows] page to find recent builds. The pipeline configuration
files are in [.github/workflows/] which are built from Jinja templates in
[.github/workflows/templates/] using `make_workflows.py` which is automatically
run by `pre-commit`. To make changes to the workflows, edit the templates.

[GitHub Actions]: https://docs.github.com/en/actions
[workflows]: https://github.com/glotzerlab/hoomd-blue/actions
[.github/workflows/]: .github/workflows/
[.github/workflows/templates/]: .github/workflows/templates/

## Python

The Python code in HOOMD-blue is mostly written to wrap core functionality
written in C++/HIP.  Priority is given to ease of use for users in Python even
at the cost of code complexity (within reason).

**Note**: Most internal functions, classes, and methods are documented for
developers (where they are not feel free to add documentation or request it).
Thus, this section will not go over individual functions or methods in detail
except where necessary.

### Base Data Model

#### Definitions
- added: An object is associated with a `Simulation`.
- attached: An object has its corresponding C++ class instantiated and is
  connected to a `Simulation`.
- unattached: An object's data resides in pure Python and may be or not be
  added.
- detach: The transition from attached to unattached.
- removed: The removal of an unattached object from a `Simulation`.
- sync: The process of making a C++ container match a corresponding Python
  container. See the section on `SyncedList` for a concrete example.
- type parameter: An attribute that must be specified for all types or groups of
  types of a set length.

In Python, many base classes exist to facilitate code reuse. This leads to very
lean user facing classes, but requires more fundamental understanding of the model
to address cases where customization is necessary, or changes are required. The
first aspect of the data model discussed is that of most HOOMD operations or
related classes.

- `_HOOMDGetSetAttrBase`
- `_DependencyRelation`
    - `_HOOMDBaseObject`
        - `Operation`

#### `_DependencyRelation`

`_DependencyRelation` helps define a dependent dependency relationship between
two objects. The class is inherited by `_HOOMDBaseObject` to handle dependent
relationships between operations in Python. The class defines *dependents* and
*dependencies* of an object whose removal from a simulation (detaching) can be
handled by overwriting specific methods defined in `_DependencyRelation` in
`hoomd/operation.py`. See the interface of neighbor lists to pair potentials as
an example of this in `hoomd/md/nlist.py` and `hoomd/md/pair/pair.py`.

#### `_HOOMDGetSetAttrBase`

`_HOOMDGetSetAttrBase` provides hooks for exposing object attributes through two
internal (underscored) attributes `_param_dict` and `_typeparam_dict`.
`_param_dict` is an instance of type `hoomd.data.parameterdicts.ParameterDict`,
and `_typeparam_dict` is a dictionary of attribute names to
`hoomd.data.typeparam.TypeParameter`. This serves as the fundamental way of
specifying object attributes in Python. The class provides a multitude of hooks
to enable custom attribute querying and setting when necessary. See
`hoomd/operation.py` for the source code and the sections on `TypeParameter`'s
and `ParameterDict`'s for more information.

**Note**: This class allows the use of `ParameterDict` and `TypeParameter`
(described below) instances without C++ syncing or attaching. Internal custom
actions use this see `hoomd/custom/custom_action.py` for more information.

#### `_HOOMDBaseObject`

`_HOOMDBaseObject` combines `_HOOMDGetSetAttrBase` and `_DependencyRelation` to
provide dependency handling, validated and processed attribute setting,
pickling, and pybind11 C++ class syncing in an automated and structured way.
Most methods unique to or customized from base classes revolve around allowing
`_param_dict` and `_typeparam_dict` to sync to and from C++ when attached and
not when unattached. See `hoomd/operation.py` for source code. The `_add`,
`_attach`, `_remove`, and `_detach` methods handle the process of adding,
attaching, detaching, and removal.

### Attribute Validation and Defaults

In Python, five fundamental classes exist for value validation, processing, and
syncing when attached: `hoomd.data.parameterdicts.ParameterDict`,
`hoomd.data.parameterdicts.TypeParameterDict`,
`hoomd.data.parameterdicts.AttachedTypeParameterDict`,
`hoomd.data.typeparam.TypeParameter`, and
`hoomd.data.syncedlist.SyncedList`.
These can be/are used by `_HOOMDBaseObject` subclasses as well as others.

#### `ParameterDict`

`ParameterDict` provides a mapping (`dict`) interface from attribute names to
validated and processed values. Each instance of `ParameterDict` has its own
specification defining the validation logic for its keys. `_HOOMDBaseObject`
subclasses automatically sync the `ParameterDict` instance in the `_param_dict`
attribute. This requires that all `ParameterDict` keys be available as
properties of the C++ object using the pybind11 property implementation
(https://pybind11.readthedocs.io/en/stable/classes.html#instance-and-static-fields).
Properties can be read-only which means they will never be set through
`ParameterDict`, but can be through the C++ class constructor. Attempting to set
such a property after attaching will result in an exception being thrown.

This class should be used to define all attributes shared with C++ member
variables that are on a per-object basis (i.e. not per type). Examples of
this can be seen in many HOOMD classes.  One good example is
`hoomd.md.update.ReversePerturbationFlow`.

#### `TypeParameterDict` and `AttachedTypeParameterDict`

These classes work together to define validated mappings from types or groups of
types to values. The type validation and processing logic is identical to that
used of `ParameterDict`, but on a per key basis. In addition, these classes
support advanced indexing features compared to standard Python `dict` instances.
The class also supports smart defaults. These features can come with some
complexity, so looking at the code with `hoomd.data.typeconverter`,
`hoomd.data.smart_default`, and `hoomd.data.parameterdicts`, should help.

The class is used with `TypeParameter` to specify quantities such as pair
potential `params` (see `hoomd/md/pair/pair.py`) and HPMC shape specs (see
`hoomd/hpmc/integrate.py`).

#### `TypeParameter`

This class is a wrapper for `TypeParameterDict` and `AttachedTypeParameterDict`
to work with `_HOOMDBaseObject` subclass instances. It provides very little
independent logic and can be found in `hoomd/data/typeparam.py`.
`_HOOMDBaseObject` expects the values of `_typeparam_dict` keys to be
`TypeParameter` instances. See the methods
`hoomd.operation._HOOMDBaseObject._append_typeparm` and
`hoomd.operation._HOOMDBaseObject._extend_typeparm` for more information.

This class automatically handles attaching and providing the C++ class the
necessary information through a setter interface (retrieving data is similar).
The name of the setter/getter is the camel cased version of the name given to
the `TypeParameter` (e.g. if the type parameter is named `shape_def` then the
methods are `getShapeDef` and `setShapeDef`). These setters and getters need to
be exposed by the internal C++ class `obj._cpp_obj` instance.

#### `SyncedList`

`SyncedList` implements an arbitrary length list that is value validated and
synced with C++. List objects do not need to have a C++ direct counterpart, but
`SyncedList` must be provided a transform function from the Python object to the
expected C++ one. `SyncedList` can also handle the added and attached status of
its items automatically. An example of this class in use is in the MD integrator
for forces and methods (see `hoomd/md/integrate.py`).

#### Value Validation

The API for specifying value validation and processing in most cases is fairly
simple. The spec `{"foo": float, "bar": str}` does what would be expected,
`"foo"` must be a float and `"bar"` a string. In addition, both value do not
have a default. The basis for the API is that the container type `{}` for `dict`
and `set` (currently not supported), `[]` for `list`, and `()` for `tuple`
defines the object type (tuples validators are considered fixed size) while the
value(s) interior to them define the type to expect or a callable defining
validation logic. For instance,
`{"foo": [(float, float)], "bar": {"baz": lambda x: x % 5}` is perfectly valid
and validates or processes as expected. For more information see
`hoomd/data/typeconverter.py`.

#### Type Parameter Defaults

The effects of these defaults is found in `hoomd.data.TypeParameter`'s API
documentation; however the implementation can be found in
`hoomd/data/smart_default.py`. The defaults are similar to the value validation
in ability to be nested but has less customization due to terminating in
concrete values.

### Internal Python Actions

HOOMD-blue version 3 allows for much more interaction with Python objects within
its C++ core. One feature is custom actions (see the tutorial
https://hoomd-blue.readthedocs.io/en/latest/tutorial/03-Custom-Actions-In-Python/00-index.html
or API documentation for more introductory information). When using custom
actions internally for HOOMD, the classes `hoomd.custom._InternalAction` and one
of the `hoomd.custom._InternalOperation` subclasses are to be used. They allow
for the same interface of other HOOMD operations while having their logic
written in Python. See the examples in `hoomd/write/table.py` and
`hoomd/hpmc/tune/move_size.py` for more information.

### Pickling

By default all Python subclasses of `hoomd.operation._HOOMDBaseObject` support
pickling. This is to facilitate restartability and reproducibility of
simulations. For understanding what *pickling* and Python's supported magic
methods regarding it are see https://docs.python.org/3/library/pickle.html. In
general we prefer using `__getstate__` and `__setstate__` if possible to make
class's picklable.  For the implementation of the default pickling support for
`hoomd.operation._HOOMDBaseObject` see the class's `__getstate__` method.
*Notice* that we do not implement a generic `__setstate__`. We rely on Python's
default generally which is somewhat equivalent to `self.__dict__ =
self.__getstate__()`. Adding a custom `__setstate__` method is fine if necessary
(see [hoomd/write/table.py](hoomd/write/table.py)).  However, using `__reduce__`
is an appropriate alternative if is significantly reduces code complexity or has
demonstrable advantages; see [hoomd/filter/set\_.py](hoomd/filter/set_.py) for
an example of this approach.  _Note_ that `__reduce__` requires that a function
be able to fully recreate the current state of the object (this means that often
times the constructor will not work). Also, note `_HOOMDBaseObject`'s support a
class attribute `_remove_for_pickling` that allows attributes to be removed
before pickling (such as `_cpp_obj`).

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
pickling, if a function other than the constructor is used to reinstantiate the
object then any necessary changes should be made (if the constructor is used
then any new arguments to constructor must have defaults via semantic
versioning and no changes should be needed to support pickling). The removal of
internal attributes should not cause problems as well.
