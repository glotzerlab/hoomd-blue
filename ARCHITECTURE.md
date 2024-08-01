# HOOMD-blue code architecture

This document provides a high level overview of the codebase for developers. Details that are reported here relate to the internal API design. For detail on the user-facing public API, see the
[user documentation][hoomd_documentation].

[hoomd_documentation]: https://hoomd-blue.readthedocs.io/en/latest/

## Hardware and software support

Each minor and major release of HOOMD-blue at a minimum supports:

* x86_64 CPUs released in the four prior years.
* NVIDIA GPUs released in the four prior years.
* Compilers and software dependencies available on the two most recent Ubuntu LTS releases.
* The most recent major **CUDA** toolkit version.

## Testing

### Continuous integration

[Github Actions] performs continuous integration testing on HOOMD-blue. GitHub
Actions compiles HOOMD-blue, runs the unit tests, and and reports the
status to GitHub pull requests. A number of parallel builds test a variety of
compiler and build configurations as defined above.

Visit the [workflows] page to find recent builds. The pipeline configuration
files are in [.github/workflows/] and are built from Jinja templates in
[.github/workflows/templates/] using `make_workflows.py` which is automatically
run by `pre-commit`. To make changes to the workflows, edit the templates.

[GitHub Actions]: https://docs.github.com/en/actions
[workflows]: https://github.com/glotzerlab/hoomd-blue/actions
[.github/workflows/]: .github/workflows/
[.github/workflows/templates/]: .github/workflows/templates/

## Build system

HOOMD-blue consists of C++ code, Python code, and the CMake configuration scripts necessary to
compile and assemble a functioning Python module. The CMake configuration copies the `.py` files
into the build directory so that developers can iteratively build and test without needing to make
the `install` target and modify files outside the build directory. For more details on using the
build system, see the [user documentation][hoomd_documentation].

HOOMD-blue's CMake configuration follows the most modern CMake standards possible given the software
support constraint given above. For example, it uses `find_package(... CONFIG)` to find package
config files. It manages the linked libraries and additional include directories with the
appropriate visibility in `target_link_libraries` to pass these dependencies on to external
components. HOOMD-blue itself produces a CMake config file to use with `find_package`.

HOOMD has many optional dependencies (e.g. LLVM) and developers can build with or without components
or features (e.g. HPMC). These are set in CMake `ENABLE_*` and `BUILD_*` variables and passed
into the C++ code as preprocessor definitions. New code must observe these definitions so that
the code compiles correctly (or is excluded as needed) when a given option is set or not set.

## C++

The majority of HOOMD-blue's simulation engine is implemented in C++ with a design that strikes a
balance between performance, readability, and maintenance burden. In general, most classes in HOOMD
operate on the entire system of particles so that they can implement loops over the entire system
efficiently. To the extent possible, each class is responsible for a single isolated task and is
composable with other classes. Where needed classes provide a signal/slot mechanism to escape the
isolation and provide notification to client classes when relevant data changes. For example,
`ParticleData` emits a signal when the system box size changes.

This document provides a high level overview of the design, describing how the elements
interoperate. For full details on these classes, see the documentation in the source code comments.
With few exceptions, the C++ code for a class `ClassName` is in `ClassName.h`, `ClassName.cc`,
`ClassName.cuh`, and/or `ClassName.cu`. These files are in the directory corresponding to the Python
package where they reside.

## Execution model

HOOMD-blue implements all operations on the CPU and GPU. The CPU implementation is vanilla C++, and
the GPU implementation uses HIP to support both AMD and NVIDIA GPUs. The `ExecutionConfiguration`
class selects the device (CPU, GPU, or multiple GPUs) and configures global execution options. Each
operation class that needs to know the device configuration is given a shared pointer to the
`ExecutionConfiguration` - most classes store this in the member variable `m_exec_conf`.

To minimize code duplication and to provide a common interface for both CPU and GPU code paths,
HOOMD defines the CPU implementation of an operation in `ClassName` and the GPU implementation in a
subclass `ClassNameGPU`. The base class defines the data structures, parameters, getter/setter
methods, initialization, and other common tasks. The GPU class overrides key methods to perform the
expensive part of the computation on the GPU. The GPU subclass may use alternate data structures for
performance if needed, but this increases the code maintenance burden.

HOOMD-blue uses MPI for domain decomposition simulations on multiple CPUs or GPUs. The
`MPIConfiguration` class (held by `ExecutionConfiguration`) defines the MPI partition and ranks.
Many classes, such as `ParticleData` provide separate methods to access *local* properties on the
current rank and *global* properties (e.g. `ParticleData::getBox` and `ParticleData::getGlobalBox`).
The `Communicator` class is responsible for communicating and migrating particles and bonds between
neighboring ranks.

## Data model

If you think HOOMD-blue's data model is unnecessarily complex, you are correct. Understand that it
is a product of continual development since the year 2007 and has grown along with improving CUDA
functionality. No developer has the time or inclination to completely refactor the entire codebase
to be consistent with the current features. New code should follow the guidelines documented here.

Base data types:

* `Scalar` - Base floating point data type for particle properties. Configurable to either `double`
  or `float` at compile time.
* `Scalar2`, `Scalar3`, `Scalar4`, `int2`, `int3`, ... - 2,3, and 4-vectors of values. These map to
  the CUDA vector types which are aligned properly to enable efficient vector load instructions on
  the GPU. Use these types to store arrays of vector data. Prefer the `2` and `4` size vectors as
  they require fewer memory transactions to read/write than 3-vectors.
* `vec2<Real>` `vec3<Real>`, `quat<Real>` - Templated vector and quaternion types defined in
  `VectorMath.h`. Use these types and the corresponding methods (e.g. `dot`, `operator+`) to perform
  vector and quaternion math with a clean and readable syntax. Convert from and to the `ScalarN`
  vector types when reading inputs and writing outputs to arrays.

Array data:

* `GPUArray<T>` - Template array data type that stores two copies of the data, one on the CPU and
  one on the GPU. Use `ArrayHandle` to request a pointer to the data, which will copy the most
  recently written data to the requested device when needed. New code should use `ArrayHandle` to
  access existing data structures that use `GPUArray`. New code **should not** define new `GPUArray`
  arrays, use `GlobalArray` or `std::vector` with a managed allocator.
* `GlobalArray<T>` - Template array data type that stores one copy of the data in CUDA's unified
  memory in single process, multi-GPU execution. When using a single GPU per process, falls back on
  `GPUArray`. Use `ArrayHandle` to access data in `GlobalArray`.
* `std::vector<T, hoomd::detail::managed_allocator<T>>` - Store array data in a `std::vector` in
  CUDA's unifed memory. This data type is useful for parameter arrays that are exposed to Python.

When using `GlobalArray` or `std::vector<T, hoomd::detail::managed_allocator<T>>`, call
`cudaMemadvise` to set the appropriate memory hints for the array. Small parameter arrays should be
set to `cudaMemAdviseSetReadMostly`. Larger arrays accessed in portions in single-process multi-GPU
execution should be set to `cudaMemAdviseSetPreferredLocation` appropriately for the different
portions of the array.

System data:

* `ParticleData` - Stores the particle positions, velocities, masses, and other per-particle
  properties.
* `ParticleGroup` - Stores the indices of a subset of particles in the system.
* `BondedGroupData` - Stores bonds, angles, and dihedrals.
* `SystemDefinition` - Combines particle data and all bond data.
* `SnapshotSystemData` - Stores a copy of the global system state. Used for file I/O and user
  initialization/analysis.

## Class overview

Users configure HOOMD simulations by defining lists of **Operations** that act on the system state
and schedule when they occur with `Trigger`. The `System` class manages these lists and executes the
simulation in `System::run`. There are numerous types of operations, each with their own base class:

* `Compute` - Compute properties of the system state without modifying it. May provide results
  directly to the user and/or another operation class (e.g. `PotentialPair` uses `NeighborList`).
  A single compute instance may be used by multiple operations, so it must avoid recomputing results
  when `compute` is called multiple times during a single timestep.
* `Updater` - Change the system state.
* `Integrator` - Move the system state forward in time. There is only one `Integrator` in a
  `System`.
* `Analyzer` (named `Writer` in Python) - Computes properties of the system state without modifying
  it and writes them to an output stream or file.
* `Tuner` - Modify parameters of other operations without changing the system state or the
  correctness of the simulation. For example, `SFCPackTuner` reorders the particles in memory to
  improve performance by reducing cache misses.

### HPMC

The integrator `HPMCIntegrator` defines and stores the core parameters of the simulation, such as
the particle shape. All HPMC specific operations (such as `UpdaterClusters` and `UpdaterBoxMC`) take
a shared pointer to the HPMC integrator to access this information.

### MD

There are two MD integrators: `IntegratorTwoStep` implements normal MD simulations and
`FIREENergyMinimizer` implements energy minimization. In MD, the integrator maintains the list of
user-supplied forces (`ForceCompute`) to apply to particles. Both integrators also maintain a list
of user-supplied integration methos (`IntegrationMethodTwoStep`). Each method instance operates
on a single particle group (`ParticleGroup`) and is solely responsible for integrating the equations
of motion of all particles in that group.

## Template evaluator

Many operations in HOOMD-blue provide similar functionality with different functional forms. For
example pair potentials with many different V(r) and HPMC integration with many different particle
shape classes. To reduce code duplication while maintaining high performance, HOOMD-blue uses
template evaluator classes combined with a single implementation of the general method. This allows
the method (e.g. pair potential evaluation) to be implemented only twice (once on the GPU and once
on the CPU) while each specific evaluator (e.g. V(r)) is also implemented only once. With the
functional form defined in a template class, the compiler is free to inline the evaluation of that
function into the inner loop generated in each template instantiation.

To add a new functional form to the code, a developer must:

1. Implement the evaluator class.
2. Instantiate the GPU kernel driver with the evaluator.
3. Instantiate the method class with the evaluator and export them to Python.
4. Add the Python Operation class to wrap the C++ implementation.

See existing examples in the codebase (e.g. grep for `EvaluatorPairLJ>`) for details. For most
classes, steps 2 and 3 are performed using file templates expanded in CMakeLists.txt and exported in
the appropriate `module*.cc` file.

## GPU kernel driver functions

Early versions of CUDA could compile only a minimal subset of C++ code. While the modern CUDA
compilers are much improved, there are still occasional cases where including complex C++ code like
`pybind11.h` (or even using some standard library features) causes compile errors. This can occur
even when the use of that code is used only in host code. To work around these cases, all GPU
kernels in HOOMD-blue are called via minimal driver functions. These driver functions are not member
functions of their respective class, and therefore must take a long C-style argument list consisting
of bare pointers to data arrays, array sizes, etc... The ABI for these calls is not strictly C, as
driver functions may be templated on functor classes and/or accept lightwight C++ objects as
parameters (such as `BoxDim`).

## Autotuning

HOOMD-blue automatically tunes kernel block sizes, threads per particle, and other kernel launch
paramters. The `Autotuner` class manages the sparse multi-dimensional of parameters for each kernel.
It dynamically cycles through the possible parameters and records the performance of each using CUDA
events. After scanning through all parameters, it selects the best performing one to continue
executing. GPU code in HOOMD-blue should instantiate and use one `Autotuner` for each kernel.
Classes that use have `Autotuner` member variables should inherit from `Autotuned` which tracks all
the autotuners and provides a UI to users. When needed, classes should override the base class
`isAutotuningComplete` and `startAutotuning` as to pass the calls on to child objects. not otherwise
managed by the `Simulation`. For example, `PotentialPair::isAutotuningComplete`, calls both
`ForceCompute::isAutotuningComplete` and `m_nlist->isAutotuningComplete` and combines the results.

## Python

The Python code in HOOMD-blue is mostly written to wrap core functionality
written in C++.  Priority is given to ease of use for users in Python even
at the cost of code complexity (within reason).

**Note**: Most internal functions, classes, and methods are documented for
developers (where they are not feel free to add documentation or request it).
Thus, this section will not go over individual functions or methods in detail
except where necessary.

### Base Data Model

#### Definitions
- attached: An object has its corresponding C++ class instantiated and is
  connected to a `Simulation`.
- unattached: An object's data resides in pure Python.
- detach: The transition from attached to unattached.
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
`hoomd/operation.py`.

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
not when unattached. See `hoomd/operation.py` for source code. The
`_attach_hook`, and `_detach_hook` methods handle the subclass specific logic
of attaching and detaching.

### Attribute Validation and Defaults

In Python, four fundamental classes exist for value validation, processing, and
syncing when attached: `hoomd.data.parameterdicts.ParameterDict`,
`hoomd.data.parameterdicts.TypeParameterDict`,
`hoomd.data.typeparam.TypeParameter`, and
`hoomd.data.syncedlist.SyncedList`.
These can be/are used by `_HOOMDBaseObject` subclasses as well as others.
In addition, classes provided by `hoomd.data.collection` allow for nested
editing of Python objects while maintaining a correspondence to C++.

#### `ParameterDict`

`ParameterDict` provides a mapping (`dict`) interface from attribute names to
validated and processed values. Each instance of `ParameterDict` has its own
specification defining the validation logic for its keys. `ParameterDict`
contains logic when given a pybind11 produced Python object can sync between C++
and Python. See `ParameterDict.__setitem__` for this logic. Attribute specific
logic can be created using the `_getters` and `_setters` attributes. The logic
requires (outside custom getters and setters that all `ParameterDict` keys be
available as properties of the C++ object using the pybind11 property
implementation
(https://pybind11.readthedocs.io/en/stable/classes.html#instance-and-static-fields).
Properties can be read-only which means they will never be set through
`ParameterDict`, but can be through the C++ class constructor. Attempting to set
such a property after attaching will result in an `MutabiliyError` being thrown.

This class should be used to define all attributes shared with C++ member
variables that are on a per-object basis (i.e. not per type). Examples of
this can be seen in many HOOMD classes.  One good example is
`hoomd.md.update.ReversePerturbationFlow`.

#### `TypeParameterDict`

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

This class is a wrapper for `TypeParameterDict` to work with `_HOOMDBaseObject`
subclass instances. It provides very little independent logic and can be found
in `hoomd/data/typeparam.py`. This class primarily serves as the source of user
documentation for type parameters. `_HOOMDBaseObject` expects the values of
`_typeparam_dict` keys to be `TypeParameter` instances. See the methods
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
expected C++ one. `SyncedList` can also handle the attached status of its items
automatically. An example of this class in use is in the MD integrator for
forces and methods (see `hoomd/md/integrate.py`).

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

#### HOOMD Collection Types

In `hoomd.data.collection` classes exist which mimic Python `dict`s, `list`s,
and `tuples` (`set`s could easily be added). These classes keep a reference to
their owning object (e.g. a `ParameterDict` or `TypeParameterDict` instance).
Through this reference and read-modify-write approach, these classes facilitate
nested editing of Python attributes while maintaining a synced status in C++. In
general, developers should not need to worry about this as the use of these
classes is automated through previously described classes.

### Internal Python Actions

HOOMD-blue version 3 allows for much more interaction with Python objects within
its C++ core. One feature is custom actions (see the tutorial
https://hoomd-blue.readthedocs.io/en/latest/tutorial/04-Custom-Actions-In-Python/00-index.html
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

A full test suite for collection-like objects can be found in
[hoomd/confest.py](hoomd/conftest.py). This suite is used by all HOOMD
collection like classes.

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

## Zero Copy Buffer Access

HOOMD allows for C++ classes to expose their GPU and CPU data buffers directly
in Python using the `__cuda_array_interface__` and `__array_interface__`. This
behavior is controlled using the `hoomd.data.local_access._LocalAcces` class in
Python and the classes found in `hoomd/PythonLocalDataAccess.h`. See these files
for more details. For example implementations look at `hoomd/ParticleData.h`.

## Directory structure

The top level directories are:

* `CMake` - CMake scripts.
* `example_plugin` - External developers to copy this to start developing an external component.
* `hoomd` - Source code for the `hoomd` Python package. Subdirectories under `hoomd` follow the same
  layout as the final Python package.
* `sphinx-doc` - Sphinx configuration and input files for the user-facing documentation.

## Documentation

## User

The user facing documentation is compiled into a human readable document by Sphinx. The
documentation consists of `.rst` files in the `sphinx-doc` directory and the docstrings of
user-facing Python classes in the implementation (imported by the Sphinx autodoc extension).
HOOMD-blue's Sphinx configuration defines mocked imports so that the documentation may be built from
the source directory without needing to compile the C++ source code. This is greatly beneficial when
building the documentation on readthedocs.

The tutorial portion of the documentation is written in Jupyter notebooks housed in the
[hoomd-examples][hoomd_examples] repository. HOOMD-blue includes these in the generated Sphinx
documentation using [nbsphinx][nbsphinx].

[hoomd_examples]: https://github.com/glotzerlab/hoomd-examples
[nbsphinx]: https://nbsphinx.readthedocs.io/

## Detailed developer documentation

Like the user facing classes, internal Python classes document themselves with docstrings.
Similarly, C++ classes provide developer documentation in Javadoc comments. Browse the developer
documentation by viewing the source directly as HOOMD-blue provides no configuration for C++
documentation generation tools.
