Variable period specification
=============================

Most updaters and analyzers in hoomd accept a variable period specification.
Just specify a function taking a single argument to the period parameter.

For example, dump gsd files at time steps 1, 10, 100, 1000, ...::

    dump.gsd(filename="dump.gsd", period = lambda n: 10**n)

More examples::

    dump.gsd(filename="dump.gsd", period = lambda n: n**2)
    dump.gsd(filename="dump.gsd", period = lambda n: 2**n)
    dump.gsd(filename="dump.gsd", period = lambda n: 1005 + 0.5 * 10**n)

The object passed into *period* must be callable, accept one argument, and return
a floating point number or integer. The function should also be monotonically increasing.

- First, the current time step of the simulation is saved when the analyzer is created.
- *n* is also set to 1 when the analyzer is created
- Every time the analyzer performs it's output, it evaluates the given function at the current value of *n*
  and records that as the next time step to perform the analysis. *n* is then incremented by 1
