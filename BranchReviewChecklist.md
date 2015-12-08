# Checklist for review

Here is a brief checklist to go through when reviewing new feature branches for inclusion into the mainline.

* Functionality is implemented in a general and flexible fashion
* Performance
    * Performance of existing features is not degraded
    * New functionality is as optimized as is feasible for the current generation of GPUs
    * GPU kernels are connected to autotuners (if applicable)
* Documentation
    * Code is documented with doxygen comments
    * User documentation exists, is complete and understandable, and is linked to the index.
* Code
   * Compiles without warnings
   * No unnecessary code modifications are made
   * Well documented and understandable
   * Meets style guidelines
   * Passes `hoomd_lint_detector` test
* Tests
    * Reasonable python level tests for all new functionality
    * `hoomd_script` unit tests exist for all new user level script commands
    * Tests pass on all supported hardware configurations
    * Valid results are produced for several different validity tests
    * C++ level tests of individual components if necessary
* Credits page updated appropriately
