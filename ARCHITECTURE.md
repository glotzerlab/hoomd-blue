# HOOMD-blue code architecture

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
