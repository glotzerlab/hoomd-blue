---
name: Release checklist
about: '[for maintainer use]'
title: 'Release 4.8.1'
labels: ''
assignees: 'joaander'

---

Minor and major releases:

- [ ] Update tutorial submodule.
- [ ] Check for new or duplicate contributors since the last release:
  `comm -13 (git log $(git describe --tags --abbrev=0) --format="%aN <%aE>" | sort | uniq | psub) (git log --format="%aN <%aE>" | sort | uniq | psub)`.
  Add entries to `.mailmap` to remove duplicates.
- [ ] Ensure that added features have Sybil examples.
- [ ] Run [hoomd-benchmarks](https://github.com/glotzerlab/hoomd-benchmarks), check for performance
  regressions with the previous release, and post the tables in the release pull request.
- [ ] Run [hoomd-validation](https://github.com/glotzerlab/hoomd-validation).

All releases:

- [ ] Update change log.
  - ``git log --format=oneline --first-parent $(git log -n 1 --pretty=format:%H -- CHANGELOG.rst)...``
- [ ] Check readthedocs build, especially change log formatting.
- [ ] Run *bumpversion*.
- [ ] Tag and push.
- [ ] Update conda-forge recipe.
- [ ] Update *glotzerlab-software*.
- [ ] Update *hoomd-component-template*.
- [ ] Update *hpmc-energy-template*.
