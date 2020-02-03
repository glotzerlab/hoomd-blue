---
name: Release checklist
about: '[for maintainer use]'
title: 'Release v2.XX.Y'
labels: ''
assignees: 'joaander'

---

Release checklist:

- [ ] Run *bumpversion*.
- [ ] Update change log.
  - ``git log --format=oneline --first-parent `git log -n 1 --pretty=format:%H -- CHANGELOG.rst`...``
  - [milestone](https://github.com/glotzerlab/hoomd-blue/milestones)
- [ ] Check readthedocs build, especially change log formatting.
  - [Build status](https://readthedocs.org/projects/hoomd-blue/builds/)
  - [Output](https://hoomd-blue.readthedocs.io/en/latest/)
- [ ] Tag and push.
- [ ] Update conda-forge recipe.
