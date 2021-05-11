---
name: Release checklist
about: '[for maintainer use]'
title: 'Release v3.0.0-beta.N'
labels: ''
assignees: 'joaander'

---

Release checklist:

- [ ] Update actions versions.
  - See current actions usage with: `rg --no-filename --hidden uses: | awk '{$1=$1;print}' | sort | uniq`
  - Use global search and replace to update them to the latest tags
- [ ] Run *bumpversion*.
- [ ] Update change log.
  - ``git log --format=oneline --first-parent `git log -n 1 --pretty=format:%H -- CHANGELOG.rst`...``
  - [milestone](https://github.com/glotzerlab/hoomd-blue/milestones)
- [ ] Check readthedocs build, especially change log formatting.
  - [Build status](https://readthedocs.org/projects/hoomd-blue/builds/)
  - [Output](https://hoomd-blue.readthedocs.io/en/latest/)
- [ ] Tag and push.
- [ ] Update conda-forge recipe.
- [ ] Update glotzerlab-software.
- [ ] Announce release on mailing list.
