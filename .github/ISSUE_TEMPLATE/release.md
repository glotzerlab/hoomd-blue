---
name: Release checklist
about: '[for maintainer use]'
title: 'Release 4.x.y'
labels: ''
assignees: 'joaander'

---

Release checklist:

- [ ] Update tutorial submodule.
- [ ] Update change log.
  - ``git log --format=oneline --first-parent `git log -n 1 --pretty=format:%H -- CHANGELOG.rst`...``
  - [milestone](https://github.com/glotzerlab/hoomd-blue/milestones)
- [ ] Check readthedocs build, especially change log formatting.
  - [Build status](https://readthedocs.org/projects/hoomd-blue/builds/)
  - [Output](https://hoomd-blue.readthedocs.io/en/latest/)
- [ ] Update actions versions.
  - See current actions usage with: `rg --no-filename --hidden uses: | awk '{$1=$1;print}' | sort | uniq`
  - Use global search and replace to update them to the latest tags
- [ ] Check for new or duplicate contributors since the last release:
  `comm -13 <(git log LAST_TAG --format="%aN <%aE>" | sort | uniq) <(git log --format="%aN <%aE>" | sort | uniq)`.
  Add entries to `.mailmap` to remove duplicates.
- [ ] Run [hoomd-benchmarks](https://github.com/glotzerlab/hoomd-benchmarks), check for performance
  regressions with the previous release, and post the tables in the release pull request.
- [ ] Run *bumpversion*.
- [ ] Tag and push.
- [ ] Update conda-forge recipe.
- [ ] Update glotzerlab-software.
- [ ] Announce release on mailing list.
