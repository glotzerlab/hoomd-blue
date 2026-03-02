---
name: Release checklist
about: '[for maintainer use]'
title: 'Release 6.1.1'
labels: ''

---

To make a new HOOMD-blue release:

- [ ] Make a new `release-{X.Y.Z}` branch (where `{X.Y.Z}` is the new version).

On that branch, take the following steps (committing after each step when needed):

- [ ] Run `prek autoupdate`
- [ ] Check for new or duplicate contributors since the last release:
  `comm -13 (git log $(git describe --tags --abbrev=0) --format="%aN <%aE>" | sort | uniq | psub) (git log --format="%aN <%aE>" | sort | uniq | psub)`.
  Add entries to `.mailmap` to remove duplicates.
- [ ] Update tutorial submodule.
- [ ] Ensure that added features have Sybil examples.
- [ ] Review `CHANGELOG.rst` and revise if needed.
- [ ] Run `bump-my-version bump {type}`. Replace `{type}` with:
  - `patch` when this release *only* includes bug fixes.
  - `minor` when this release includes new features and possibly bug fixes.
  - `major` when this release includes API breaking changes.
- [ ] Push the branch and open a pull request.
- [ ] Check that readthedocs builds the docs correctly in the pull request checks.
- [ ] Merge the pull request after all tests pass.
- [ ] Make a new tag on the trunk branch:
  ```
  git switch trunk
  git pull
  git tag -a v{X.Y.Z}
  git push origin --tags
  ```

> [!IMPORTANT]
> Make sure to **include** `v` in the tag name!

- [ ] Add a blank release notes entry for the next release:
  ```
  Next release
  ^^^^^^^^^^^^^^^^^^^^

  *Added*

  *Changed*

  *Deprecated*

  *Removed*

  *Fixed*
  ```

> [!NOTE]
> Paste `Next release` exactly as shown. `bump-my-version` will replace that
> string with the version number and date.

GitHub Actions will trigger on the tag and create a GitHub release. After a few
hours, the conda-forge autotick bot will submit a PR for the new release.

- [ ] Check that the GitHub release posted correctly: https://github.com/glotzerlab/hoomd-blue/releases
- [ ] Merge the conda-forge recipe, updating it first if necessary (e.g. when adding dependencies).
- [ ] Update *glotzerlab-software*. Build the new version on:
  - [ ] Great Lakes
  - [ ] Anvil
  - [ ] Delta
  - [ ] Frontier
  - [ ] Andes

For major and minor releases:

- [ ] Test *hoomd-component-template* against the new version.
- Sync those changes with:
  - [ ] *hoomd-md-pair-template*.
  - [ ] *hoomd-hpmc-shape-template*.
  - [ ] *hpmc-energy-template*.
- [ ] Run [hoomd-benchmarks](https://github.com/glotzerlab/hoomd-benchmarks), check for performance
  regressions with the previous release, and post the tables in the release pull request.
- [ ] Run [hoomd-validation](https://github.com/glotzerlab/hoomd-validation).
