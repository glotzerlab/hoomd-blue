#!/bin/bash

# Execute this script to update all lock files to the latest versions of dependencies.

rm *-conda-lock.yml

for python_version in 3.9 3.10 3.11 3.12
do
  sed "s/python==.*/python=$python_version/g" environment.yaml > tmp.yaml|| exit 2
  conda lock -f tmp.yaml -p osx-arm64 -p osx-64 -p linux-64 --lockfile py${python_version//.}-conda-lock.yml || exit 2
done
rm tmp.yaml
