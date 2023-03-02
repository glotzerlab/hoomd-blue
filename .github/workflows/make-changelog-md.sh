#!/bin/bash

set -euo pipefail

tag=$(echo "$1" | sed  -e 's/\./\\\./g')
pcregrep -M "^${tag}.*\n(\^\^\^\^+|^---+)+.*\n(.*\n)+?(\^\^\^\^+|^---+)$" CHANGELOG.rst \
  | tail -n +3 \
  | head -n -2 \
  | pandoc --from=rst --to=markdown --wrap=none
