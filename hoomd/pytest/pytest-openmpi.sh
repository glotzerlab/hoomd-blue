#!/bin/bash

# This is a wrapper for mpirun to call 'python3 -m pytest [args]'. It does the following:
# * Only write a junit-xml file on rank 0
# * Redirect ranks > 0 output to pytest.out.rank

args=("$@")

if (( OMPI_COMM_WORLD_RANK > 0 ));
then

    for ((i=0; i<"${#args[@]}"; ++i)); do
        case ${args[i]} in
            --junit-xml*) unset args[i]; break;;
        esac
    done

    python3 -m pytest "${args[@]}" > pytest.out.${OMPI_COMM_WORLD_RANK} 2>&1
else
    python3 -m pytest "${args[@]}"
fi
