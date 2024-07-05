#!/usr/bin/env python3
# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Evaluate the workflow jinja templates."""

import jinja2
import yaml
import os

if __name__ == '__main__':
    # change to the directory of the script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'),
                             block_start_string='<%',
                             block_end_string='%>',
                             variable_start_string='<<',
                             variable_end_string='>>',
                             comment_start_string='<#',
                             comment_end_string='#>',
                             trim_blocks=True,
                             lstrip_blocks=True)
    template = env.get_template('test.yml')
    with open('templates/configurations.yml', 'r') as f:
        configurations = yaml.safe_load(f)

    # preprocess configurations and fill out additional fields needed by
    # `test.yml` to configure the matrix jobs
    for name, configuration in configurations.items():
        for entry in configuration:
            if entry['config'].startswith('[cuda'):
                entry['build_runner'] = "[self-hosted,jetstream2,CPU]"

                entry['test_runner'] = "[self-hosted,GPU]"
                # device options needed to access the GPU devices on the runners
                # because the nvidia container toolkit is built without cgroups
                # support:
                # https://aur.archlinux.org/packages/nvidia-container-toolkit
                entry['test_docker_options'] = \
                    "--gpus=all --device /dev/nvidia0 " \
                    "--device /dev/nvidia1 " \
                    "--device /dev/nvidia-uvm " \
                    "--device /dev/nvidia-uvm-tools " \
                    "--device /dev/nvidiactl"
            else:
                entry['test_runner'] = "ubuntu-latest"
                entry['build_runner'] = "[self-hosted,jetstream2,CPU]"
                entry['test_docker_options'] = ""

    with open('test.yml', 'w') as f:
        f.write(template.render(configurations))

    template = env.get_template('release.yml')
    with open('release.yml', 'w') as f:
        f.write(template.render())
