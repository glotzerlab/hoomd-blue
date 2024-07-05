# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Install pybind11, cereal, and eigen."""

import os
import sys
import tempfile
import pathlib
import subprocess
import copy
import logging
import argparse
import urllib.request
import tarfile

log = logging.getLogger(__name__)


def find_cmake_package(name,
                       version,
                       location_variable=None,
                       ignore_system=False):
    """Find a package with cmake.

    Return True if the package is found
    """
    if location_variable is None:
        location_variable = name + "_DIR"

    find_package_options = ''
    if ignore_system:
        find_package_options += 'NO_SYSTEM_ENVIRONMENT_PATH ' \
            'NO_CMAKE_PACKAGE_REGISTRY NO_CMAKE_SYSTEM_PATH ' \
            'NO_CMAKE_SYSTEM_PACKAGE_REGISTRY'

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = pathlib.Path(tmpdirname)

        # write the cmakelists file
        with open(tmp_path / 'CMakeLists.txt', 'w') as f:
            f.write(f"""
project(test)
set(PYBIND11_PYTHON_VERSION 3)
cmake_minimum_required(VERSION 3.9)
find_package({name} {version} CONFIG REQUIRED {find_package_options})
""")

        # add the python prefix to the cmake prefix path
        env = copy.copy(os.environ)
        env['CMAKE_PREFIX_PATH'] = sys.prefix

        os.mkdir(tmp_path / 'build')
        cmake_out = subprocess.run(['cmake', tmpdirname],
                                   cwd=tmp_path / 'build',
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   timeout=120,
                                   env=env,
                                   encoding='UTF-8')

        log.debug(cmake_out.stdout.strip())

        # if cmake completed correctly, the package was found
        if cmake_out.returncode == 0:
            location = ''
            with open(tmp_path / 'build' / 'CMakeCache.txt', 'r') as f:
                for line in f.readlines():
                    if line.startswith(location_variable):
                        location = line.strip()

            log.info(f"Found {name}: {location}")
            return True
        else:
            log.debug(cmake_out.stdout.strip())
            return False


def install_cmake_package(url, cmake_options):
    """Install a cmake package."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = pathlib.Path(tmpdirname)

        log.info(f"Fetching {url}")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(tmp_path / 'file.tar.gz', 'wb') as f:
            f.write(urllib.request.urlopen(req).read())

        with tarfile.open(tmp_path / 'file.tar.gz') as tar:

            def is_within_directory(directory, target):

                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)

                prefix = os.path.commonprefix([abs_directory, abs_target])

                return prefix == abs_directory

            def safe_extract(tar,
                             path=".",
                             members=None,
                             *,
                             numeric_owner=False):

                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tar, path=tmp_path)
            root = tar.getnames()[0]
            if '/' in root:
                root = os.path.dirname(root)

        # add the python prefix to the cmake prefix path
        env = copy.copy(os.environ)
        env['CMAKE_PREFIX_PATH'] = sys.prefix

        log.info(f"Configuring {root}")
        os.mkdir(tmp_path / 'build')
        cmake_out = subprocess.run(
            ['cmake', tmp_path / root, f'-DCMAKE_INSTALL_PREFIX={sys.prefix}']
            + cmake_options,
            cwd=tmp_path / 'build',
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            env=env,
            encoding='UTF-8')

        log.debug(cmake_out.stdout.strip())

        if cmake_out.returncode != 0:
            log.error(f"Error configuring {root} (run with -v to see detailed "
                      "error messages)")
            raise RuntimeError('Failed to configure package')

        log.info(f"Installing {root}")
        cmake_out = subprocess.run(
            ['cmake', '--build', tmp_path / 'build', '--', 'install'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            env=env,
            encoding='UTF-8')

        log.debug(cmake_out.stdout.strip())

        if cmake_out.returncode != 0:
            log.error(f"Error installing {root} (run with -v to see detailed "
                      "error messages)")
            raise RuntimeError('Failed to install package')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Install header-only libraries needed to build HOOMD-blue.')
    parser.add_argument('-q',
                        action='store_true',
                        default=False,
                        help='Suppress info messages.')
    parser.add_argument('-v',
                        action='store_true',
                        default=False,
                        help='Show debug messages (overrides -q).')
    parser.add_argument('-y',
                        action='store_true',
                        default=False,
                        help='Skip user input and force installation.')
    parser.add_argument('--ignore-system',
                        action='store_true',
                        default=False,
                        help='Ignore packages installed at the system level.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.q:
        log.setLevel(level=logging.WARNING)
    if args.v:
        log.setLevel(level=logging.DEBUG)

    log.info(f"Searching for packages in: {sys.prefix}")

    pybind = find_cmake_package('pybind11',
                                '2.0',
                                ignore_system=args.ignore_system)
    cereal = find_cmake_package('cereal', '', ignore_system=args.ignore_system)
    eigen = find_cmake_package('Eigen3',
                               '3.2',
                               ignore_system=args.ignore_system)

    all_found = all([pybind, cereal, eigen])

    if all_found:
        log.info("Done. Found all packages.")
    else:
        missing_packages = ''
        if not pybind:
            missing_packages += 'pybind11, '
        if not cereal:
            missing_packages += 'cereal, '
        if not eigen:
            missing_packages += 'Eigen, '
        missing_packages = missing_packages[:-2]

        if args.y:
            proceed = 'y'
        else:
            print(f"*** About to install {missing_packages} into {sys.prefix}")
            proceed = input('Proceed (y/n)? ')

        if proceed == 'y':
            log.info(f"Installing packages in: {sys.prefix}")

            if not pybind:
                install_cmake_package(
                    'https://github.com/pybind/pybind11/archive/v2.10.1.tar.gz',
                    cmake_options=[
                        '-DPYBIND11_INSTALL=on', '-DPYBIND11_TEST=off'
                    ])

            if not cereal:
                install_cmake_package(
                    'https://github.com/USCiLab/cereal/archive/v1.3.2.tar.gz',
                    cmake_options=['-DJUST_INSTALL_CEREAL=on'])

            if not eigen:
                install_cmake_package(
                    'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/'
                    'eigen-3.4.0.tar.gz',
                    cmake_options=[
                        '-DBUILD_TESTING=off', '-DEIGEN_TEST_NOQT=on'
                    ])
            log.info('Done.')
        else:
            print('Cancelled')
