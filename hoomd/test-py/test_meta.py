# -*- coding: iso-8859-1 -*-
# Maintainer: csadorf

from hoomd import *
import hoomd;
context.initialize()
import unittest
import os
import tempfile

# unit tests for meta.dump_metadata
class metadata_tests(unittest.TestCase):

    def setUp(self):
        print()
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

    def tearDown(self):
        if init.is_initialized():
            context.initialize()

    def test_before_init(self):
        context.initialize()
        with self.assertRaises(RuntimeError):
            meta.dump_metadata()

    def test_after_init(self):
        meta.dump_metadata()

    def test_with_user(self):
        user = {'my_extra_field': 123}
        metadata = meta.dump_metadata(user = user)
        self.assertEqual(metadata['user']['my_extra_field'], 123)

    def test_with_file(self):
        import json, socket
        user = {'my_extra_field': 123}
        tmp = tempfile.NamedTemporaryFile()
        metadata = meta.dump_metadata(filename = tmp.name, user = user)
        self.assertEqual(metadata['user']['my_extra_field'], 123)

        if comm.get_rank() == 0:
            with tmp:
                metadata_check = json.loads(tmp.read().decode())
            self.assertEqual(len(metadata), len(metadata_check))
            self.assertEqual(len(metadata), len(metadata_check))
            self.assertEqual(metadata['user']['my_extra_field'], metadata_check['user']['my_extra_field'])
            for a, b in zip(metadata, metadata_check):
                self.assertEqual(metadata[a], metadata_check[b])

    def test_context(self):
        import json, socket
        import socket
        metadata = meta.dump_metadata()
        self.assertEqual(metadata['context']['hostname'], socket.gethostname())

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
