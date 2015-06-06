# -*- coding: iso-8859-1 -*-
# Maintainer: csadorf

from hoomd_script import *
import unittest
import os
import tempfile

# unit tests for meta.dump_metadata
class metadata_tests(unittest.TestCase):

    def setUp(self):
        print()
        init.create_random(N = 100, phi_p = 0.05)

    def tearDown(self):
        if init.is_initialized():
            init.reset()

    def test_before_init(self):
        init.reset()
        with self.assertRaises(RuntimeError):
            meta.dump_metadata()

    def test_after_init(self):
        meta.dump_metadata()

    def test_with_extra(self):
        extra = {'my_extra_field': 123}
        metadata = meta.dump_metadata(extra = extra)
        self.assertEqual(metadata[0]['my_extra_field'], 123)

    def test_with_file(self):
        import json, socket
        extra = {'my_extra_field': 123}
        tmp = tempfile.NamedTemporaryFile()
        metadata = meta.dump_metadata(filename = tmp.name, overwrite = False, extra = extra)
        self.assertEqual(metadata[0]['my_extra_field'], 123)
        with tmp:
            metadata_check = json.loads(tmp.read().decode())
        self.assertEqual(len(metadata), len(metadata_check))
        self.assertEqual(len(metadata[0]), len(metadata_check[0]))
        self.assertEqual(metadata[0]['my_extra_field'], metadata_check[0]['my_extra_field'])
        for a,b in zip(metadata, metadata_check):
            for key in a.keys():
                self.assertEqual(a[key], b[key])

    def test_context(self):
        import json, socket
        import socket
        metadata = meta.dump_metadata()
        self.assertEqual(metadata[0]['context']['hostname'], socket.gethostname())

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
