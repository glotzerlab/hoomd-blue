"""This file defines the core tests for metadata generation."""
import unittest
import numpy as np

import hoomd
from hoomd import md


def requires_run():
    def skip_if_not_runnable(func):
        def wrapper(self, *args, **kwargs):
            if getattr(self, 'runnable', False):
                self.skipTest()
            else:
                func(self, *args, **kwargs)
        return wrapper
    return skip_if_not_runnable

def runnable(cls):
    """Decorate a metadata testing class to indicate that we should test a
    hoomd.run call.

    Used to differentiate from tests that test the metadata of a single class
    but do not provide enough logic for HOOMD to successfully run.
    """
    setattr(cls, 'runnable', True)
    return cls

# Parent class defining setup for metadata testing.
class SetupTestMetadata(unittest.TestCase):

    def make_system(self):
        return hoomd.init.create_lattice(hoomd.lattice.sc(a=10), n=[5, 5, 5])

    def setUp(self):
        hoomd.context.initialize()
        self.system = self.make_system()

# Parent class defining setup for all metadata tests. This class is separated
# from the SetupTestMetadata class because we want to explicitly override the
# setUp (and potentially other) method of unittest.TestCase without creating a
# runnable instance. Combining these two into one class that inherits from
# TestCase will make the unittest library try to run it, while making a
# combined class that does not inherit from TestCase leads to MRO issues with
# calling setUp.
class TestMetadata:
    def test_object_dump(self):
        obj = self.build_object()
        original_metadata = hoomd.meta.dump_metadata('metadata.json', fields=['objects', 'modules'])
        old_context = hoomd.context.current
        with hoomd.context.SimulationContext() as new_context:
            system = self.make_system()
            objects = hoomd.meta.load_metadata(system, filename='metadata.json')
            new_metadata = hoomd.meta.dump_metadata('metadata.json', fields=['objects', 'modules'])
            self.assertEqual(new_metadata, original_metadata)

    # @requires_run()
    # @unittest.skip
    # def test_run(self):
        # pass

class Test(TestMetadata, SetupTestMetadata):
    """Note """
    def build_object(self):
        return md.nlist.tree()

if __name__ == "__main__":
    unittest.main()
