"""This file defines the core tests for metadata generation."""
import unittest
import tempfile
import json
import warnings

import hoomd
from hoomd import md


def requires_run(func):
    def wrapper(self, *args, **kwargs):
        if self.runnable:
            func(self, *args, **kwargs)
        else:
            self.skipTest(
                "This test case contains an incomplete simulation "
                "configuration and cannot be run.")
    return wrapper


def runnable(cls):
    """Decorate a metadata testing class to indicate that we should test a
    hoomd.run call.

    Used to differentiate from tests that test the metadata of a single class
    but do not provide enough logic for HOOMD to successfully run.
    """
    cls.runnable = True
    return cls


def unsupported(cls):
    """Decorate a metadata testing class to indicate that it does not support
    metadata; if it doesn't, it should warn.
    """
    cls.supported = False
    return cls


class TestMetadata(object):
    """Parent class defining setup for all metadata tests. This class is separated
    from the SetupTestMetadata class because we want to explicitly override the
    setUp (and potentially other) method of unittest.TestCase without creating
    a runnable instance. Combining these two into one class that inherits from
    TestCase will make the unittest library try to run it, while making a
    combined class that does not inherit from TestCase leads to MRO issues with
    calling setUp.
    """
    # By default we assume tests are supported and not runnable unless
    # overridden by the decorators provided above.
    supported = True
    runnable = False

    def make_system(self):
        return hoomd.init.create_lattice(hoomd.lattice.sc(a=10), n=[5, 5, 5])

    def test_dump_redump(self):
        """Check that an object's metadata is identical when reloaded and dumped again."""
        hoomd.context.initialize('--notice-level=0')
        self.system = self.make_system()
        obj = self.build_object()
        tf = tempfile.NamedTemporaryFile()

        if not self.supported:
            # Unsupported features must raise a MetadataUnsupportedWarning.
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                hoomd.meta.dump_metadata(tf.name, fields=['objects', 'modules'])
                self.assertTrue(len(w) == 1)
                assert issubclass(w[0].category, hoomd.meta.MetadataUnsupportedWarning)
        else:
            original_metadata = hoomd.meta.dump_metadata(tf.name, fields=['objects', 'modules'])
            tf.seek(0)
            original_metadata_file = json.load(tf)
            old_context = hoomd.context.current
            with hoomd.context.SimulationContext() as new_context:
                system = self.make_system()
                objects = hoomd.meta.load_metadata(system, filename=tf.name)
                new_metadata = hoomd.meta.dump_metadata(tf.name, fields=['objects', 'modules'])
                tf.seek(0)
                new_metadata_file = json.load(tf)
                self.assertEqual(new_metadata, original_metadata)
                self.assertEqual(new_metadata_file, original_metadata_file)

    def test_dump_reload(self):
        """Check that an object is equal to the version created when metadata is reloaded."""
        ###TODO: Requires implementing __eq__ for classes
        pass

    @requires_run
    def test_run(self):
        """Test whether a hoomd.run succeeds and produces identical output."""
        pass
        # obj = self.build_object()
        # tf = tempfile.NamedTemporaryFile()
        # original_metadata = hoomd.meta.dump_metadata(tf.name, fields=['objects', 'modules'])
        # tf.seek(0)
        # original_metadata_file = json.load(tf)
        # old_context = hoomd.context.current
        # with hoomd.context.SimulationContext() as new_context:
            # system = self.make_system()
            # objects = hoomd.meta.load_metadata(system, filename=tf.name)
            # new_metadata = hoomd.meta.dump_metadata(tf.name, fields=['objects', 'modules'])
            # tf.seek(0)
            # new_metadata_file = json.load(tf)
            # self.assertEqual(new_metadata, original_metadata)
            # self.assertEqual(new_metadata_file, original_metadata_file)


@unsupported
class TestAnalyzeCallback(TestMetadata, unittest.TestCase):
    """Tests hoomd.analyze.callback."""
    def build_object(self):
        return hoomd.analyze.callback(
            lambda ts: ts, period=10)


class TestAnalyzeIMD(TestMetadata, unittest.TestCase):
    """Tests hoomd.analyze.imd."""
    def build_object(self):
        return hoomd.analyze.imd(
            8888, period=10, rate=10, pause=True)


class TestWithFile(unittest.TestCase):
    """Parent for tests that require an additional temporary file."""
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile()

    def tearDown(self):
        # Should happen automatically, but be safe and explicitly close the file.
        del self.file


class TestAnalyzeLog(TestMetadata, TestWithFile):
    """Tests hoomd.analyze.log."""
    def build_object(self):
        return hoomd.analyze.log(
             filename=self.file.name, quantities=['potential_energy', 'temperature'],
             period=100,
             overwrite=True)


class TestComputeThermo(TestMetadata, unittest.TestCase):
    """Tests hoomd.compute.thermo."""
    def build_object(self):
        return hoomd.compute.thermo(
            hoomd.group.rigid())


class TestDumpDcd(TestMetadata, TestWithFile):
    """Tests hoomd.dump.dcd."""
    def build_object(self):
        return hoomd.dump.dcd(
            filename=self.file.name, period=10, group=hoomd.group.all(),
            overwrite=True, phase=10, unwrap_full=True)


class TestDumpGetar(TestMetadata, TestWithFile):
    """Tests hoomd.dump.getar."""
    def build_object(self):
        return hoomd.dump.getar(
            filename=self.file.name, static=['viz_dynamic', 'bond_all'],
            dynamic={'orientation': 10000,
                     'velocity': 5000})


class TestDumpGsd(TestMetadata, TestWithFile):
    """Tests hoomd.dump.gsd."""
    def build_object(self):
        return hoomd.dump.gsd(
            filename=self.file.name, period=10, group=hoomd.group.all(),
            overwrite=True, truncate=True, phase=10,
            dynamic=['momentum', 'topology'])


class TestVariantLinear(TestMetadata, unittest.TestCase):
    """Tests hoomd.variant.linear_interp."""

    def build_object(self):
        self.maxDiff=None
        return hoomd.variant.linear_interp(points = [[0, 1.0], [1e5, 2.0]])


@unsupported
class TestHDF5Log(TestMetadata, TestWithFile):
    """Tests hoomd.hdf5.log."""
    # hoomd.hdf5 is not automatically imported.
    import hoomd.hdf5
    def build_object(self):
        return hoomd.hdf5.log(hoomd.hdf5.File(self.file.name, 'w'), 10)



if __name__ == "__main__":
    unittest.main()
