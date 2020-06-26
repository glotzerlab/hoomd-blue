# -*- coding: iso-8859-1 -*-
# Maintainer: mphoward

from hoomd import *
import hoomd
from hoomd import md
context.initialize();
import unittest
import os
import tempfile

## Automatic citation generation tests
class cite_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

        context.current.sorter.set_params(grid=8)

        # ensure that the bibliography is there
        cite._ensure_global_bib()

        # tmp file
        if comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.bib');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

    ## Test global bibliography exists and gets filled correctly with default values
    def test_global_bib(self):
        # global bibliography should exist after initialization
        self.assertIsInstance(hoomd.context.bib, cite.bibliography)

        # check that it has two entries, the default HOOMD citations, and that the keys match what they should be
        print(hoomd.context.bib.entries)
        self.assertEqual(len(hoomd.context.bib.entries), 1)
        self.assertIn('Anderson2020', hoomd.context.bib.entries)

        # stringify the global bibliography (basically, save the pointer)
        s1 = str(hoomd.context.bib)

        # add another element to the bibliography
        c = cite.misc(cite_key='test')
        cite._ensure_global_bib().add(c)

        # check that the global bibliography is unchanged
        self.assertEqual(str(hoomd.context.bib), s1)

        # wipeout the global bibliography and create it again
        hoomd.context.bib = None
        cite._ensure_global_bib()
        self.assertIsInstance(hoomd.context.bib, cite.bibliography)

    ## Test groups of authors get names formatted properly
    def test_format_authors(self):
        c = cite.misc(cite_key='test')

        # no author specified should return none
        self.assertEqual(c.format_authors(), None)

        # single author is just his name
        c = cite.misc(cite_key='test', author='M P Howard')
        self.assertEqual(c.format_authors(), 'M P Howard')

        # two authors looks the same whether its pretty or bibtex
        c = cite.misc(cite_key='test', author=['M P Howard','Joshua A. Anderson'])
        self.assertEqual(c.format_authors(), 'M P Howard and Joshua A. Anderson')
        self.assertEqual(c.format_authors(fancy=True), 'M P Howard and Joshua A. Anderson')

        # three authors gets the oxford comma in fancy format
        c = cite.misc(cite_key='test', author=('M P Howard', 'Joshua A. Anderson', 'Mickey Mouse'))
        self.assertEqual(c.format_authors(), 'M P Howard and Joshua A. Anderson and Mickey Mouse')
        self.assertEqual(c.format_authors(fancy=True), 'M P Howard, Joshua A. Anderson, and Mickey Mouse')

    ## Test display values are properly toggled on and off
    def test_display(self):
        c = cite.misc(cite_key='test')

        # there is nothing in this string, so should return none
        self.assertEqual(c.log(), None)

        # if display is off, it should also return none
        c = cite.misc(cite_key='test', author='M P Howard', display=False)
        self.assertEqual(c.log(), None)

        # if display is on, then we should get something back
        c = cite.misc(cite_key='test', author='M P Howard')
        self.assertNotEqual(c.log(), None)
        self.assertNotEqual(len(c.log()), 0)

    ## Test that an error is thrown when a citation is missing a key field
    def test_validate(self):
        c = cite.misc(cite_key='test')
        # override the required entries to force an error
        c.required_entries = ['title']
        self.assertRaises(RuntimeError, c.validate)

    ## Test that the bibliography is automatically (or forcibly) generated as requested by the user
    def test_autosave(self):
        all = group.all()
        md.integrate.mode_standard(dt=0.005)
        md.integrate.nve(group=all)

        if comm.get_rank() == 0:
            # at first, nothing should be in the file
            nl = sum(1 for line in open(self.tmp_file))
            self.assertEqual(nl, 0)

        # force a save at this immediate moment
        cite.save(file=self.tmp_file)

        if comm.get_rank() == 0:
            nl1 = sum(1 for line in open(self.tmp_file))
            self.assertTrue(nl1 > nl)

        # add a citation and file should be resaved
        c = cite.misc(cite_key='test')
        cite._ensure_global_bib().add(c)
        if comm.get_rank() == 0:
            nl2 = sum(1 for line in open(self.tmp_file))
            self.assertTrue(nl2 > nl1)

    def tearDown(self):
        context.initialize();
        hoomd.context.bib = None;
        if (comm.get_rank()==0):
            os.remove(self.tmp_file);

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
