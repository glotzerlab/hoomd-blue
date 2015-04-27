# -*- coding: iso-8859-1 -*-
# Maintainer: mphoward

from hoomd_script import *
import unittest
import os
import tempfile

## Automatic citation generation tests
class cite_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)
        
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
        self.assertIsInstance(globals.bib, cite.bibliography)
        
        # check that it has two entries, the default HOOMD citations, and that the keys match what they should be
        self.assertEqual(len(globals.bib.entries), 2)
        self.assertIn('anderson2008', globals.bib.entries)
        self.assertIn('hoomdweb', globals.bib.entries)

        # stringify the global bibliography (basically, save the pointer)
        s1 = str(globals.bib)

        # add another element to the bibliography
        c = cite.misc(cite_key='test')
        cite._ensure_global_bib().add(c)
        
        # check that the global bibliography is unchanged
        self.assertEqual(str(globals.bib), s1)

        # wipeout the global bibliography and create it again
        globals.bib = None
        cite._ensure_global_bib()
        self.assertIsInstance(globals.bib, cite.bibliography)
    
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
    
    ## Test that bibtex file will only overwrite when it is supposed to
    def test_overwrite(self):
        # only the root rank should attempt to save
        if comm.get_rank() == 0:
            # the tmp file already exists, so we can only write if overwrite is enabled
            cite.save(file=self.tmp_file,overwrite=True,force=True)

            # okay, now try to do it with overwrite disabled and make sure we get an error
            self.assertRaises(RuntimeError, cite.save, file=self.tmp_file, overwrite=False, force=True)
        else:
            fname = 'invalid%d' % comm.get_rank()
            cite.save(file=fname,overwrite=True,force=True)
            self.assertFalse(os.path.isfile(fname))

    ## Test that the bibliography is automatically (or forcibly) generated as requested by the user
    def test_autosave(self):
        all = group.all()
        integrate.mode_standard(dt=0.005)
        integrate.nve(group=all)

        if comm.get_rank() == 0:
            # at first, nothing should be in the file
            nl = sum(1 for line in open(self.tmp_file))
            self.assertEqual(nl, 0)

        # force a save at this immediate moment
        cite.save(file=self.tmp_file, force=True)

        if comm.get_rank() == 0:
            nl1 = sum(1 for line in open(self.tmp_file))
            self.assertTrue(nl1 > nl)

        # add a citation and request a delayed save
        c = cite.misc(cite_key='test')
        cite._ensure_global_bib().add(c)
        cite.save(file=self.tmp_file)

        # this should not change the file yet
        if comm.get_rank() == 0:
            nl2 = sum(1 for line in open(self.tmp_file))
            self.assertEqual(nl1, nl2)

        # then, call run. shouldn't be able to save because a file already exists
        cite.save(file=self.tmp_file, overwrite=False)
        if comm.get_rank() == 0:
            self.assertRaises(RuntimeError, run, 1)
            nl2 = sum(1 for line in open(self.tmp_file))
            self.assertEqual(nl1, nl2)

        # when overwriting is turned on, run should work now
        cite.save(file=self.tmp_file, overwrite=True)
        run(1)
        if comm.get_rank() == 0:
            nl2 = sum(1 for line in open(self.tmp_file))
            self.assertTrue(nl2 > nl1)

        # add another record
        c2 = cite.misc(cite_key='test2')
        cite._ensure_global_bib().add(c2)
        cite.save(file=self.tmp_file, overwrite=False)
        # run again, this should add the extra citation (and because we already turned saving on,
        # it will just do it again)
        run(1)
        if comm.get_rank() == 0:
            nl3 = sum(1 for line in open(self.tmp_file))
            self.assertTrue(nl3 > nl2)

        # finally, make sure that if we switch overwriting off, it still works
        c3 = cite.misc(cite_key='test3')
        cite._ensure_global_bib().add(c3)
        # run again, this should add the extra citation (and because we already turned saving on,
        # it will just do it again)
        run(1)
        if comm.get_rank() == 0:
            nl4 = sum(1 for line in open(self.tmp_file))
            self.assertTrue(nl4 > nl3)

    def tearDown(self):
        init.reset();
        if (comm.get_rank()==0):
            os.remove(self.tmp_file);

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
