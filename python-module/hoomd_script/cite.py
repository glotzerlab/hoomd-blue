# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
# the University of Michigan All rights reserved.

# HOOMD-blue may contain modifications ("Contributions") provided, and to which
# copyright is held, by various Contributors who have granted The Regents of the
# University of Michigan the right to modify and/or distribute such Contributions.

# You may redistribute, use, and create derivate works of HOOMD-blue, in source
# and binary forms, provided you abide by the following conditions:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions, and the following disclaimer both in the code and
# prominently in any materials provided with the distribution.

# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions, and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# * All publications and presentations based on HOOMD-blue, including any reports
# or published results obtained, in whole or in part, with HOOMD-blue, will
# acknowledge its use according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http://codeblue.umich.edu/hoomd-blue/

# * Apart from the above required attributions, neither the name of the copyright
# holder nor the names of HOOMD-blue's contributors may be used to endorse or
# promote products derived from this software without specific prior written
# permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
# WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- end license --

# Maintainer: mphoward / All Developers are free to add commands for new features

from hoomd_script import util
from hoomd_script import globals
import textwrap
import os

## \package hoomd_script.cite
# \brief Commands to support automatic citation generation

## \internal
# 
class _citation(object):
    def __init__(self, cite_key, feature, author):
        self.cite_key = cite_key
        self.feature = feature
        
        # author requires special handling for I/O
        self.author = None
        if author is not None:
            if not isinstance(author, (list,tuple)):
                self.author = [author]
            else:
                self.author = author
        
        self.required_entries = []
        self.bibtex_type = None
        
        for key in _citation.standard_keys:
            setattr(self,key,None)

        # doi and url require special handling for I/O
        self.doi = None
        self.url = None
    
    def log(self):
        wrapper = textwrap.TextWrapper(initial_indent = '* ', subsequent_indent = '  ', width = 80)
        
        print '-'*5
        if self.feature is not None:
            print textwrap.fill('You are using %s. Read and cite the following:' % self.feature)
        else:
            print textwrap.fill('Read and cite the following:')
        
        out = ''
        if self.author is not None:
            out += self.format_authors(True)
            out += '. '
        
        out += self.fancy_print()
        
        out += '.'
        
        print wrapper.fill(out)
        print '-'*5
    
    def fancy_print(self):
        globals.msg.error('Bug in hoomd_script.cite: each deriving class must implement its own fancy_print\n')
        raise RuntimeError()
    
    def validate(self):
        for entry in self.required_entries:
            if getattr(self,entry) is None:
                globals.msg.error('Bug in hoomd_script.cite: required field %s not set, please report\n' % entry)
                raise RuntimeError()
        
    def format_authors(self, fancy=False):
        if self.author is None:
            return
        
        if len(self.author) > 1:
            if fancy:
                return '%s, and %s' % (', '.join(self.author[:-1]), self.author[-1])
            else:
                return ' and '.join(self.author)
        else:
            return self.author[0]
    
    def bibtex(self):
        if self.bibtex_type is None:
            globals.msg.error('Bug in hoomd_script.cite: BibTeX record type must be set, please report\n')
            raise RuntimeError()
            
        lines = ['@%s{%s,' % (self.bibtex_type, self.cite_key)]
        if self.author is not None:
            lines += ['  author = {%s},' % self.format_authors(False)]
            
        for key in _citation.standard_keys:
            val = getattr(self, key)
            if getattr(self, key) is not None:
                lines += ['  %s = {%s},' % (key, val)]
        
        # doi requires special handling
        if self.doi is not None:
            lines += ['  doi = {%s},' % self.doi, '  url = {http://dx.doi.org/%s},' % self.doi]
        
        # add note based on the feature if a note has not been set
        if self.feature is not None and self.note is None:
            lines += ['  note = {HOOMD-blue feature: %s},' % self.feature]
        
        # remove trailing comma
        lines[-1] = lines[-1][:-1]
        
        #close off record
        lines += ['}']
        
        return '\n'.join(lines)
_citation.standard_keys = ['address','annote','booktitle','chapter','crossref','edition','editor','howpublished',
                           'institution', 'journal','key','month','note','number','organization','pages','publisher',
                           'school','series','title', 'type','volume','year']


## Article BibTeX entry
# 
class article(_citation):
    def __init__(self, cite_key, author, title, journal, year, volume, number=None, pages=None, month=None, note=None, key=None, doi=None, feature=None, display=True):
        _citation.__init__(self, cite_key, feature, author)
        
        self.required_entries = ['author', 'title', 'journal', 'year', 'volume', 'pages']
        self.bibtex_type = 'article'
        
        self.title = title
        self.journal = journal
        self.year = year
        self.volume = volume
        self.number = number
        self.pages = pages
        self.month = month
        self.note = note
        self.key = key
        self.doi = doi
        
        # attach to the global bibliography
        _ensure_global_bib().add(self)
        
        if display:
            self.log()
    
    def fancy_print(self):
        return '"%s", %s %s (%s) %s' % (self.title, self.journal, str(self.volume), str(self.year), str(self.pages))

## Miscellaneous BibTeX entry
#
class misc(_citation):
    def __init__(self, cite_key, author=None, title=None, howpublished=None, year=None, month=None, note=None, key=None, feature=None, display=True):
        _citation.__init__(self, cite_key, feature, author)
        self.required_entries = []
        self.bibtex_type = 'misc'
        
        self.title = title
        self.howpublished = howpublished
        self.year = year
        self.month = month
        self.note = note
        self.key = key
        
        if display:
            self.log()
        
    def fancy_print(self):
        out = ''
        if self.title is not None:
            out += '"%s", ' % self.title
        if self.howpublished is not None:
            out += self.howpublished
        if len(out) > 0:
            out += ' (%s)' % str(year)
        return out

## Bibliography
#
class bibliography(object):
    def __init__(self):
        self.entries = {}
    
    def add(self, entry):
        entry.validate()
        
        self.entries[entry.cite_key] = entry
    
    def save(self, file):
        if len(self.entries) == 0:
            globals.msg.warning('Empty bibliography generated, not saving anything to file.\n')
            return
            
        if os.path.isfile(file):
            globals.msg.warning('Bibliography file %s already exists, overwriting.\n' % file)
        
        f = open(file, 'w')
        f.write('% This BibTeX file was automatically generated from HOOMD-blue\n')
        f.write('% Encoding: UTF-8\n\n')
        for entry in self.entries:
            f.write(self.entries[entry].bibtex() + '\n\n')
        f.close()

## \internal
# \brief Ensures that the global bibliography is properly initialized
# \returns Global bibliography
def _ensure_global_bib():
    if globals.bib is None:
        globals.bib = bibliography()
    
        # the hoomd bibliography always includes the following citations
        hoomd = article(cite_key = 'anderson2008',
                        author = ['J A Anderson','C D Lorenz','A Travesset'],
                        title = 'General purpose molecular dynamics simulations fully implemented on graphics processing units',
                        journal = 'Journal of Computational Physics',
                        volume = 227,
                        number = 10,
                        pages = '5342--5359',
                        year = 2008,
                        month = 'may',
                        doi = '10.1016/j.jcp.2008.01.047',
                        display = False)
        hoomd_web = misc(cite_key = 'hoomdweb', howpublished = 'http://codeblue.umich.edu/hoomd-blue', display = False)
                    
        globals.bib.add(hoomd)
        globals.bib.add(hoomd_web)
    return globals.bib

## Wrapper to write global bibliography to file
#
def save_bib(file='hoomd.bib'):
    util.print_status_line()
    
    if globals.bib is not None:
        globals.bib.save(file)
    else:
        globals.msg.warning('No bibliography generated, not saving anything to file.\n')
    
