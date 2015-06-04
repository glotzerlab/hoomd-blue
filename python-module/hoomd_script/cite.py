# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
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
from hoomd_script import comm
import textwrap
import os

## \package hoomd_script.cite
# \brief Commands to support automatic citation generation
#
# Certain features of HOOMD-blue require citation because they represent significant contributions from developers.
# In order to make these citations transparent and easy, HOOMD-blue will automatically print citation notices at run
# time when you use a feature that should be cited. Users should read and cite these publications in their work.
# Citations can be saved as a BibTeX file for easy incorporation into bibliographies. The bibliography is generated
# in accordance with the terms laid out at \ref page_citing.
#

## \internal
# \brief Generic citation object
# Objects representing specific BibTeX records derive from this object, which stores relevant citation information,
# and also supplies common methods for printing and saving these methods.
#
# Each deriving BibTeX record must supply the following:
# 1. __str__ to print the record in a human readable format
# 2. the string name for the BibTeX record
#
# Optionally, deriving objects may supply a list of BibTeX keys that are required. The record will then be validated
# for the existence of these keys when the record is added to a bibliography.
class _citation(object):
    ## \internal
    # \brief Constructs a citation
    # \param cite_key Unique key identifying this reference
    # \param feature HOOMD functionality this citation corresponds to
    # \param author Author or list of authors for this citation
    # \param display Flag to automatically print the citation when adding to a bibliography
    def __init__(self, cite_key, feature=None, author=None, display=False):
        self.cite_key = cite_key
        self.feature = feature
        self.display = display
        
        # author requires special handling for I/O
        self.author = None
        if author is not None:
            if not isinstance(author, (list,tuple)):
                self.author = [author]
            else:
                self.author = author
        
        self.required_entries = []
        self.bibtex_type = None
        
        # initialize all standard keys as member variables
        for key in _citation.standard_keys:
            setattr(self,key,None)

        # doi and url require special handling for I/O
        self.doi = None
        self.url = None
    
    ## \var cite_key
    # \internal
    # \brief Unique key identifying this reference

    ## \var feature
    # \internal
    # \brief String representation of the HOOMD functionality

    ## \var author
    # \internal
    # \brief List of authors

    ## \var required_entries
    # \internal
    # \brief List of BibTeX values that \b must be set in the record

    ## \var bibtex_type
    # \internal
    # \brief String for BibTeX record type

    ## \var doi
    # \internal
    # \brief The DOI for this citation

    ## \var url
    # \internal
    # \brief The URL for this citation (if web reference)

    ## \internal
    # \brief Prints the citation as a human readable notice
    def log(self):
        if self.display is False:
            return None

        out = str(self)
        if len(out) == 0:
            return None # quit if there is nothing to actually log in this citation

        wrapper = textwrap.TextWrapper(initial_indent = '* ', subsequent_indent = '  ', width = 80)
        return wrapper.fill(out) + '\n'
    
    ## \internal
    # \brief Get the citation in human readable format
    # \note Deriving classes \b must implement this method themselves.
    def __str__(self):
        globals.msg.error('Bug in hoomd_script.cite: each deriving class must implement its own string method\n')
        raise RuntimeError('Citation does not implement string method')
    
    ## \internal
    # \brief Ensures that all required fields have been set
    def validate(self):
        for entry in self.required_entries:
            if getattr(self,entry) is None:
                globals.msg.error('Bug in hoomd_script.cite: required field %s not set, please report\n' % entry)
                raise RuntimeError('Required citation field not set')

    ## \internal
    # \brief Formats the author name list
    # \param fancy Flag to print as a human readable list
    # \returns Author list as a string or None if there are no authors
    def format_authors(self, fancy=False):
        if self.author is None:
            return None
        
        if len(self.author) > 1:
            if not fancy:
                return ' and '.join(self.author)
            elif len(self.author) > 2:
                return '%s, and %s' % (', '.join(self.author[:-1]), self.author[-1])
            else:
                return '%s and %s' % tuple(self.author)
        else:
            return self.author[0]
    
    ## \internal
    # \brief Converts the citation to a BibTeX record
    # \returns BibTeX record as a string
    #
    # If a DOI is set for the citation and no URL record has been specified, the URL is specified from the DOI.
    # If no note is set for the citation, a default note identifying the HOOMD feature used is generated.
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
        
        # doi requires special handling because it is not "standard"
        if self.doi is not None:
            lines += ['  doi = {%s},' % self.doi]
        # only override the url with the doi if it is not set
        if self.url is None and self.doi is not None:
            lines += ['  url = {http://dx.doi.org/%s},' % self.doi]
        elif self.url is not None:
            lines += ['  url = {%s},' % self.url]
        
        # add note based on the feature if a note has not been set
        if self.feature is not None and self.note is None:
            lines += ['  note = {HOOMD-blue feature: %s},' % self.feature]
        
        # remove trailing comma
        lines[-1] = lines[-1][:-1]
        
        # close off record
        lines += ['}']
        
        return '\n'.join(lines)
## \internal
# List of standard BibTeX keys that citations may use
_citation.standard_keys = ['address','annote','booktitle','chapter','crossref','edition','editor','howpublished',
                           'institution', 'journal','key','month','note','number','organization','pages','publisher',
                           'school','series','title', 'type','volume','year']


## \internal
# \brief Article BibTeX entry
class article(_citation):
    ## \internal
    # \brief Creates an article entry
    #
    # \param cite_key Unique key identifying this reference
    # \param author Author or list of authors for this citation
    # \param title Article title
    # \param journal Journal name (full or abbreviated)
    # \param year Year of publication
    # \param volume Journal volume
    # \param pages Page range or article id
    # \param number Journal issue number
    # \param month Month of publication
    # \param note Optional note on the article
    # \param key Optional key
    # \param doi Digital object identifier
    # \param feature Name of HOOMD feature corresponding to citation
    # \param display Flag to automatically print the citation when adding to a bibliography
    def __init__(self, cite_key, author, title, journal, year, volume, pages, number=None, month=None, note=None, key=None, doi=None, feature=None, display=True):
        _citation.__init__(self, cite_key, feature, author, display)
        
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

    def __str__(self):
        out = ''
        if self.author is not None:
            out += self.format_authors(True)
            out += '. '
        out += '"%s", %s %s (%s) %s' % (self.title, self.journal, str(self.volume), str(self.year), str(self.pages))
        return out

## \internal
# \brief Miscellaneous BibTeX entry
class misc(_citation):
    ## \internal
    # \brief Creates a miscellaneous entry
    #
    # \param cite_key Unique key identifying this reference
    # \param author Author or list of authors for this citation
    # \param title Article title
    # \param howpublished Format of publication (<i>e.g.</i> a url for a website)
    # \param year Year of publication
    # \param month Month of publication
    # \param note Optional note on the article
    # \param key Optional key
    # \param feature Name of HOOMD feature corresponding to citation
    # \param display Flag to automatically print the citation when adding to a bibliography
    def __init__(self, cite_key, author=None, title=None, howpublished=None, year=None, month=None, note=None, key=None, feature=None, display=True):
        _citation.__init__(self, cite_key, feature, author, display)
        self.required_entries = []
        self.bibtex_type = 'misc'
        
        self.title = title
        self.howpublished = howpublished
        self.year = year
        self.month = month
        self.note = note
        self.key = key

    def __str__(self):
        out = ''
        if self.author is not None:
            out += self.format_authors(True)
        if self.title is not None:
            if len(out) > 0:
                out += '. '
            out += '"%s"' % self.title
        if self.howpublished is not None:
            if len(out) > 0:
                out += ', '
            out += self.howpublished
        if self.year is not None:
            if len(out) > 0:
                out += ' '
            out += '(%s)' % str(year)
        return out

## \internal
# \brief Collection of citations
#
# A %bibliography is a dictionary of citations. It ensures that each citation exists only once for a given key,
# and provides a mechanism for generating a BibTeX file. If two citations attempt to use the same citation key, only
# the most recent citation object is used for that entry.
class bibliography(object):
    ## \internal
    # \brief Creates a bibliography
    def __init__(self):
        self.entries = {}
        self.autosave = False
        self.updated = False
        self.file = 'hoomd.bib'
    ## \var entries
    # \internal
    # \brief Dictionary of citations

    ## \var autosave
    # \internal
    # \brief Flag marking if the bibliography should be saved to file on run time

    ## \var updated
    # \internal
    # \brief Flag marking if the bibliography has been updated since the last save

    ## \var file
    # \internal
    # \brief File name to use to save the bibliography

    ## \internal
    # \brief Adds a citation, ensuring each key is only saved once
    # \param entry Citation or list of citations to add to the bibliography
    def add(self, entry):
        # wrap the entry as a list if it is not
        if not isinstance(entry, (list,tuple)):
            entry = [entry]

        # parse unique sets of features out of attached citations
        citations = {}
        for e in entry:
            e.validate()
            self.entries[e.cite_key] = e

            log_str = e.log()
            if log_str is not None: # if display is enabled and log returns an output
                # if a feature is specified, we try to group them together into logical sets
                if e.feature is not None:
                    if e.feature not in citations:
                        citations[e.feature] = [log_str]
                    else:
                        citations[e.feature] += [log_str]
                else: # otherwise, just print the feature without any decoration
                    cite_str = '-'*5 + '\n'
                    cite_str += 'Read and cite the following:\n'
                    cite_str += log_str
                    cite_str += 'You can save this citation to file using cite.save().\n'
                    cite_str += '-'*5 + '\n'
                    globals.msg.notice(1, cite_str)

        # print each feature set together
        for feature in citations:
            cite_str = '-'*5 + '\n'
            cite_str += 'You are using %s. Read and cite the following:\n' % feature
            cite_str += 'and\n'.join(citations[feature])
            if len(citations[feature]) > 1:
                cite_str += 'You can save these citations to file using cite.save().\n'
            else:
                cite_str += 'You can save this citation to file using cite.save().\n'
            cite_str += '-'*5 + '\n'
            globals.msg.notice(1, cite_str)

        # after adding, we need to update the file
        self.updated = True

        # if autosaving is enabled, bibliography should try to save itself to file
        if self.autosave:
            self.save()

    ## \internal
    # \brief Saves the bibliography to file
    def save(self):
        if not self.should_save():
            return

        # dump all BibTeX entries to file (in no particular order)
        f = open(self.file, 'w')
        f.write('% This BibTeX file was automatically generated from HOOMD-blue\n')
        f.write('% Encoding: UTF-8\n\n')
        for entry in self.entries:
            f.write(self.entries[entry].bibtex() + '\n\n')
        f.close()

        # after saving, we no longer need updating
        self.updated = False

    ## \internal
    # \brief Set parameters for saving the %bibliography file
    # \param file %Bibliography file name
    # \param autosave Flag to have %bibliography automatically saved as needed during run
    #
    # If \a autosave is true, the bibliography will be automatically saved to file each time a new citation is added.
    def set_params(self, file=None, autosave=None):
        if file is not None:
            self.file = file
        if autosave is not None:
            self.autosave = autosave

    ## \internal
    # \brief Determines if the current rank should save the bibliography file
    def should_save(self):
        # only the root rank should save the bibliography
        if len(self.entries) == 0 or comm.get_rank() != 0:
            return False

        # otherwise, check if the bibliography has been updated since last save
        return self.updated

## \internal
# \brief Ensures that the global bibliography is properly initialized
# \returns Global bibliography
#
# Citations generated in hoomd_script should always attach to a single global bibliography. This makes %bibliography
# generation invisible to the HOOMD users (that is, they should never actually instantiate a bibliography themselves).
# This function provides a convenient way to get the global bibliography while ensuring that it exists: if globals.bib
# already exists, it returns it. Otherwise, globals.bib is first created and then returned. Any %bibliography in HOOMD
# always includes two references: (1) the original HOOMD paper and (2) the HOOMD-blue website, which are automatically
# put into the global bibliography. Subsequent citations are then added to these citations.
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
                        feature = 'HOOMD-blue')
        hoomd_web = misc(cite_key = 'hoomdweb', howpublished = 'http://codeblue.umich.edu/hoomd-blue', feature = 'HOOMD-blue')
        globals.bib.add([hoomd, hoomd_web])

    return globals.bib

## Saves the automatically generated %bibliography to a BibTeX file
#
# \param file File name for the saved %bibliography
#
# After save() is called for the first time, the %bibliography will (re-)generate each time that a new feature is added
# to ensure that all citations have been included. If \a file already exists, it will be overwritten.
#
# \b Examples
# \code
# cite.save()
# cite.save(file='cite.bib')
# \endcode
def save(file='hoomd.bib'):
    util.print_status_line()
    
    # force a bibliography to exist
    bib = _ensure_global_bib()

    # update the bibliography save parameters
    bib.set_params(file=file, autosave=True)

    # save to file
    bib.save()
