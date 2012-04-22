/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

/*! \file Messenger.cc
    \brief Defines the Messenger class
*/

#include "Messenger.h"
#include <assert.h>
using namespace std;

#include <boost/python.hpp>
using namespace boost::python;

/*! \post Warning and error streams are set to cerr
    \post The notice stream is set to cout
    \post The notice level is set to 1
    \post prefixes are "error!!!!" , "warning!!" and "notice"
*/
Messenger::Messenger()
    {
    m_err_stream = &cerr;
    m_warning_stream = &cerr;
    m_notice_stream = &cout;
    m_nullstream = boost::shared_ptr<nullstream>(new nullstream());
    m_notice_level = 2;
    m_err_prefix     = "**ERROR**";
    m_warning_prefix = "*Warning*";
    m_notice_prefix  = "notice";
    }

Messenger::~Messenger()
    {
    // set pointers to NULL
    m_err_stream = NULL;
    m_warning_stream = NULL;
    m_notice_stream = NULL;
    }

/*! \returns The error stream for use in printing error messages
    \post If the error prefix is not the empty string, the message is preceded with
    "${error_prefix}: ".
*/
std::ostream& Messenger::error() const
    {
    assert(m_err_stream);
    if (m_err_prefix != string(""))
        *m_err_stream << m_err_prefix << ": ";
    return *m_err_stream;
    }

/*! \param msg Message to print
    \sa error()
*/
void Messenger::errorStr(const std::string& msg) const
    {
    error() << msg;
    }

/*! \returns The warning stream for use in printing warning messages
    \post If the warning prefix is not the empty string, the message is preceded with
    "${warning_prefix}: ".
*/
std::ostream& Messenger::warning() const
    {
    assert(m_warning_stream);
    if (m_warning_prefix != string(""))
        *m_warning_stream << m_warning_prefix << ": ";
    return *m_warning_stream;
    }

/*! \param msg Message to print
    \sa warning()
*/
void Messenger::warningStr(const std::string& msg) const
    {
    warning() << msg;
    }

/*! \returns The notice stream for use in printing notice messages
    \post If the notice prefix is not the empty string and the level is greater than 1 the message is preceded with
    "${notice_prefix}(n): ".

    If level is greater than the notice level, a null stream is returned so that the output is not printed.
*/
std::ostream& Messenger::notice(unsigned int level) const
    {
    assert(m_notice_stream);
    if (level <= m_notice_level)
        {
        if (m_notice_prefix != string("") && level > 1)
            *m_notice_stream << m_notice_prefix << "(" << level << "): ";
        return *m_notice_stream;
        }
    else
        {
        return *m_nullstream;
        }
    }

/*! \param level Notice level
    \param msg Message to print
    \sa notice()
*/
void Messenger::noticeStr(unsigned int level, const std::string& msg) const
    {
    notice(level) << msg;
    }

/*! \param fname File name
    The file is ovewritten if it exists. If there is an error opening the file, all level's streams are left
    as is and an error() is issued.
*/
void Messenger::openFile(const std::string& fname)
    {
    m_file = boost::shared_ptr<std::ofstream>(new ofstream(fname.c_str()));
    m_err_stream = m_file.get();
    m_warning_stream = m_file.get();
    m_notice_stream = m_file.get();
    }

/*! Any open file is closed. stdout is opened again for notices and stderr for warnings and errors.
*/
void Messenger::openStd()
    {
    m_file = boost::shared_ptr<std::ofstream>();
    m_err_stream = &cerr;
    m_warning_stream = &cerr;
    m_notice_stream = &cout;
    }

void export_Messenger()
    {
    class_<Messenger, boost::shared_ptr<Messenger>, boost::noncopyable >
         ("Messenger", init< >())
         .def("error", &Messenger::errorStr)
         .def("warning", &Messenger::warningStr)
         .def("notice", &Messenger::noticeStr)
         .def("getNoticeLevel", &Messenger::getNoticeLevel)
         .def("setNoticeLevel", &Messenger::setNoticeLevel)
         .def("getErrorPrefix", &Messenger::getErrorPrefix, return_value_policy<copy_const_reference>())
         .def("setErrorPrefix", &Messenger::setErrorPrefix)
         .def("getWarningPrefix", &Messenger::getWarningPrefix, return_value_policy<copy_const_reference>())
         .def("setWarningPrefix", &Messenger::setWarningPrefix)
         .def("getNoticePrefix", &Messenger::getNoticePrefix, return_value_policy<copy_const_reference>())
         .def("setWarningPrefix", &Messenger::setWarningPrefix)
         .def("openFile", &Messenger::openFile)
         .def("openStd", &Messenger::openStd)
         ;
    }
