/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: harperic

/*! \file POSDumpWriter.h
    \brief Declares the POSDumpWriter class
*/

#include "Analyzer.h"

#include <string>
#include <fstream>
#include <boost/shared_ptr.hpp>

#ifndef __POS_DUMP_WRITER_H__
#define __POS_DUMP_WRITER_H__

//! Analyzer for writing out POS dump files
/*! POSDumpWriter writes to a single .pos formatted dump file. Each time analyze() is called, a new frame is written
    at the end of the file.

    \ingroup analyzers
*/
class POSDumpWriter : public Analyzer
    {
    public:
        //! Construct the writer
        POSDumpWriter(boost::shared_ptr<SystemDefinition> sysdef, std::string fname);

        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);

        //! Set the def string for a shape
        void setDef(unsigned int tid, std::string def);

        //! Set whether rigid body coordinates should be written out wrapped or unwrapped.
        void setUnwrapRigid(bool enable)
            {
            m_unwrap_rigid = enable;
            }

        //! Set whether or not there is additional information to be printed via the callback
        void setAddInfo(boost::python::object callback)
            {
            m_write_info = true;
            m_add_info = callback;
            }

    private:
        std::ofstream m_file;    //!< File to write to

        std::vector< std::string > m_defs;  //!< Shape defs

        boost::shared_ptr<RigidData> m_rigid_data; //!< For accessing rigid body data
        bool m_unwrap_rigid;     //!< If true, unwrap rigid bodies
        bool m_write_info; //!< If true, there is additional info to write
        boost::python::object m_add_info; // callback 
    };

//! Exports the POSDumpWriter class to python
void export_POSDumpWriter();

#endif
