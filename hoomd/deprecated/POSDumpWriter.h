// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: harperic

/*! \file POSDumpWriter.h
    \brief Declares the POSDumpWriter class
*/

#include "hoomd/Analyzer.h"

#include <string>
#include <fstream>
#include <memory>

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

#ifndef __POS_DUMP_WRITER_H__
#define __POS_DUMP_WRITER_H__

//! Analyzer for writing out POS dump files
/*! POSDumpWriter writes to a single .pos formatted dump file. Each time analyze() is called, a new frame is written
    at the end of the file.

    \ingroup analyzers
*/
class PYBIND11_EXPORT POSDumpWriter : public Analyzer
    {
    public:
        //! Construct the writer
        POSDumpWriter(std::shared_ptr<SystemDefinition> sysdef, std::string fname);

        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);

        //! Set the def string for a shape
        void setDef(unsigned int tid, std::string def);

        //! Set whether rigid body coordinates should be written out wrapped or unwrapped.
        void setUnwrapRigid(bool enable)
            {
            m_unwrap_rigid = enable;
            }

        //! Set whether or not there is additional information to be printed via the python method addInfo
        void setAddInfo(pybind11::object addInfo)
            {
            m_write_info = true;
            m_add_info = addInfo;
            }

    private:
        std::ofstream m_file;    //!< File to write to

        std::vector< std::string > m_defs;  //!< Shape defs

        bool m_unwrap_rigid;     //!< If true, unwrap rigid bodies
        bool m_write_info; //!< If true, there is additional info to write
        pybind11::object m_add_info; // method that returns additional information
    };

//! Exports the POSDumpWriter class to python
void export_POSDumpWriter(pybind11::module& m);

#endif
