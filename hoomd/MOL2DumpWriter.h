// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file MOL2DumpWriter.h
    \brief Declares the MOL2DumpWriter class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Analyzer.h"

#include <string>

#include <boost/shared_ptr.hpp>

#ifndef __MOL2_DUMP_WRITER_H__
#define __MOL2_DUMP_WRITER_H__

//! Analyzer for writing out MOL2 dump files
/*! MOL2DumpWriter writes a single .mol2 formated file each time analyze() is called. The timestep is
    added into the file name the same as HOOMDDumpWriter and PDBDumpWriter do.

    \ingroup analyzers
*/
class MOL2DumpWriter : public Analyzer
    {
    public:
        //! Construct the writer
        MOL2DumpWriter(boost::shared_ptr<SystemDefinition> sysdef, std::string fname_base);

        //! Destructor
        ~MOL2DumpWriter();

        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);

        //! Write the mol2 file
        void writeFile(std::string fname);
    private:
        std::string m_base_fname;   //!< String used to store the file name of the output file
    };

//! Exports the MOL2DumpWriter class to python
void export_MOL2DumpWriter();

#endif
