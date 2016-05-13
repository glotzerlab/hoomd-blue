// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file PDBDumpWriter.h
    \brief Declares the PDBDumpWriter class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Analyzer.h"

#include <string>

#include <boost/shared_ptr.hpp>

#ifndef __PDB_DUMP_WRITER_H__
#define __PDB_DUMP_WRITER_H__

//! Analyzer for writing out HOOMD  dump files
/*! PDBDumpWriter dumps the current positions of all particles (and optionall bonds) to a pdb file periodically
    during a simulation.

    \ingroup analyzers
*/
class PDBDumpWriter : public Analyzer
    {
    public:
        //! Construct the writer
        PDBDumpWriter(boost::shared_ptr<SystemDefinition> sysdef, std::string base_fname);

        //! Destructor
        ~PDBDumpWriter();

        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);

        //! Set the output bond flag
        void setOutputBond(bool enable)
            {
            m_output_bond = enable;
            }

        //! Helper function to write file
        void writeFile(std::string fname);
    private:
        std::string m_base_fname;   //!< String used to store the base file name of the PDB file
        bool m_output_bond;         //!< Flag telling whether to output bonds
    };

//! Exports the PDBDumpWriter class to python
void export_PDBDumpWriter();

#endif
