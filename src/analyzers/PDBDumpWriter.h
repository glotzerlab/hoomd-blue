/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

/*! \file PDBDumpWriter.h
	\brief Declares the PDBDumpWriter class
*/

#include <string>

#include <boost/shared_ptr.hpp>

#include "Analyzer.h"

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

		//! Write out the data for the current timestep
		void analyze(unsigned int timestep);

		//! Set the output bond flag
		void setOutputBond(bool enable)	{ m_output_bond = enable; }
	
		//! Helper function to write file
		void writeFile(std::string fname);	
	private:
		std::string m_base_fname;	//!< String used to store the base file name of the PDB file
		bool m_output_bond;			//!< Flag telling whether to output bonds
	};
	
//! Exports the PDBDumpWriter class to python
void export_PDBDumpWriter();

#endif
