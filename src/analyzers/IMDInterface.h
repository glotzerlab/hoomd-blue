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

/*! \file IMDInterface.h
	\brief Declares the IMDInterface class
*/

#include <boost/shared_ptr.hpp>

#include "Analyzer.h"

#ifndef __IMD_INTERFACE_H__
#define __IMD_INTERFACE_H__

//! Iterfaces with VMD through the IMD communcations port
/*! analyze() can be called very often. When not connected to
	VMD, it will do nothing. After a connection has been established,
	which can only happen during a call to analyze(), further calls will 
	transmit particle positions to VMD.
	
	In its current implementation, only a barebones set of commands are 
	supported. The sending of any command that is not understood will
	result in the socket closing the connection.
	\ingroup analyzers
*/
class IMDInterface : public Analyzer
	{
	public:
		//! Constructor
		IMDInterface(boost::shared_ptr<ParticleData> pdata, int port = 54321);
			
		//! Destructor
		~IMDInterface();
		
		//! Handle connection requests and send current positions if connected
		void analyze(unsigned int timestep);
	private:
		void *m_listen_sock;	//!< Socket we are listening on
		void *m_connected_sock;	//!< Socket to transmit/receive data
		float *m_tmp_coords;	//!< Temporary holding location for coordinate data
		
		bool m_active;			//!< True if we have received a go command
	};
	
//! Exports the IMDInterface class to python
void export_IMDInterface();
	
#endif
