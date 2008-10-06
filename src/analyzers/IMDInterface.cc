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

/*! \file IMDInterface.cc
	\brief Defines the IMDInterface class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "IMDInterface.h"

#include "vmdsock.h"
#include "imd.h"

#include <stdexcept>

using namespace std;

/*! After construction, IMDInterface is listening for connections on port \a port.
	analyze() must be called to handle any incoming connections.
	\param pdata ParticleData that will be transmitted to VMD
	\param port port number to listen for connections on
*/	
IMDInterface::IMDInterface(boost::shared_ptr<ParticleData> pdata, int port) : Analyzer(pdata)
	{
	int err = 0;
	
	assert(pdata);
	if (port <= 0)
		{
		cerr << endl << "***Error! Invalid port specified" << endl << endl;
		throw runtime_error("Error initializing IMDInterface");
		}
		
	// start by initializing memory
	m_tmp_coords = new float[pdata->getN() * 3];
	
	// intialize the listening socket
	vmdsock_init();
	m_listen_sock = vmdsock_create();
	
	// check for errors
	if (m_listen_sock == NULL)
		{
		cerr << endl << "***Error! Unable to create listening socket" << endl << endl;
		throw runtime_error("Error initializing IMDInterface");
		}	
			
	// bind the socket and start listening for connections on that port
	m_connected_sock = NULL;
	err = vmdsock_bind(m_listen_sock, port);
	
	if (err == -1)
		{
		cerr << endl << "***Error! Unable to bind listening socket" << endl << endl;
		throw runtime_error("Error initializing IMDInterface");
		}
	
	err = vmdsock_listen(m_listen_sock);
	
	if (err == -1)
		{
		cerr << endl << "***Error! Unable to listen on listening socket" << endl << endl;
		throw runtime_error("Error initializing IMDInterface");
		}

	cout << "analyze.imd: listening on port " << port << endl;
	
	// initialize state
	m_active = false;
	}
	
IMDInterface::~IMDInterface()
	{
	// free all used memory
	delete[] m_tmp_coords;
	vmdsock_destroy(m_connected_sock);
	vmdsock_destroy(m_listen_sock);
	
	m_tmp_coords = NULL;
	m_connected_sock = NULL;
	m_listen_sock = NULL;
	}
	
/*! If there is no active connection, analyze() will check to see if a connection attempt
	has been made since the last call. If so, it will attempt to handshake with VMD and
	on success will start transmitting data every time analyze() is called
	
	\param timestep Current time step of the simulation
*/
void IMDInterface::analyze(unsigned int timestep)
	{
	// handle a dead connection
	if (m_connected_sock == NULL)
		{
		// check to see if there is an incoming connection
		if (vmdsock_selread(m_listen_sock, 0) > 0) 
			{
			// create the connection
			m_connected_sock = vmdsock_accept(m_listen_sock);
			if (imd_handshake(m_connected_sock)) 
				{
				vmdsock_destroy(m_connected_sock);
				m_connected_sock = NULL;
				return;
				}
			else
				{
				cout << "analyze.imd: accepted connection" << endl;
				}
			}
		}

	// handle a live connection
	if (m_connected_sock && !m_active)
		{
		// begin by checking to see if any commands have been received
		int length;
		int res = vmdsock_selread(m_connected_sock, 0);
		// also check to see if there are any errors
		if (res == -1)
			{
			cout << "analyze.imd: connection appears to have been terminated" << endl;
			vmdsock_destroy(m_connected_sock);
			m_connected_sock = NULL;
			m_active = false;
			return;
			}
		// if a command is waiting
		if (res == 1)
			{
			// receive the header
			IMDType header = imd_recv_header(m_connected_sock, &length);
			// currently, only the GO command is implemented
			if (header != IMD_GO)
				{
				cout << "analyze.imd: received an unimplemented command, disconnecting" << endl;
				vmdsock_destroy(m_connected_sock);
				m_connected_sock = NULL;
				m_active = false;
				return;
				}
			else
				{
				cout << "analyze.imd: received IMD_GO, transmitting data now" << endl;
				m_active = true;
				}
			}
		}
		
	// handle a live connection that has been activated
	if (m_connected_sock && m_active)
		{
		int length;
		int res = vmdsock_selread(m_connected_sock, 0);
		// also check to see if there are any errors
		if (res == -1)
			{
			cout << "analyze.imd: connection appears to have been terminated" << endl;
			vmdsock_destroy(m_connected_sock);
			m_connected_sock = NULL;
			m_active = false;
			return;
			}
		// if a command is waiting
		if (res == 1)
			{
			// receive the header
			IMDType header = imd_recv_header(m_connected_sock, &length);
			// currently, only the GO command is implemented
			if (header == IMD_DISCONNECT || header == IMD_KILL)
				{
				cout << "analyze.imd: received a disconnect command, disconnecting" << endl;
				vmdsock_destroy(m_connected_sock);
				m_connected_sock = NULL;
				m_active = false;
				return;
				}
			else
				{
				cout << "analyze.imd: received unknown command, ignoring" << endl;
				}
			}
		
		// setup and send the energies structure
		IMDEnergies energies;
		energies.tstep = timestep;
		energies.T = 0.0f;
		energies.Etot = 0.0f;
		energies.Epot = 0.0f;
		energies.Evdw = 0.0f;
		energies.Eelec = 0.0f;
		energies.Ebond = 0.0f;
		energies.Eangle = 0.0f;
		energies.Edihe = 0.0f;
		energies.Eimpr = 0.0f;
		
		int err = imd_send_energies(m_connected_sock, &energies);
		if (err)
			{
			cerr << endl << "***Error! Error sending energies, disconnecting" << endl << endl;
			vmdsock_destroy(m_connected_sock);
			m_connected_sock = NULL;
			m_active = false;
			return;
			}
		
		// copy the particle data to the hodling array and send it
		ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
		for (unsigned int i = 0; i < arrays.nparticles; i++)
			{
			unsigned int tag = arrays.tag[i];
			m_tmp_coords[tag*3] = float(arrays.x[i]);
			m_tmp_coords[tag*3 + 1] = float(arrays.y[i]);
			m_tmp_coords[tag*3 + 2] = float(arrays.z[i]);
			}
		m_pdata->release();
			
		err = imd_send_fcoords(m_connected_sock, arrays.nparticles, m_tmp_coords);
		
		if (err)
			{
			cerr << "***Error! Error sending coordinates, disconnecting" << endl << endl;
			vmdsock_destroy(m_connected_sock);
			m_connected_sock = NULL;
			m_active = false;
			return;
			}
		}
	}
	
void export_IMDInterface()
	{
	class_<IMDInterface, boost::shared_ptr<IMDInterface>, bases<Analyzer>, boost::noncopyable>("IMDInterface", init< boost::shared_ptr<ParticleData>, int >())
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif
