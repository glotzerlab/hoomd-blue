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

/*! \file IMDInterface.cc
    \brief Defines the IMDInterface class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
#include <boost/shared_array.hpp>
using namespace boost::python;
using namespace boost;

#include "IMDInterface.h"
#include "SignalHandler.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#include "HOOMDMPI.h"
#endif

#include "vmdsock.h"
#include "imd.h"

#include <stdexcept>

using namespace std;

/*! After construction, IMDInterface is listening for connections on port \a port.
    analyze() must be called to handle any incoming connections.
    \param sysdef SystemDefinition containing the ParticleData that will be transmitted to VMD
    \param port port number to listen for connections on
    \param pause Set to true to pause the simulation and waith for IMD_GO before continuing
    \param rate Initial rate at which to send data
    \param force Constant force used to apply forces received from VMD
    \param force_scale Factor by which to scale all forces from IMD
*/
IMDInterface::IMDInterface(boost::shared_ptr<SystemDefinition> sysdef,
                           int port,
                           bool pause,
                           unsigned int rate,
                           boost::shared_ptr<ConstForceCompute> force,
                           float force_scale)
    : Analyzer(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing IMDInterface: " << port << " " << pause << " " << rate << " " << force_scale << endl;

    if (port <= 0)
        {
        m_exec_conf->msg->error() << "analyze.imd: Invalid port specified" << endl;
        throw runtime_error("Error initializing IMDInterface");
        }

    // initialize state
    m_active = false;
    m_paused = pause;
    m_trate = rate;
    m_count = 0;
    m_force = force;
    m_force_scale = force_scale;
    m_port = port;
    if (m_force)
        m_force->setForce(0,0,0);

    // TCP socket will be initialized later
    m_is_initialized = false;
    }

void IMDInterface::initConnection()
    {
    int err = 0;
    
    // start by initializing memory
    m_tmp_coords = new float[m_pdata->getNGlobal() * 3];
    
    // intialize the listening socket
    vmdsock_init();
    m_listen_sock = vmdsock_create();
    
    // check for errors
    if (m_listen_sock == NULL)
        {
        m_exec_conf->msg->error() << "analyze.imd: Unable to create listening socket" << endl;
        throw runtime_error("Error initializing IMDInterface");
        }
        
    // bind the socket and start listening for connections on that port
    m_connected_sock = NULL;
    err = vmdsock_bind(m_listen_sock, m_port);
    
    if (err == -1)
        {
        m_exec_conf->msg->error() << "analyze.imd: Unable to bind listening socket" << endl;
        throw runtime_error("Error initializing IMDInterface");
        }
        
    err = vmdsock_listen(m_listen_sock);
    
    if (err == -1)
        {
        m_exec_conf->msg->error() << "analyze.imd: Unable to listen on listening socket" << endl;
        throw runtime_error("Error initializing IMDInterface");
        }
        
    m_exec_conf->msg->notice(2) << "analyze.imd: listening on port " << m_port << endl;
    
    m_is_initialized = true;
    }

IMDInterface::~IMDInterface()
    {
    m_exec_conf->msg->notice(5) << "Destroying IMDInterface" << endl;
 
    if (m_is_initialized)
        {
        // free all used memory
        delete[] m_tmp_coords;
        vmdsock_destroy(m_connected_sock);
        vmdsock_destroy(m_listen_sock);
        
        m_tmp_coords = NULL;
        m_connected_sock = NULL;
        m_listen_sock = NULL;
        }
    }

/*! If there is no active connection, analyze() will check to see if a connection attempt
    has been made since the last call. If so, it will attempt to handshake with VMD and
    on success will start transmitting data every time analyze() is called

    \param timestep Current time step of the simulation
*/
void IMDInterface::analyze(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("IMD");
  
#ifdef ENABLE_MPI
    bool is_root = true;
    if (m_comm)
        is_root = m_exec_conf->isRoot(); 

    if (is_root && ! m_is_initialized)
        initConnection();

    if (is_root)
#else
    if (! m_is_initialized)
        initConnection();
#endif
        {
        m_count++;
        
        do
            {
            // establish a connection if one has not been made
            if (m_connected_sock == NULL)
                establishConnectionAttempt();
            
            // dispatch incoming commands
            if (m_connected_sock)
                {
                do
                    {
                    dispatch();
                    }
                    while (m_connected_sock && messagesAvailable());
                }
            
            // quit if cntrl-C was pressed
            if (g_sigint_recvd)
                {
                g_sigint_recvd = 0;
                throw runtime_error("SIG INT received while paused in IMD");
                }
            }
            while (m_paused);
        } 

#ifdef ENABLE_MPI
    unsigned char send_coords = 0;
    if (is_root && m_connected_sock && m_active && (m_trate == 0 || m_count % m_trate == 0))
        send_coords = 1;

    if (m_comm)
        {
        bcast(send_coords, 0, m_exec_conf->getMPICommunicator());
        }

    if (send_coords)
        sendCoords(timestep);
#else
    // send data when active, connected, and the rate matches
    if (m_connected_sock && m_active && (m_trate == 0 || m_count % m_trate == 0))
        sendCoords(timestep);
#endif    

    if (m_prof)
        m_prof->pop();
    }

/*! \pre \a m_connected_sock is connected and handshaking has occured
*/
void IMDInterface::dispatch()
    {
    assert(m_connected_sock != NULL);
    
    // wait for messages, but only when paused
    int timeout = 0;
    if (m_paused)
        timeout = 5;
    
    // begin by checking to see if any commands have been received
    int length;
    int res = vmdsock_selread(m_connected_sock, timeout);
    // check to see if there are any errors
    if (res == -1)
        {
        m_exec_conf->msg->notice(3) << "analyze.imd: connection appears to have been terminated" << endl;
        processDeadConnection();
        return;
        }
    // if a command is waiting
    if (res == 1)
        {
        // receive the header
        IMDType header = imd_recv_header(m_connected_sock, &length);

        switch (header)
            {
            case IMD_DISCONNECT:
                processIMD_DISCONNECT();
                break;
            case IMD_GO:
                processIMD_GO();
                break;
            case IMD_KILL:
                processIMD_KILL();
                break;
            case IMD_MDCOMM:
                processIMD_MDCOMM(length);
                break;
            case IMD_TRATE:
                processIMD_TRATE(length);
                break;
            case IMD_PAUSE:
                processIMD_PAUSE();
                break;
            case IMD_IOERROR:
                processIMD_IOERROR();
                break;
            default:
                m_exec_conf->msg->notice(3) << "analyze.imd: received an unimplemented command (" << header << "), disconnecting" << endl;
                processDeadConnection();
                break;
            }
        }
    // otherwise no message was received, do nothing
    }

/*! \pre m_connected_sock is connected
*/
bool IMDInterface::messagesAvailable()
    {
    int res = vmdsock_selread(m_connected_sock, 0);
    
    if (res == -1)
        {
        m_exec_conf->msg->notice(3) << "analyze.imd: connection appears to have been terminated" << endl;
        processDeadConnection();
        return false;
        }
    if (res == 1)
        return true;
    else
        return false;
    }

void IMDInterface::processIMD_DISCONNECT()
    {
    // cleanly disconnect and continue running the simulation. This is no different than what we do with a dead
    // connection
    processDeadConnection();
    }

void IMDInterface::processIMD_GO()
    {
    // unpause and start transmitting data
    m_paused = false;
    m_active = true;
    m_exec_conf->msg->notice(3) << "analyze.imd: Received IMD_GO, transmitting data now" << endl;
    }

void IMDInterface::processIMD_KILL()
    {
    // disconnect (no different from handling a dead connection)
    processDeadConnection();
    // terminate the simulation
    m_exec_conf->msg->notice(3) << "analyze.imd: Received IMD_KILL message, stopping the simulation" << endl;
    throw runtime_error("Received IMD_KILL message");
    }

void IMDInterface::processIMD_MDCOMM(unsigned int n)
    {
    // mdcomm is not currently handled
    shared_array<int32> indices(new int32[n]);
    shared_array<float> forces(new float[3*n]);
    
    int err = imd_recv_mdcomm(m_connected_sock, n, &indices[0], &forces[0]);
    
    if (err)
        {
        m_exec_conf->msg->error() << "analyze.imd: Error receiving mdcomm data, disconnecting" << endl;
        processDeadConnection();
        return;
        }

#ifdef ENABLE_MPI
    if (m_comm)
        {
        m_exec_conf->msg->warning() << "analyze.imd: mdcomm currently not supported in MPI simulations." << endl;
        }
#endif
   
    if (m_force)
        {
        ArrayHandle< unsigned int > h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
        m_force->setForce(0,0,0);
        for (unsigned int i = 0; i < n; i++)
            {
            unsigned int j = h_rtag.data[indices[i]];
            m_force->setParticleForce(j,
                                      forces[3*i+0]*m_force_scale,
                                      forces[3*i+1]*m_force_scale,
                                      forces[3*i+2]*m_force_scale);
            }
        }
    else
        {
        m_exec_conf->msg->warning() << "analyze.imd: Receiving forces over IMD, but no force was given to analyze.imd. Doing nothing" << endl;
        }
    }
    
void IMDInterface::processIMD_TRATE(int rate)
    {
    m_exec_conf->msg->notice(3) << "analyze.imd: Received IMD_TRATE, setting trate to " << rate << endl;
    m_trate = rate;
    }

void IMDInterface::processIMD_PAUSE()
    {
    if (!m_paused)
        {
        m_exec_conf->msg->notice(3) << "analyze.imd: Received IMD_PAUSE, pausing simulation" << endl;
        m_paused = true;
        }
    else
        {
        m_exec_conf->msg->notice(3) << "analyze.imd: Received IMD_PAUSE, unpausing simulation" << endl;
        m_paused = false;
        }
    }

void IMDInterface::processIMD_IOERROR()
    {
    // disconnect (no different from handling a dead connection)
    processDeadConnection();
    // terminate the simulation
    m_exec_conf->msg->error() << "analyze.imd: Received IMD_IOERROR message, dropping the connection" << endl;
    }

void IMDInterface::processDeadConnection()
    {
    vmdsock_destroy(m_connected_sock);
    m_connected_sock = NULL;
    m_active = false;
    m_paused = false;
    if (m_force)
        m_force->setForce(0,0,0);
    }

/*! \pre \a m_connected_sock is not connected
    \pre \a m_listen_sock is listening
    
    \a m_listen_sock is checked for any incoming connections. If an incoming connection is found, a handshake is made
    and \a m_connected_sock is set. If no connection is established, \a m_connected_sock is set to NULL.
*/
void IMDInterface::establishConnectionAttempt()
    {
    assert(m_listen_sock != NULL);
    assert(m_connected_sock == NULL);
    
    // wait for messages, but only when paused
    int timeout = 0;
    if (m_paused)
        timeout = 5;
    
    // check to see if there is an incoming connection
    if (vmdsock_selread(m_listen_sock, timeout) > 0)
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
            m_exec_conf->msg->notice(2) << "analyze.imd: accepted connection" << endl;
            }
        }
    }

/*! \param timestep Current time step of the simulation
    \pre A connection has been established

    Sends the current coordinates to VMD for display.
*/
void IMDInterface::sendCoords(unsigned int timestep)
    {
    // take a snapshot of the particle data
    SnapshotParticleData snapshot(m_pdata->getNGlobal());
    m_pdata->takeSnapshot(snapshot);

#ifdef ENABLE_MPI
    // return now if not root rank
    if (m_comm)
        if (! m_exec_conf->isRoot()) return;
#endif

    assert(m_connected_sock != NULL);
    
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
        m_exec_conf->msg->error() << "analyze.imd: I/O error while sending energies, disconnecting" << endl;
        processDeadConnection();
        return;
        }
        
    // copy the particle data to the holding array and send it
    for (unsigned int tag = 0; tag < m_pdata->getNGlobal(); tag++)
        {
        m_tmp_coords[tag*3] = float(snapshot.pos[tag].x);
        m_tmp_coords[tag*3 + 1] = float(snapshot.pos[tag].y);
        m_tmp_coords[tag*3 + 2] = float(snapshot.pos[tag].z);
        }
    err = imd_send_fcoords(m_connected_sock, m_pdata->getNGlobal(), m_tmp_coords);
    
    if (err)
        {
        m_exec_conf->msg->error() << "analyze.imd: I/O error while sending coordinates, disconnecting" << endl;
        processDeadConnection();
        return;
        }
    }

void export_IMDInterface()
    {
    class_<IMDInterface, boost::shared_ptr<IMDInterface>, bases<Analyzer>, boost::noncopyable>
        ("IMDInterface", init< boost::shared_ptr<SystemDefinition>, int, bool, unsigned int, boost::shared_ptr<ConstForceCompute> >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

