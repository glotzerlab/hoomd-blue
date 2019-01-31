// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file IMDInterface.h
    \brief Declares the IMDInterface class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#include "Analyzer.h"
#include "ConstForceCompute.h"

#include <memory>

#ifndef __IMD_INTERFACE_H__
#define __IMD_INTERFACE_H__

//! Interfaces with VMD through the IMD communications port
/*! analyze() can be called very often. When not connected to
    VMD, it will do nothing. After a connection has been established,
    which can only happen during a call to analyze(), further calls will
    transmit particle positions to VMD.

    In its current implementation, only a barebones set of commands are
    supported. The sending of any command that is not understood will
    result in the socket closing the connection.
    \ingroup analyzers
*/
class PYBIND11_EXPORT IMDInterface : public Analyzer
    {
    public:
        //! Constructor
        IMDInterface(std::shared_ptr<SystemDefinition> sysdef,
                     int port = 54321,
                     bool pause = false,
                     unsigned int rate=1,
                     std::shared_ptr<ConstForceCompute> force = std::shared_ptr<ConstForceCompute>(),
                     float force_scale=1.0);

        //! Destructor
        ~IMDInterface();

        //! Handle connection requests and send current positions if connected
        void analyze(unsigned int timestep);
    private:
        void *m_listen_sock;    //!< Socket we are listening on
        void *m_connected_sock; //!< Socket to transmit/receive data
        float *m_tmp_coords;    //!< Temporary holding location for coordinate data

        bool m_active;          //!< True if we have received a go command
        bool m_paused;          //!< True if we are paused
        unsigned int m_trate;   //!< Transmission rate
        unsigned int m_count;   //!< Count the number of times analyze() is called (used with trate)

        bool m_is_initialized;  //!< True if the interface has been initialized
        int m_port;             //!< Port to listen on
        unsigned int m_nglobal; //!< Initial number of particles

        std::shared_ptr<ConstForceCompute> m_force;   //!< Force for applying IMD forces
        float m_force_scale;                            //!< Factor by which to scale all IMD forces

        //! Helper function that reads message headers and dispatches them to the relevant process functions
        void dispatch();
        //! Helper function to determine of messages are still available
        bool messagesAvailable();
        //! Process the IMD_DISCONNECT message
        void processIMD_DISCONNECT();
        //! Process the IMD_GO message
        void processIMD_GO();
        //! Process the IMD_KILL message
        void processIMD_KILL();
        //! Process the IMD_MDCOMM message
        void processIMD_MDCOMM(unsigned int n);
        //! Process the IMD_TRATE message
        void processIMD_TRATE(int rate);
        //! Process the IMD_PAUSE message
        void processIMD_PAUSE();
        //! Process the IMD_IOERROR message
        void processIMD_IOERROR();
        //! Process a dead connection
        void processDeadConnection();

        //! Helper function to establish a connection
        void establishConnectionAttempt();
        //! Helper function to send current data to VMD
        void sendCoords(unsigned int timestep);

        //! Initialize socket and internal state variables for communication
        void initConnection();
    };

//! Exports the IMDInterface class to python
void export_IMDInterface(pybind11::module& m);

#endif
