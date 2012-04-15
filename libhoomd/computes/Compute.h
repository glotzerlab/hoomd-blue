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

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>
#include <string>
#include <vector>

#include "SystemDefinition.h"
#include "Profiler.h"

#ifndef __COMPUTE_H__
#define __COMPUTE_H__

/*! \file Compute.h
    \brief Declares a base class for all computes
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

/*! \ingroup hoomd_lib
    @{
*/

/*! \defgroup computes Computes
    \brief All classes that implement the Compute concept.
    \details See \ref page_dev_info for more information
*/

/*! @}
*/

//! Performs computations on ParticleData structures
/*! The Compute is an abstract concept that performs some kind of computation on the
    particles in a ParticleData structure. This computation is to be done by reading
    the particle data only, no writing. Computes will be used to generate neighbor lists,
    calculate forces, and calculate temperatures, just to name a few.

    For performance and simplicity, each compute is associated with a ParticleData
    on construction. ParticleData pointers are managed with reference counted boost::shared_ptr.
    Since each ParticleData cannot change size, this allows the Compute to preallocate
    any data structures that it may need.

    Computes may be referenced more than once and may reference other computes. To prevent
    uneeded data from being calculated, the time step will be passed into the compute
    method so that it can skip caculations if they have already been done this timestep.
    For convenience, the base class will provide a shouldCompute() method that implements
    this behaviour. Derived classes can override if more complicated behavior is needed.

    See \ref page_dev_info for more information
    \ingroup computes
*/
class Compute : boost::noncopyable
    {
    public:
        //! Constructs the compute and associates it with the ParticleData
        Compute(boost::shared_ptr<SystemDefinition> sysdef);
        virtual ~Compute() {};
        
        //! Abstract method that performs the computation
        /*! \param timestep Current time step
            Derived classes will implement this method to calculate their results
        */
        virtual void compute(unsigned int timestep) = 0;

        //! Abstract method that performs a benchmark
        virtual double benchmark(unsigned int num_iters);
        
        //! Print some basic stats to stdout
        /*! Derived classes can optionally implement this function. A System will
            call all of the Compute's printStats functions at the end of a run
            so the user can see useful information 
        */
        virtual void printStats()
            {
            }
            
        //! Reset stat counters
        /*! If derived classes implement printStats, they should also implement resetStats() to clear any running 
            counters printed by printStats. System will reset the stats before any run() so that stats printed
            at the end of the run only apply to that run() alone.
        */
        virtual void resetStats()
            {
            }
            
        //! Sets the profiler for the compute to use
        void setProfiler(boost::shared_ptr<Profiler> prof);
        
        //! Returns a list of log quantities this compute calculates
        /*! The base class implementation just returns an empty vector. Derived classes should override
            this behavior and return a list of quantities that they log.
        
            See Logger for more information on what this is about.
        */
        virtual std::vector< std::string > getProvidedLogQuantities()
            {
            return std::vector< std::string >();
            }
            
        //! Calculates the requested log value and returns it
        /*! \param quantity Name of the log quantity to get
            \param timestep Current time step of the simulation
        
            The base class just returns 0. Derived classes should override this behavior and return
            the calculated value for the given quantity. Only quantities listed in
            the return value getProvidedLogQuantities() will be requested from
            getLogValue().
        
            See Logger for more information on what this is about.
        */
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep)
            {
            return Scalar(0.0);
            }

        //! Force recalculation of compute
        /*! If this function is called, recalculation of the compute will be forced (even if had
         *  been calculated earlier in this timestep)
         * \param timestep current timestep
         */
        virtual void forceCompute(unsigned int timestep);

#ifdef ENABLE_MPI
        //! Set communicator this Compute is to use
        /*! \param comm The communicator
         */
        virtual void setCommunicator(boost::shared_ptr<Communicator> comm)
            {
            m_comm = comm;
            }
#endif

    protected:
        const boost::shared_ptr<SystemDefinition> m_sysdef; //!< The system definition this compute is associated with
        const boost::shared_ptr<ParticleData> m_pdata;      //!< The particle data this compute is associated with
        boost::shared_ptr<Profiler> m_prof;                 //!< The profiler this compute is to use
        boost::shared_ptr<const ExecutionConfiguration> exec_conf; //!< Stored shared ptr to the execution configuration
#ifdef ENABLE_MPI
        boost::shared_ptr<Communicator> m_comm;             //!< The communicator this compute is to use
#endif
        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Stored shared ptr to the execution configuration
        // OK, the dual exec_conf and m_exe_conf is weird - exec_conf was from legacy code. m_exec_conf is the new
        // standard. But I don't want to remove the old one until we have fewer branches open in hoomd so as to avoid
        // merge conflicts.

        //! Simple method for testing if the computation should be run or not
        virtual bool shouldCompute(unsigned int timestep);
    private:
        unsigned int m_last_computed;   //!< Stores the last timestep compute was called
        bool m_first_compute;           //!< true if compute has not yet been called
        bool m_force_compute;           //!< true if calculation is enforced
        
        //! The python export needs to be a friend to export shouldCompute()
        friend void export_Compute();
    };

//! Exports the Compute class to python
void export_Compute();

#endif

