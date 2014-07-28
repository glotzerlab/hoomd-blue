/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

// inclusion guard
#ifndef _AUTOTUNER_H_
#define _AUTOTUNER_H_

/*! \file Autotuner.h
    \brief Declaration of Autotuner
*/

#include "ExecutionConfiguration.h"

#include <vector>
#include <string>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

//! Autotuner for low level GPU kernel parameters
/*! **Overview** <br>
    Autotuner is a helper class that autotunes GPU kernel parameters (such as block size) for performance. It runs an
    internal state machine and makes sweeps over all valid parameter values. Performance is measured just for the single
    kernel in question with cudaEvent timers. A number of sweeps are combined with a median to determine the fastest
    parameter. Additional timing sweeps are performed at a defined period in order to update to changing conditions.

    The begin() and end() methods must be called before and after the kernel launch to be tuned. The value of the tuned
    parameter should be set to the return value of getParam(). begin() and end() drive the state machine to choose
    parameters and insert the cuda timing events (when needed).

    Autotuning can be enabled/disabled by calling setEnabled(). A disabled Autotuner makes no more parameter sweeps,
    but continues to return the last determined optimal parameter. If an Autotuner is disabled before it finishes the
    first complete sweep through parameters, the first parameter in the list is returned and a warning is issued.
    isComplete() queries if the initial scan is complete. setPeriod() changes the period at which the autotuner performs
    new scans.

    Each Autotuner instance has a string name to help identify it's output on the notice stream.

    Autotuner is not useful in non-GPU builds. Timing is performed with CUDA events and requires ENABLE_CUDA=on.
    Behavior of Autotuner is undefined when ENABLE_CUDA=off.

    ** Implementation ** <br>
    Internally, m_nsamples is the number of samples to take (odd for median computation). m_current_sample is the
    current sample being taken in a circular fashion, and m_current_element is the index of the current parameter being
    sampled. m_samples stores the time of each sampled kernel launch, and m_sample_median stores the current median of
    each set of samples. When idle, the number of calls is counted in m_calls. m_state lists the current state in the
    state machine.
*/
class Autotuner
    {
    public:
        //! Constructor
        Autotuner(const std::vector<unsigned int>& parameters,
                  unsigned int nsamples,
                  unsigned int period,
                  const std::string& name,
                  boost::shared_ptr<const ExecutionConfiguration> exec_conf);

        //! Constructor with implicit range
        Autotuner(unsigned int start,
                  unsigned int end,
                  unsigned int step,
                  unsigned int nsamples,
                  unsigned int period,
                  const std::string& name,
                  boost::shared_ptr<const ExecutionConfiguration> exec_conf);

        //! Destructor
        ~Autotuner();

        //! Call before kernel launch
        void begin();

        //! Call after kernel launch
        void end();

        //! Get the parameter to set for the kernel launch
        /*! \returns the current parameter that should be set for the kernel launch

        While sampling, the value returned by this function will sweep though all valid parameters. Otherwise, it will
        return the fastest performing parameter.
        */
        unsigned int getParam()
            {
            return m_current_param;
            }

        //! Enable/disable sampling
        /*! \param enabled true to enable sampling, false to disable it
        */
        void setEnabled(bool enabled)
            {
            m_enabled = enabled;

            if (!enabled)
                {
                m_exec_conf->msg->notice(6) << "Disable Autotuner " << m_name << std::endl;

                // if not complete, issue a warning
                if (!isComplete())
                    {
                    m_exec_conf->msg->warning() << "Disabling Autotuner " << m_name << " before initial scan completed!" << std::endl;
                    }
                else
                    {
                    // ensure that we are in the idle state and have an up to date optimal parameter
                    m_current_element = 0;
                    m_state = IDLE;
                    m_current_param = computeOptimalParameter();
                    }
                }
            else
                {
                m_exec_conf->msg->notice(6) << "Enable Autotuner " << m_name << std::endl;
                }
            }

        //! Test if initial sampling is complete
        /*! \returns true if the initial sampling run is complete
        */
        bool isComplete()
            {
            if (m_state != STARTUP)
                return true;
            else
                return false;
            }

        //! Change the sampling period
        /*! \param period New period to set
        */
        void setPeriod(unsigned int period)
            {
            m_exec_conf->msg->notice(6) << "Set Autotuner " << m_name << " period = " << period << std::endl;
            m_period = period;
            }

        //! Set flag for synchronization via MPI
        /*! \param sync If true, synchronize parameters across all MPI ranks
         */
        void setSync(bool sync)
            {
            m_sync = sync;
            }

        //! Set average flag
        /*! \param avg If true, use average instead of median of samples to compute kernel time
         */
        void setAverage(bool avg)
            {
            m_avg = avg;
            }

    protected:
        unsigned int computeOptimalParameter();

        //! State names
        enum State
           {
           STARTUP,
           IDLE,
           SCANNING
           };

        // parameters
        unsigned int m_nsamples;    //!< Number of samples to take for each parameter
        unsigned int m_period;      //!< Number of calls before sampling occurs again
        bool m_enabled;             //!< True if enabled
        std::string m_name;         //!< Descriptive name
        std::vector<unsigned int> m_parameters;  //!< valid parameters

        // state info
        State m_state;                  //!< Current state
        unsigned int m_current_sample;  //!< Current sample taken
        unsigned int m_current_element; //!< Index of current parameter sampled
        unsigned int m_calls;           //!< Count of the number of calls since the last sample
        unsigned int m_current_param;   //!< Value of the current parameter

        std::vector< std::vector< float > > m_samples;  //!< Raw sample data for each element
        std::vector< float > m_sample_median;           //!< Current sample median for each element

        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Execution configuration

        #ifdef ENABLE_CUDA
        cudaEvent_t m_start;      //!< CUDA event for recording start times
        cudaEvent_t m_stop;       //!< CUDA event for recording end times
        #endif

        bool m_sync;              //!< If true, synchronize results via MPI
        bool m_avg;               //!< If true, use sample average instead of median
    };

//! Export the Autotuner class to python
void export_Autotuner();

#endif // _AUTOTUNER_H_
