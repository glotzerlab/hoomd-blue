// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


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

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

//! Autotuner for low level GPU kernel parameters
/*! **Overview** <br>
    Autotuner is a helper class that autotunes GPU kernel parameters (such as block size) for performance. It runs an
    internal state machine and makes sweeps over all valid parameter values. Performance is measured just for the single
    kernel in question with cudaEvent timers. A number of sweeps are combined with a median to determine the fastest
    parameter. Additional timing sweeps are performed at a defined period in order to update to changing conditions.
    The sampling mode can also be changed to average or maximum. The latter is helpful when the distribution of kernel
    runtimes is bimodal, e.g. because it depends on input of variable size.

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
class PYBIND11_EXPORT Autotuner
    {
    public:
        //! Constructor
        Autotuner(const std::vector<unsigned int>& parameters,
                  unsigned int nsamples,
                  unsigned int period,
                  const std::string& name,
                  std::shared_ptr<const ExecutionConfiguration> exec_conf);

        //! Constructor with implicit range
        Autotuner(unsigned int start,
                  unsigned int end,
                  unsigned int step,
                  unsigned int nsamples,
                  unsigned int period,
                  const std::string& name,
                  std::shared_ptr<const ExecutionConfiguration> exec_conf);

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
                    m_exec_conf->msg->notice(2) << "Disabling Autotuner " << m_name << " before initial scan completed!" << std::endl;
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

        //!< Enumeration of different sampling modes
        enum mode_Enum {
            mode_median = 0, //!< Median
            mode_avg,        //!< Average
            mode_max         //!< Maximum
            };

        //! Set sampling mode
        /*! \param avg If true, use average maximum instead of median of samples to compute kernel time
         */
        void setMode(mode_Enum mode)
            {
            m_mode = mode;
            }


        //! build list of thread per particle targets
        static std::vector<unsigned int> getTppListPow2(unsigned int warpSize)
            {
            std::vector<unsigned int> v;

            for (unsigned int s=4; s <= warpSize; s*=2)
                {
                v.push_back(s);
                }
            v.push_back(1);
            v.push_back(2);
            return v;
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

        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Execution configuration

        #ifdef ENABLE_CUDA
        cudaEvent_t m_start;      //!< CUDA event for recording start times
        cudaEvent_t m_stop;       //!< CUDA event for recording end times
        #endif

        bool m_sync;              //!< If true, synchronize results via MPI
        mode_Enum m_mode;         //!< The sampling mode
    };

//! Export the Autotuner class to python
void export_Autotuner(pybind11::module& m);

#endif // _AUTOTUNER_H_
