// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _AUTOTUNER_H_
#define _AUTOTUNER_H_

/*! \file Autotuner.h
    \brief Declaration of Autotuner
*/

#include "ExecutionConfiguration.h"

#include <string>
#include <vector>

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hoomd
    {

/// Autotuner interface
/*! Provide a templateless interface for common Autotuner methods

Autotuners start tuning on construction and lock to the best performing parameters when complete.
Call start() to begin a new autotuning sequence.

The methods getParameterPython and setParameterPython() facilitate user queries and restoring saved
autotuner values. These encode the autotuner's parameter as a Python tuple. setParameterPython()
stops any current tuning in process and uses the given parameter future steps.
*/
class PYBIND11_EXPORT AutotunerInterface
    {
    public:
    /// Call to start the autotuning sequence.
    virtual void startScan() { }

#ifndef __HIPCC__
    /// Get the autotuner parameters as a Python tuple.
    pybind11::tuple getParameterPython()
        {
        return pybind11::tuple();
        }

    /// Set autother parameters from a Python tuple.
    void setParameterPython(pybind11::tuple parameter) {};
#endif
    };

//! Autotuner for low level GPU kernel parameters
/*! Autotuner is autotunes GPU kernel parameters (such as block size) for
    performance. It runs an internal state machine and makes sweeps over all valid parameter
    values. Performance is measured just for the single kernel in question with cudaEvent timers.
    A number of sweeps are combined with a median to determine the fastest parameter.
    The sampling mode can also be changed to average or maximum. The latter is helpful when the
    distribution of kernel runtimes is bimodal, e.g. because it depends on input of variable
    size.

    The begin() and end() methods must be called before and after the kernel launch to be tuned.
    The value of the tuned parameter should be set to the return value of getParam(). begin() and
    end() drive the state machine to choose parameters and insert the cuda timing events (when
    needed). Once tuning is complete isComplete() will return true and getParam() returns the best
    performing paremeter.

    Each Autotuner instance has a string name to help identify it's output on the notice stream.

    Autotuner is not useful in non-GPU builds. Timing is performed with CUDA events and requires
    ENABLE_HIP=on. Behavior of Autotuner is undefined when ENABLE_HIP=off.

    ** Implementation ** <br>
    Internally, m_nsamples is the number of samples to take (odd for median computation).
    m_current_sample is the current sample being taken in a circular fashion, and
    m_current_element is the index of the current parameter being sampled. m_samples stores the
    time of each sampled kernel launch, and m_sample_median stores the current median of each set
    of samples. When idle, the number of calls is counted in m_calls. m_state lists the current
    state in the state machine.
*/
class PYBIND11_EXPORT Autotuner : public AutotunerInterface
    {
    public:
    /// Constructor.
    Autotuner(const std::vector<unsigned int>& parameters,
              unsigned int nsamples,
              unsigned int period,
              const std::string& name,
              std::shared_ptr<const ExecutionConfiguration> exec_conf);

    /// Constructor with implicit range.
    Autotuner(unsigned int start,
              unsigned int end,
              unsigned int step,
              unsigned int nsamples,
              unsigned int period,
              const std::string& name,
              std::shared_ptr<const ExecutionConfiguration> exec_conf);

    /// Destructor.
    ~Autotuner();

    /// Start a parameter scan.
    virtual void startScan()
        {
        m_exec_conf->msg->notice(4) << "Autotuner " << m_name << " starting scan." << endl;
        m_state = SCANNING;
        m_current_element = 0;
        m_current_sample = 0;
        m_current_param = m_parameters[m_current_element];
        }

    /// Call before kernel launch.
    void begin();

    /// Call after kernel launch.
    void end();

    /// Get the parameter to set for the kernel launch.
    /*! \returns the current parameter that should be set for the kernel launch

    While sampling, the value returned by this function will sweep though all valid parameters.
    Otherwise, it will return the fastest performing parameter.
    */
    unsigned int getParam()
        {
        return m_current_param;
        }

    /// Test if the scan is complete.
    /*! \returns true when autotuning is complete.
    */
    bool isComplete()
        {
        if (m_state != SCANNING)
            return true;
        else
            return false;
        }

    /// Set flag for synchronization via MPI
    /*! \param sync If true, synchronize parameters across all MPI ranks
     */
    void setSync(bool sync)
        {
        m_sync = sync;
        }

    /// Enumeration of sampling modes.
    enum mode_Enum
        {
        mode_median = 0, //!< Median
        mode_avg,        //!< Average
        mode_max         //!< Maximum
        };

    /// Set sampling mode
    /*! \param mode Mode to use when determining optimal parameters.
     */
    void setMode(mode_Enum mode)
        {
        m_mode = mode;
        }

    /// Build a list of thread per particle targets.
    static std::vector<unsigned int> getTppListPow2(unsigned int warpSize)
        {
        std::vector<unsigned int> v;

        for (unsigned int s = 4; s <= warpSize; s *= 2)
            {
            v.push_back(s);
            }
        v.push_back(1);
        v.push_back(2);
        return v;
        }

    protected:
    unsigned int computeOptimalParameter();

    /// State names
    enum State
        {
        IDLE,
        SCANNING
        };

    /// Number of samples to take for each parameter.
    unsigned int m_nsamples;

    /// Descriptive name.
    std::string m_name;

    /// Valid parameters.
    std::vector<unsigned int> m_parameters;

    /// Current state.
    State m_state;

    /// Current sample counter.
    unsigned int m_current_sample;

    /// Current element of the parameter array in the sample.
    unsigned int m_current_element;

    /// The current parameter value being sampled (when SCANNING) or optimal (when IDLE).
    unsigned int m_current_param;

    /// Time taken for each parameter at each sample.
    std::vector<std::vector<float>> m_samples;

    /// Processed (avg, median, or max) time for each parameter.
    std::vector<float> m_sample_median;

    /// The Execution configuration.
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf;

#ifdef ENABLE_HIP
    hipEvent_t m_start; //!< CUDA event for recording start times
    hipEvent_t m_stop;  //!< CUDA event for recording end times
#endif

    /// Synchronize results over MPI when true.
    bool m_sync;

    /// Sampling mode.
    mode_Enum m_mode;
    };

    } // end namespace hoomd

#endif // _AUTOTUNER_H_
