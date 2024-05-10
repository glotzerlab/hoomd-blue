// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _AUTOTUNER_H_
#define _AUTOTUNER_H_

/*! \file Autotuner.h
    \brief Declaration of Autotuner
*/

#include "ExecutionConfiguration.h"

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

#include <algorithm>
#include <array>
#include <cfloat>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
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
class PYBIND11_EXPORT AutotunerBase
    {
    public:
    AutotunerBase(const std::string& name) : m_name(name) { }

    virtual ~AutotunerBase() { }

    /// Call to start the autotuning sequence.
    virtual void startScan() { }

    /// Check if autotuning is complete.
    virtual bool isComplete()
        {
        return true;
        }

    /// Get the autotuner's name
    std::string getName()
        {
        return m_name;
        }

#ifndef __HIPCC__
    /// Get the autotuner parameters as a Python tuple.
    virtual pybind11::tuple getParameterPython()
        {
        return pybind11::tuple();
        }

    /// Set autother parameters from a Python tuple.
    virtual void setParameterPython(pybind11::tuple parameter) {};
#endif

#ifdef ENABLE_HIP
    /// Build a block size range that steps on the warp size.
    static std::vector<unsigned int>
    makeBlockSizeRange(const std::shared_ptr<const ExecutionConfiguration> exec_conf)
        {
        std::vector<unsigned int> range;
        unsigned int start = exec_conf->dev_prop.warpSize;
        unsigned int step = start;
        unsigned int end = exec_conf->dev_prop.maxThreadsPerBlock;

        range.resize((end - start) / step + 1);

        unsigned int cur_param = start;
        for (unsigned int i = 0; i < range.size(); i++)
            {
            range[i] = cur_param;
            cur_param += step;
            }
        return range;
        }

    /// Build a list of thread per particle targets.
    /*! Defaults to powers of two within a warp size. Pass a value to force_size to choose powers
        of two up to and including force_size.
    */
    static std::vector<unsigned int>
    getTppListPow2(const std::shared_ptr<const ExecutionConfiguration> exec_conf,
                   int force_size = -1)
        {
        std::vector<unsigned int> v;
        unsigned int warp_size = exec_conf->dev_prop.warpSize;
        if (force_size > 0)
            {
            warp_size = static_cast<unsigned int>(force_size);
            }

        for (unsigned int s = 4; s <= warp_size; s *= 2)
            {
            v.push_back(s);
            }

        // Include sizes 1 and 2 at the end. This forces the unit tests (which typically run only
        // the first few parameters) to run the more general tpp >= 2 code path.
        v.push_back(1);
        v.push_back(2);
        return v;
        }

#endif

    protected:
    /// Descriptive name.
    std::string m_name;
    };

//! Autotuner for low level GPU kernel parameters
/*! Autotuner is autotunes GPU kernel parameters (such as block size) for performance. It runs an
    internal state machine and makes sweeps over all valid parameter values. Each parameter value is
    a std::array<unsigned int, n_dimensions>. Performance is measured just for the single kernel in
    question with cudaEvent timers. A number of sweeps are combined with a median to determine the
    fastest parameter. The sampling mode can also be changed to average or maximum. The latter is
    helpful when the distribution of kernel runtimes is bimodal, e.g. because it depends on input of
    variable size.

    The begin() and end() methods must be called before and after the kernel launch to be tuned. The
    value of the tuned parameter should be set to the return value of getParam(). begin() and end()
    drive the state machine to choose parameters and insert the cuda timing events (when needed).
    Once tuning is complete isComplete() will return true and getParam() returns the best performing
    paremeter.

    Each Autotuner instance has a string name to help identify it's output on the notice stream.

    Autotuner is not useful in non-GPU builds. Timing is performed with CUDA events and requires
    ENABLE_HIP=on. Behavior of Autotuner is undefined when ENABLE_HIP=off.

    Internally, m_n_samples is the number of samples to take (odd for median computation).
    m_current_sample is the current sample being taken, and m_current_element is the index of the
    current parameter being sampled. m_samples stores the time of each sampled kernel launch, and
    m_sample_center stores the current median of each set of samples. m_state lists the current
    state in the state machine.

    Some classes may activate some autotuners optionally based on run time parameters. Set
    *optional* to `true` and the Autotuner will report that it is complete before it starts
    scanning. This prevents the optional autotuners from flagging the whole class as not complete
    indefinitely. This is implemented with an INACTIVE state that goes to SCANNING on the first
    call to begin().
*/
template<size_t n_dimensions> class PYBIND11_EXPORT Autotuner : public AutotunerBase
    {
    public:
    /// Constructor.
    Autotuner(
        const std::vector<std::vector<unsigned int>>& dimension_ranges,
        std::shared_ptr<const ExecutionConfiguration> exec_conf,
        const std::string& name,
        unsigned int n_samples = 5,
        bool optional = false,
        std::function<bool(const std::array<unsigned int, n_dimensions>&)> is_parameter_valid
        = [](const std::array<unsigned int, n_dimensions>& parameter) -> bool { return true; });

    /// Destructor.
    ~Autotuner()
        {
        m_exec_conf->msg->notice(5) << "Destroying Autotuner " << m_name << std::endl;
#ifdef ENABLE_HIP
        hipEventDestroy(m_start);
        hipEventDestroy(m_stop);
#endif
        }

    /// Start a parameter scan.
    virtual void startScan()
        {
        m_exec_conf->msg->notice(4) << "Autotuner " << m_name << " starting scan." << std::endl;
        m_current_element = 0;
        m_current_sample = 0;
        m_current_param = m_parameters[m_current_element];

        if (m_optional)
            {
            m_state = INACTIVE;
            }
        else
            {
            m_state = SCANNING;
            }
        }

    /// Call before kernel launch.
    void begin()
        {
        if (m_state == INACTIVE)
            {
            m_state = SCANNING;
            }

#ifdef ENABLE_HIP
        // if we are scanning, record a cuda event - otherwise do nothing
        if (m_state == SCANNING)
            {
            hipEventRecord(m_start, 0);
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
#endif
        }

    /// Call after kernel launch.
    void end();

    /// Get the parameter to set for the kernel launch.
    /*! \returns the current parameter that should be set for the kernel launch

    While sampling, the value returned by this function will sweep though all valid parameters.
    Otherwise, it will return the fastest performing parameter.
    */
    std::array<unsigned int, n_dimensions> getParam()
        {
        return m_current_param;
        }

    /// Get the autotuner parameters as a Python tuple.
    virtual pybind11::tuple getParameterPython()
        {
        pybind11::list params;
        for (auto v : m_current_param)
            {
            params.append(v);
            }
        return pybind11::tuple(params);
        }

    /// Set the autotuner parameter from a Python tuple.
    virtual void setParameterPython(pybind11::tuple parameter)
        {
        size_t n_params = pybind11::len(parameter);
        if (n_params != n_dimensions)
            {
            std::ostringstream s;
            s << "Error setting autotuner parameter " << m_name << ". Got " << n_params
              << " parameters, " << "expected " << n_dimensions << ".";
            throw std::runtime_error(s.str());
            }

        std::array<unsigned int, n_dimensions> cpp_param;
        for (size_t i = 0; i < n_dimensions; i++)
            {
            cpp_param[i] = pybind11::cast<unsigned int>(parameter[i]);
            }

        /// Check that the parameter is valid.
        auto found = std::find(m_parameters.begin(), m_parameters.end(), cpp_param);
        if (found == m_parameters.end())
            {
            std::ostringstream s;
            s << "Error setting autotuner parameter " << m_name << ". " << formatParam(cpp_param)
              << " is not a valid parameter.";
            throw std::runtime_error(s.str());
            }

        m_current_param = cpp_param;
        m_state = IDLE;
        m_current_sample = 0;

        m_exec_conf->msg->notice(4) << "Autotuner " << m_name << " setting user-defined parameter "
                                    << formatParam(cpp_param) << std::endl;
        }

    static std::string formatParam(const std::array<unsigned int, n_dimensions>& p)
        {
        std::ostringstream s;
        s << "(";
        for (auto v : p)
            {
            s << v << ",";
            }
        s << ")";
        return s.str();
        }

    /// Test if the scan is complete.
    /*! \returns true when autotuning is complete.
     */
    virtual bool isComplete()
        {
        if (m_state != SCANNING)
            return true;
        else
            {
            m_exec_conf->msg->notice(5)
                << "Autotuner " << m_name << " is not complete" << std::endl;
            return false;
            }
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

    protected:
    size_t computeOptimalParameterIndex();

    /// State names
    enum State
        {
        INACTIVE,
        IDLE,
        SCANNING
        };

    /// Number of samples to take for each parameter.
    unsigned int m_n_samples;

    /// Valid parameters.
    std::vector<std::array<unsigned int, n_dimensions>> m_parameters;

    /// Current state.
    State m_state;

    /// Current sample counter.
    unsigned int m_current_sample;

    /// Current element of the parameter array in the sample.
    unsigned int m_current_element;

    /// The current parameter value being sampled (when SCANNING) or optimal (when IDLE).
    std::array<unsigned int, n_dimensions> m_current_param;

    /// Time taken for each parameter at each sample.
    std::vector<std::vector<float>> m_samples;

    /// Processed (avg, median, or max) time for each parameter.
    std::vector<float> m_sample_center;

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

    /// True when this is an optional tuner.
    bool m_optional;

    /// Helper method to initialize multi-dimensional arrays recursively.
    void initializeParameters(
        const std::vector<std::vector<unsigned int>>& dimension_ranges,
        std::array<unsigned int, n_dimensions> parameter,
        std::function<bool(const std::array<unsigned int, n_dimensions>&)> is_parameter_valid,
        size_t current_dimension)
        {
        for (auto value : dimension_ranges[current_dimension])
            {
            if (current_dimension < n_dimensions)
                {
                parameter[current_dimension] = value;
                }

            if (current_dimension == dimension_ranges.size() - 1)
                {
                if (is_parameter_valid(parameter))
                    {
                    m_parameters.push_back(parameter);
                    m_exec_conf->msg->notice(5) << "Autotuner " << m_name << " adding parameter "
                                                << formatParam(parameter) << std::endl;
                    }
                }
            else
                {
                // populate parameter for higher dimensions
                initializeParameters(dimension_ranges,
                                     parameter,
                                     is_parameter_valid,
                                     current_dimension + 1);
                }
            }
        }
    };

/*! \param dimension_ranges Valid values for the parameters in each dimension.
    \param exec_conf Execution configuration.
    \param name Descriptive name (used in messenger output).
    \param n_samples Number of time samples to take at each parameter.
*/
template<size_t n_dimensions>
Autotuner<n_dimensions>::Autotuner(
    const std::vector<std::vector<unsigned int>>& dimension_ranges,
    std::shared_ptr<const ExecutionConfiguration> exec_conf,
    const std::string& name,
    unsigned int n_samples,
    bool optional,
    std::function<bool(const std::array<unsigned int, n_dimensions>&)> is_parameter_valid)
    : AutotunerBase(name), m_n_samples(n_samples), m_exec_conf(exec_conf), m_sync(false),
      m_mode(mode_median), m_optional(optional)
    {
    m_exec_conf->msg->notice(5) << "Constructing Autotuner " << name << " with " << n_samples
                                << " samples." << std::endl;

    if (dimension_ranges.size() != n_dimensions)
        {
        std::ostringstream s;
        s << "Autotuner " << m_name << " given invalid number of dimensions.";
        throw std::invalid_argument(s.str());
        }

    size_t n_parameters_max = 1;
    for (auto dimension : dimension_ranges)
        {
        if (dimension.size() == 0)
            {
            std::ostringstream s;
            s << "Autotuner " << m_name << " given a dimension with no values.";
            throw std::invalid_argument(s.str());
            }
        n_parameters_max *= dimension.size();
        }

    // Populate the array of samples.
    m_parameters.reserve(n_parameters_max);
    std::array<unsigned int, n_dimensions> placeholder;
    placeholder.fill(0);
    initializeParameters(dimension_ranges, placeholder, is_parameter_valid, 0);

    if (m_parameters.size() == 0)
        {
        std::ostringstream s;
        s << "Autotuner " << m_name << " has no valid parameters.";
        throw std::invalid_argument(s.str());
        }
    m_parameters.shrink_to_fit();

    // Ensure that m_n_samples is non-zero and odd to simplify the median computation.
    if ((m_n_samples & 1) == 0)
        m_n_samples += 1;

    // Initialize memory.
    m_samples.resize(m_parameters.size());
    m_sample_center.resize(m_parameters.size());

    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        m_samples[i].resize(m_n_samples);
        }

// create CUDA events
#ifdef ENABLE_HIP
    hipEventCreate(&m_start);
    hipEventCreate(&m_stop);
    CHECK_CUDA_ERROR();
#endif

    startScan();
    }

template<size_t n_dimensions> void Autotuner<n_dimensions>::end()
    {
#ifdef ENABLE_HIP
    // handle timing updates if scanning
    if (m_state == SCANNING)
        {
        hipEventRecord(m_stop, 0);
        hipEventSynchronize(m_stop);
        hipEventElapsedTime(&m_samples[m_current_element][m_current_sample], m_start, m_stop);

        m_exec_conf->msg->notice(9)
            << "Autotuner " << m_name << ": t[" << formatParam(m_current_param) << ","
            << m_current_sample << "] = " << m_samples[m_current_element][m_current_sample]
            << std::endl;

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
#endif

    // Handle state data updates and transitions.
    if (m_state == SCANNING)
        {
        // move on to the next element
        m_current_element++;

        // If we hit the end of the elements
        if (m_current_element >= m_parameters.size())
            {
            // Move on to the next sample.
            m_current_sample++;
            m_current_element = 0;

            // If this is the last sample, go to the idle state and compute the optimal parameter.
            if (m_current_sample >= m_n_samples)
                {
                m_state = IDLE;
                m_current_sample = 0;
                m_current_param = m_parameters[computeOptimalParameterIndex()];
                }
            else
                {
                m_current_param = m_parameters[m_current_element];
                }
            }
        else
            {
            m_current_param = m_parameters[m_current_element];
            }
        }
    }

/*! \returns The index of the optimal parameter given the current data in m_samples.

    computeOptimalParameter computes the median, average, or maximum time among all samples for all
    elements. It then chooses the fastest time (with the lowest index breaking a tie) and returns
    the index of the parameter that resulted in that time.
*/
template<size_t n_dimensions> size_t Autotuner<n_dimensions>::computeOptimalParameterIndex()
    {
    bool is_root = true;

#ifdef ENABLE_MPI
    unsigned int nranks = 0;
    if (m_sync)
        {
        nranks = m_exec_conf->getNRanks();
        is_root = !m_exec_conf->getRank();
        }
#endif

    // Start by computing the summary for each element.
    std::vector<float> v;
    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        v = m_samples[i];
#ifdef ENABLE_MPI
        if (m_sync && nranks)
            {
            // Combine samples from all ranks on rank zero.
            std::vector<std::vector<float>> all_v;
            MPI_Barrier(m_exec_conf->getMPICommunicator());
            gather_v(v, all_v, 0, m_exec_conf->getMPICommunicator());
            if (is_root)
                {
                v.clear();
                assert(all_v.size() == nranks);
                for (unsigned int j = 0; j < nranks; ++j)
                    v.insert(v.end(), all_v[j].begin(), all_v[j].end());
                }
            }
#endif
        if (is_root)
            {
            if (m_mode == mode_avg)
                {
                // Compute average.
                float sum = 0.0f;
                for (std::vector<float>::iterator it = v.begin(); it != v.end(); ++it)
                    sum += *it;
                m_sample_center[i] = sum / float(v.size());
                }
            else if (m_mode == mode_max)
                {
                // Compute maximum.
                m_sample_center[i] = -FLT_MIN;
                for (std::vector<float>::iterator it = v.begin(); it != v.end(); ++it)
                    {
                    if (*it > m_sample_center[i])
                        {
                        m_sample_center[i] = *it;
                        }
                    }
                }
            else
                {
                // Compute median.
                size_t n = v.size() / 2;
                nth_element(v.begin(), v.begin() + n, v.end());
                m_sample_center[i] = v[n];
                }
            }
        }

    size_t min_idx = 0;

    // Report performance characteristics of Autotuning
    if (is_root)
        {
        // Now find the minimum and maximum times in the medians.
        float min_value = m_sample_center[0];
        float max_value = m_sample_center[0];

        for (size_t i = 1; i < m_parameters.size(); i++)
            {
            if (m_sample_center[i] < min_value)
                {
                min_value = m_sample_center[i];
                min_idx = i;
                }
            if (m_sample_center[i] > max_value)
                {
                max_value = m_sample_center[i];
                }
            }

        // Get the optimal parameter.
        unsigned int percent = int(max_value / min_value * 100.0f) - 100;

        // Notify user ot optimal parameter selection.
        m_exec_conf->msg->notice(4)
            << "Autotuner " << m_name << " found optimal parameter "
            << formatParam(m_parameters[min_idx]) << " with a performance spread of " << percent
            << "%." << std::endl;
        }

#ifdef ENABLE_MPI
    if (m_sync && nranks)
        bcast(min_idx, 0, m_exec_conf->getMPICommunicator());
#endif
    return min_idx;
    }

    } // end namespace hoomd

#endif // _AUTOTUNER_H_
