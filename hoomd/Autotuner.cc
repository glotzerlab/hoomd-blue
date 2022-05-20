// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Autotuner.h"

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

#include <algorithm>
#include <cfloat>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace hoomd
    {
/*! \file Autotuner.cc
    \brief Definition of Autotuner
*/

/*! \param parameters List of valid parameters
    \param nsamples Number of time samples to take at each parameter
    \param period Number of calls to begin() before sampling is redone
    \param name Descriptive name (used in messenger output)
    \param exec_conf Execution configuration
*/
Autotuner::Autotuner(const std::vector<unsigned int>& parameters,
                     unsigned int nsamples,
                     unsigned int period,
                     const std::string& name,
                     std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_nsamples(nsamples), m_name(name),
      m_parameters(parameters),
      m_exec_conf(exec_conf), m_mode(mode_median)
    {
    m_exec_conf->msg->notice(5) << "Constructing Autotuner " << nsamples << " " << period << " "
                                << name << endl;

    // ensure that m_nsamples is odd to simplify the median computation. This also ensures that
    // m_nsamples > 0.
    if ((m_nsamples & 1) == 0)
        m_nsamples += 1;

    // initialize memory
    if (m_parameters.size() == 0)
        {
        std::ostringstream s;
        s << "Error initializing autotuner: Autotuner " << m_name << " has no valid parameters.";
        throw std::runtime_error(s.str());
        }
    m_samples.resize(m_parameters.size());
    m_sample_median.resize(m_parameters.size());

    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        m_samples[i].resize(m_nsamples);
        }

// create CUDA events
#ifdef ENABLE_HIP
    hipEventCreate(&m_start);
    hipEventCreate(&m_stop);
    CHECK_CUDA_ERROR();
#endif

    m_sync = false;

    startScan();
    }

/*! \param start first valid parameter
    \param end last valid parameter
    \param step spacing between valid parameters
    \param nsamples Number of time samples to take at each parameter
    \param period Number of calls to begin() before sampling is redone
    \param name Descriptive name (used in messenger output)
    \param exec_conf Execution configuration

    \post Valid parameters will be generated with a spacing of \a step in the range [start,end]
   inclusive.
*/
Autotuner::Autotuner(unsigned int start,
                     unsigned int end,
                     unsigned int step,
                     unsigned int nsamples,
                     unsigned int period,
                     const std::string& name,
                     std::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_nsamples(nsamples), m_name(name),
      m_exec_conf(exec_conf), m_mode(mode_median)
    {
    m_exec_conf->msg->notice(5) << "Constructing Autotuner "
                                << " " << start << " " << end << " " << step << " " << nsamples
                                << " " << period << " " << name << endl;

    // initialize the parameters
    m_parameters.resize((end - start) / step + 1);
    unsigned int cur_param = start;
    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        m_parameters[i] = cur_param;
        cur_param += step;
        }

    // ensure that m_nsamples is odd to simplify the median computation. This also ensures that
    // m_nsamples > 0.
    if ((m_nsamples & 1) == 0)
        m_nsamples += 1;

    // initialize memory
    if (m_parameters.size() == 0)
        {
        std::ostringstream s;
        s << "Error initializing autotuner: Autotuner " << m_name << " has no valid parameters.";
        throw std::runtime_error(s.str());
        }
    m_samples.resize(m_parameters.size());
    m_sample_median.resize(m_parameters.size());

    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        m_samples[i].resize(m_nsamples);
        }

// create CUDA events
#ifdef ENABLE_HIP
    hipEventCreate(&m_start);
    hipEventCreate(&m_stop);
    CHECK_CUDA_ERROR();
#endif

    m_sync = false;
    startScan();
    }

Autotuner::~Autotuner()
    {
    m_exec_conf->msg->notice(5) << "Destroying Autotuner " << m_name << endl;
#ifdef ENABLE_HIP
    hipEventDestroy(m_start);
    hipEventDestroy(m_stop);
    CHECK_CUDA_ERROR();
#endif
    }

void Autotuner::begin()
    {
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

void Autotuner::end()
    {
#ifdef ENABLE_HIP
    // handle timing updates if scanning
    if (m_state == SCANNING)
        {
        hipEventRecord(m_stop, 0);
        hipEventSynchronize(m_stop);
        hipEventElapsedTime(&m_samples[m_current_element][m_current_sample], m_start, m_stop);
        m_exec_conf->msg->notice(9)
            << "Autotuner " << m_name << ": t(" << m_current_param << "," << m_current_sample
            << ") = " << m_samples[m_current_element][m_current_sample] << endl;

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
            if (m_current_sample >= m_nsamples)
                {
                m_state = IDLE;
                m_current_sample = 0;
                m_current_param = computeOptimalParameter();
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

/*! \returns The optimal parameter given the current data in m_samples.

    computeOptimalParameter computes the median, average, or maximum time among all samples for all
    elements. It then chooses the fastest time (with the lowest index breaking a tie) and returns
    the parameter that resulted in that time.
*/
unsigned int Autotuner::computeOptimalParameter()
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
                m_sample_median[i] = sum / float(v.size());
                }
            else if (m_mode == mode_max)
                {
                // Compute maximum.
                m_sample_median[i] = -FLT_MIN;
                for (std::vector<float>::iterator it = v.begin(); it != v.end(); ++it)
                    {
                    if (*it > m_sample_median[i])
                        {
                        m_sample_median[i] = *it;
                        }
                    }
                }
            else
                {
                // Compute median.
                size_t n = v.size() / 2;
                nth_element(v.begin(), v.begin() + n, v.end());
                m_sample_median[i] = v[n];
                }
            }
        }

    unsigned int opt = 0;

    if (is_root)
        {
        // Now find the minimum and maximum times in the medians.
        float min = m_sample_median[0];
        unsigned int min_idx = 0;
        float max = m_sample_median[0];

        for (unsigned int i = 1; i < m_parameters.size(); i++)
            {
            if (m_sample_median[i] < min)
                {
                min = m_sample_median[i];
                min_idx = i;
                }
            if (m_sample_median[i] > max)
                {
                max = m_sample_median[i];
                }
            }

        // Get the optimal parameter.
        opt = m_parameters[min_idx];
        unsigned int percent = int(max/min * 100.0f)-100;

        // Notify user ot optimal parameter selection.
        m_exec_conf->msg->notice(4)
            << "Autotuner " << m_name << " found optimal parameter " << opt
            << " with a performance spread of " << percent << "%." << endl;
        }

#ifdef ENABLE_MPI
    if (m_sync && nranks)
        bcast(opt, 0, m_exec_conf->getMPICommunicator());
#endif
    return opt;
    }

    } // end namespace hoomd
