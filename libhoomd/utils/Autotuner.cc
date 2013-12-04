#include <iostream>
#include <stdexcept>
#include <algorithm>

#include <boost/python.hpp>
using namespace boost::python;

#include "Autotuner.h"

using namespace std;

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
                     boost::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_nsamples(nsamples), m_period(period), m_enabled(true), m_name(name), m_parameters(parameters),
      m_state(STARTUP), m_current_sample(0), m_current_element(0), m_calls(0), m_params_good(true),
      m_exec_conf(exec_conf)
    {
    m_exec_conf->msg->notice(5) << "Constructing Autotuner " << nsamples << " " << period << " " << name << endl;

    // ensure that m_nsamples is odd (so the median is easy to get). This also ensures that m_nsamples > 0.
    if ((m_nsamples & 1) == 0)
        m_nsamples += 1;

    // initialize memory
    if (m_parameters.size() == 0)
        {
        this->m_exec_conf->msg->error() << "Autotuner " << m_name << " got no parameters" << endl;
        throw std::runtime_error("Error initializing autotuner");
        }
    m_samples.resize(m_parameters.size());
    m_sample_median.resize(m_parameters.size());

    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        m_samples[i].resize(m_nsamples);
        }

    m_current_param = m_parameters[m_current_element];

    // create CUDA events
    #ifdef ENABLE_CUDA
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
    CHECK_CUDA_ERROR();
    #endif
    }


/*! \param start first valid parameter
    \param end last valid parameter
    \param step spacing between valid parameters
    \param nsamples Number of time samples to take at each parameter
    \param period Number of calls to begin() before sampling is redone
    \param name Descriptive name (used in messenger output)
    \param exec_conf Execution configuration

    \post Valid parameters will be generated with a spacing of \a step in the range [start,end] inclusive.
*/
Autotuner::Autotuner(unsigned int start,
                     unsigned int end,
                     unsigned int step,
                     unsigned int nsamples,
                     unsigned int period,
                     const std::string& name,
                     boost::shared_ptr<const ExecutionConfiguration> exec_conf)
    : m_nsamples(nsamples), m_period(period), m_enabled(true), m_name(name),
      m_state(STARTUP), m_current_sample(0), m_current_element(0), m_calls(0), m_current_param(0),
      m_exec_conf(exec_conf)
    {
    m_exec_conf->msg->notice(5) << "Constructing Autotuner " << " " << start << " " << end << " " << step << " "
                                << nsamples << " " << period << " " << name << endl;

    // initialize the parameters
    m_parameters.resize((end - start) / step + 1);
    unsigned int cur_param = start;
    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        m_parameters[i] = cur_param;
        cur_param += step;
        }

    // ensure that m_nsamples is odd (so the median is easy to get). This also ensures that m_nsamples > 0.
    if ((m_nsamples & 1) == 0)
        m_nsamples += 1;

    // initialize memory
    if (m_parameters.size() == 0)
        {
        m_exec_conf->msg->error() << "Autotuner " << m_name << " got no parameters" << endl;
        throw std::runtime_error("Error initializing autotuner");
        }
    m_samples.resize(m_parameters.size());

    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        m_samples[i].resize(m_nsamples);
        }

    m_current_param = m_parameters[m_current_element];

    // create CUDA events
    #ifdef ENABLE_CUDA
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
    CHECK_CUDA_ERROR();
    #endif
    }

Autotuner::~Autotuner()
    {
    m_exec_conf->msg->notice(5) << "Destroying Autotuner " << m_name << endl;
    #ifdef ENABLE_CUDA
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
    CHECK_CUDA_ERROR();
    #endif
    }

void Autotuner::begin()
    {
    #ifdef ENABLE_CUDA
    // if we are scanning, record a cuda event - otherwise do nothing
    if (m_state == STARTUP || m_state == SCANNING)
        {
        cudaEventRecord(m_start, 0);
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    #endif
    }

void Autotuner::end()
    {
    #ifdef ENABLE_CUDA
    m_params_good = true;

    // handle timing updates if scanning
    if (m_state == STARTUP || m_state == SCANNING)
        {
        cudaEventRecord(m_stop, 0);
        cudaEventSynchronize(m_stop);

        // catch errors resulting from invalid parameters
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            {
            // remove this parameter from set of valid parameters
            if (m_parameters.size() > 1)
                {
                m_exec_conf->msg->notice(10) << "Autotuner " << m_name << ": Removing t(" << m_current_param
                    << ") from list of valid parameters" << endl;
                m_parameters.erase(m_parameters.begin()+m_current_element);
                m_samples.erase(m_samples.begin()+m_current_element);
                m_params_good = false;
                }
            else
                // if there are no more parameters left, we can only hope the error will be handled by the caller
                return;
            }

        if (m_params_good)
            {
            // record elapsed time
            cudaEventElapsedTime(&m_samples[m_current_element][m_current_sample], m_start, m_stop);
            m_exec_conf->msg->notice(10) << "Autotuner " << m_name << ": t(" << m_current_param << "," << m_current_sample
                                         << ") = " << m_samples[m_current_element][m_current_sample] << endl;
            }

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    #endif

    // handle state data updates and transitions
    if (m_state == STARTUP)
        {
        // move on to the next sample
        m_current_sample++;

        // if we hit the end of the samples, reset and move on to the next element
        if (m_current_sample >= m_nsamples || ! m_params_good)
            {
            m_current_sample = 0;
            if (m_params_good) m_current_element++;

            // if we hit the end of the elements, transition to the IDLE state and compute the optimal parameter
            if (m_current_element >= m_parameters.size())
                {
                m_current_element = 0;
                m_state = IDLE;
                m_current_param = computeOptimalParameter();
                }
            else
                {
                // if moving on to the next element, update the cached parameter to set
                m_current_param = m_parameters[m_current_element];
                }
            }
        }
    else if (m_state == SCANNING)
        {
        // move on to the next element
        if (m_params_good) m_current_element++;

        // if we hit the end of the elements, transition to the IDLE state and compute the optimal parameter, and move
        // on to the next sample for next time
        if (m_current_element >= m_parameters.size())
            {
            m_current_element = 0;
            m_state = IDLE;
            m_current_param = computeOptimalParameter();
            m_current_sample = (m_current_sample + 1) % m_nsamples;
            }
        else
            {
            // if moving on to the next element, update the cached parameter to set
            m_current_param = m_parameters[m_current_element];
            }
        }
    else if (m_state == IDLE)
        {
        // increment the calls counter and see if we should transition to the scanning state
        m_calls++;

        if (m_calls > m_period)
            {
            // reset state for the next time
            m_calls = 0;

            // initialize a scan
            m_current_param = m_parameters[m_current_element];
            m_state = SCANNING;
            }
        }
    }

/*! \returns The optimal parameter given the current data in m_samples

    computeOptimalParameter computes the median time among all samples for a given element. It then chooses the
    fastest time (with the lowest index breaking a tie) and returns the parameter that resulted in that time.
*/
unsigned int Autotuner::computeOptimalParameter()
    {
    // start by computing the median for each element
    std::vector<float> v;
    m_sample_median.resize(m_parameters.size());

    for (unsigned int i = 0; i < m_parameters.size(); i++)
        {
        v = m_samples[i];
        size_t n = v.size() / 2;
        nth_element(v.begin(), v.begin()+n, v.end());
        m_sample_median[i] = v[n];
        }

    // now find the minimum and maximum times in the medians
    float min = m_sample_median[0];
    unsigned int min_idx = 0;
    float max = m_sample_median[0];
    unsigned int max_idx = 0;

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
            max_idx = i;
            }
        }

    // get the optimal param
    unsigned int opt = m_parameters[min_idx];
    unsigned int percent = int(max/min * 100.0f)-100;

    // print stats
    m_exec_conf->msg->notice(4) << "Autotuner " << m_name << " found optimal parameter " << opt << " which is " << percent
                                << " percent faster than " << m_parameters[max_idx] << "." << endl;

    return opt;
    }

void export_Autotuner()
    {
    class_<Autotuner, boost::noncopyable>
    ("Autotuner", init< unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, const std::string&, boost::shared_ptr<ExecutionConfiguration> >())
    .def("getParam", &Autotuner::getParam)
    .def("setEnabled", &Autotuner::setEnabled)
    .def("setMoveRatio", &Autotuner::isComplete)
    .def("setNSelect", &Autotuner::setPeriod)
    ;
    }
