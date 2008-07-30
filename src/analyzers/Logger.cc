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

/*! \file Logger.cc
	\brief Defines the Logger class
*/

#include "Logger.h"

#ifdef USE_PYTHON
#include <boost/python.hpp>
using namespace boost::python;
#endif

#include <stdexcept>
#include <iomanip>
using namespace std;

/*! \param pdata Specified for Analyzer, but not used directly by Logger
	\param fname File name to write the log to
	
	Constructing a logger will open the file \a fname, overwriting it if it exists.
*/
Logger::Logger(boost::shared_ptr<ParticleData> pdata, const std::string& fname)
	: Analyzer(pdata), m_delimiter("\t"), m_file(fname.c_str())
	{
	// open the file
	if (!m_file.good())
		{
		cout << "Error opening log file " << fname << endl;
		throw runtime_error("Error initializing Logger");
		}
	}
		
/*! \param compute The Compute to register
	
	After the compute is registered, all of the compute's provided log quantities are available for 
	logging.
*/
void Logger::registerCompute(boost::shared_ptr<Compute> compute)
	{
	vector< string > provided_quantities = compute->getProvidedLogQuantities();
	
	// loop over all log quantities
	for (unsigned int i = 0; i < provided_quantities.size(); i++)
		{
		// first check if this quantity is already set, printing a warning if so
		if (m_compute_quantities.count(provided_quantities[i]) || m_updater_quantities.count(provided_quantities[i]))
			cout << "Warning! The log quantity " << provided_quantities[i] <<
				" has been registered more than once. Only the most recent registration takes effect" << endl;
		m_compute_quantities[provided_quantities[i]] = compute;
		}
	}
	
/*! \param updater The Updater to register
	
	After the updater is registered, all of the updater's provided log quantities are available for 
	logging.
*/	
void Logger::registerUpdater(boost::shared_ptr<Updater> updater)
	{
	vector< string > provided_quantities = updater->getProvidedLogQuantities();
	
	// loop over all log quantities
	for (unsigned int i = 0; i < provided_quantities.size(); i++)
		{
		// first check if this quantity is already set, printing a warning if so
		if (m_compute_quantities.count(provided_quantities[i]) || m_updater_quantities.count(provided_quantities[i]))
			cout << "Warning! The log quantity " << provided_quantities[i] <<
				" has been registered more than once. Only the most recent registration takes effect" << endl;
		m_updater_quantities[provided_quantities[i]] = updater;
		}
	}	
		
/*! After calling removeAll(), no quantities are registered for logging
*/
void Logger::removeAll()
	{
	m_compute_quantities.clear();
	m_updater_quantities.clear();
	}
		
/*! \param quantities A list of quantities to log
	
	When analyze() is called, each quantitiy in the list will, in order, be requested
	from the matching registered compute or updtaer and written to the file separated 
	by delimiters. After all quantities are written to the file a newline is written.
	
	Each time setLoggedQuantities is called, a header listing the column names is also written.
*/
void Logger::setLoggedQuantities(const std::vector< std::string >& quantities)
	{
	m_logged_quantities = quantities;
	
	if (quantities.size() == 0)
		{
		cout << "Warning! No quantities specified for logging" << endl;
		return;
		}
	
	// write all but the last of the quantities separated by the delimiter
	for (unsigned int i = 0; i < quantities.size()-1; i++)
		m_file << quantities[i] << m_delimiter;
	// write the last one with no delimiter after it
	m_file << quantities[quantities.size()-1] << endl;
	m_file.flush();
	}
		
		
/*! \param delimiter Delimiter to place between columns in the output file
*/
void Logger::setDelimiter(const std::string& delimiter)
	{
	m_delimiter = delimiter;
	}
	
/*! \param timestep Time step to write out data for
	
	Writes a single line of output to the log file with each specified quantity separated by 
	the delimiter;
*/
void Logger::analyze(unsigned int timestep)
	{
	// quit now if there is nothing to log
	if (m_logged_quantities.size() == 0)
		{
		return;
		}
	
	// write all but the last of the quantities separated by the delimiter
	for (unsigned int i = 0; i < m_logged_quantities.size()-1; i++)
		m_file << getValue(m_logged_quantities[i], timestep) << m_delimiter;
	// write the last one with no delimiter after it
	m_file << setprecision(10) << getValue(m_logged_quantities[m_logged_quantities.size()-1], timestep) << endl;
	m_file.flush();
	}
		
/*! \param quantity Quantity to get
	\param timestep Time step to compute value for (needed for Compute classes)
*/
Scalar Logger::getValue(const std::string &quantity, int timestep)
	{
	// check to see if the quantity exists in the compute list
	if (m_compute_quantities.count(quantity))
		{
		// update the compute
		m_compute_quantities[quantity]->compute(timestep);
		// get the log value
		return m_compute_quantities[quantity]->getLogValue(quantity);
		}
	else if (m_updater_quantities.count(quantity))
		{
		// get the log value
		return m_updater_quantities[quantity]->getLogValue(quantity);
		}
	else
		{
		cout << "Warning! Log quantity " << quantity << " is not registered, logging a value of 0" << endl;
		return Scalar(0.0);
		}
	}
	
#ifdef USE_PYTHON
void export_Logger()
	{
	class_<Logger, boost::shared_ptr<Logger>, bases<Analyzer>, boost::noncopyable>
		("Logger", init< boost::shared_ptr<ParticleData>, const std::string& >())
		.def("registerCompute", &Logger::registerCompute)
		.def("registerUpdater", &Logger::registerUpdater)
		.def("removeAll", &Logger::removeAll)
		.def("setLoggedQuantities", &Logger::setLoggedQuantities)
		.def("setDelimiter", &Logger::setDelimiter)
		;
	}
#endif	
