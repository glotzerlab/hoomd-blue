// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file LogPlainTXT.h
    \brief Declares the LogPlainTXT class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Logger.h"

#ifndef __LOGPLAINTXT_H__
#define __LOGPLAINTXT_H__

//! Logs registered quantities to a delimited file
/*! \note design notes: Computes and Updaters have getProvidedLogQuantities and getLogValue. The first lists
    all quantities that the compute/updater provides (a list of strings). And getLogValue takes a string
    as an argument and returns a scalar.

    Logger will open and overwrite its log file on construction. Any number of computes and updaters
    can be registered with the Logger. It will track which quantities are provided. If any particular
    quantity is registered twice, a warning is printed and the most recent registered source will take
    effect. setLoggedQuantities will specify a list of quantities to log. When it is called, a header
    is written to the file. Every call to analyze() will result in the computes for the logged quantities
    being called and getLogValue called for each value to produce a line in the file. If a logged quantity
    is not registered, a 0 is printed to the file and a warning to stdout.

    The removeAll method can be used to clear all registered computes and updaters. hoomd will
    removeAll() and re-register all active computes and updaters before every run()

    As an option, Logger can be initialized with no file. Such a logger will skip doing anything during
    analyze() but is still available for getQuantity() operations.

    \ingroup analyzers
*/
class LogPlainTXT : public Logger
    {
    public:
        //! Constructs a logger and opens the file
        LogPlainTXT(std::shared_ptr<SystemDefinition> sysdef,
                    const std::string& fname,
                    const std::string& header_prefix="",
                    bool overwrite=false);

        //! Destructor
        ~LogPlainTXT();

        //! Selects which quantities to log
        virtual void setLoggedQuantities(const std::vector< std::string >& quantities);

        //! Sets the delimiter to use between fields
        void setDelimiter(const std::string& delimiter);

        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);

    private:
        //! The delimiter to put between columns in the file
        std::string m_delimiter;
        //! The output file name
        std::string m_filename;
        //! The prefix written at the beginning of the header line
        std::string m_header_prefix;
        //! Flag indicating this file is being appended to
        bool m_appending;
        //! The file we write out to
        std::ofstream m_file;
        //! Flag to indicate whether we have initialized the file IO
        bool m_is_initialized;
        //! true if we are writing to the output file
        bool m_file_output;

        //! Helper function to open output files
        void openOutputFiles();
    };

//! exports the Logger class to python
void export_LogPlainTXT(pybind11::module& m);

#endif
