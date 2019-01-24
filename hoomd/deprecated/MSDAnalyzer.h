// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file MSDAnalyzer.h
    \brief Declares the MSDAnalyzer class
*/

#ifndef __MSD_ANALYZER_H__
#define __MSD_ANALYZER_H__

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/Analyzer.h"
#include "hoomd/ParticleGroup.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <string>
#include <fstream>
#include <memory>

//! Prints a log of the mean-squared displacement calculated over particles in the simulation
/*! On construction, MSDAnalyzer opens the given file name for writing. The file will optionally be overwritten
    or appended to. If the file is appended to, the added columns are assumed to be provided in the same order
    as with the initial generation of the file. It also records the initial positions of all particles in the
    simulation. Each time analyze() is called, the mean-squared displacement is calculated and written out to the file.

    The mean squared displacement (MSD) is calculated as:
    \f[ \langle |\vec{r} - \vec{r}_0|^2 \rangle \f]

    Multiple MSD columns may be desired in a single simulation run. Rather than requiring the user to specify
    many analyze.msd commands each with a separate file, a single class instance is designed to be capable of outputting
    many columns. The particles over which the MSD is calculated for each column are specified with a ParticleGroup.

    To allow for the continuation of msd data when a job is restarted from a file, MSDAnalyzer can assign the reference
    state r_0 from a given xml file.

    \ingroup analyzers
*/
class MSDAnalyzer : public Analyzer
    {
    public:
        //! Construct the msd analyzer
        MSDAnalyzer(std::shared_ptr<SystemDefinition> sysdef,
                    std::string fname,
                    const std::string& header_prefix="",
                    bool overwrite=false);

        //! Destructor
        ~MSDAnalyzer();

        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);

        //! Sets the delimiter to use between fields
        void setDelimiter(const std::string& delimiter);

        //! Adds a column to the analysis
        void addColumn(std::shared_ptr<ParticleGroup> group, const std::string& name);

        //! Sets r0 from an xml file
        void setR0(const std::string& xml_fname);

    private:
        //! The delimiter to put between columns in the file
        std::string m_delimiter;
        //! The prefix written at the beginning of the header line
        std::string m_header_prefix;
        //! Flag indicating this file is being appended to
        bool m_appending;

        bool m_columns_changed; //!< Set to true if the list of columns have changed
        std::ofstream m_file;   //!< The file we write out to

        std::vector<Scalar> m_initial_x;    //!< initial value of the x-component listed by tag
        std::vector<Scalar> m_initial_y;    //!< initial value of the y-component listed by tag
        std::vector<Scalar> m_initial_z;    //!< initial value of the z-component listed by tag

        std::vector<Scalar> m_initial_group_N; //!< initial value of number of group members

        //! struct for storing the particle group and name assocated with a column in the output
        struct column
            {
            //! default constructor
            column() {}
            //! constructs a column
            column(std::shared_ptr<ParticleGroup const> group, const std::string& name) :
                    m_group(group), m_name(name) {}

            std::shared_ptr<ParticleGroup const> m_group; //!< A shared pointer to the group definition
            std::string m_name;                             //!< The name to print across the file header
            };

        std::vector<column> m_columns;  //!< List of groups to output

        //! Helper function to write out the header
        void writeHeader();
        //! Helper function to calculate the MSD of a single group
        Scalar calcMSD(std::shared_ptr<ParticleGroup const> group, const SnapshotParticleData<Scalar>& snapshot);
        //! Helper function to write one row of output
        void writeRow(unsigned int timestep, const SnapshotParticleData<Scalar>& snapshot);

        //! Method to be called when particles are added/removed/sorted
        void slotParticleSort();

    };

//! Exports the MSDAnalyzer class to python
void export_MSDAnalyzer(pybind11::module& m);

#endif
