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

/*! \file HOOMDBinaryInitializer.h
    \brief Declares the HOOMDBinaryInitializer class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ParticleData.h"
#include "BondedGroupData.h"
#include "WallData.h"
#include "DihedralData.h"
#include "RigidData.h"
#include "IntegratorData.h"
#include "xmlParser.h"

#include <string>
#include <vector>
#include <map>

#include <boost/bind.hpp>
#include <boost/function.hpp>

#ifndef __HOOMD_BINARY_INITIALIZER_H__
#define __HOOMD_BINARY_INITIALIZER_H__

//! Forward definition of SnapshotSystemData
class SnapshotSystemData;

//! Initializes particle data from a Hoomd input file
/*! The input XML file format is identical to the output XML file format that HOOMDDumpWriter writes.
    For more information on the XML file format design see \ref page_dev_info. Although, HOOMD's
    user guide probably has a more up to date documentation on the format.

    When HOOMDBinaryInitializer is instantiated, it reads in the XML file specified in the constructor
    and parses it into internal data structures. The initializer is then ready to be passed
    to ParticleData which will then make the needed calls to copy the data into its representation.

    HOOMD's XML file format and this class are designed to be very extensible. Parsers for inidividual
    XML nodes are written in separate functions and stored by name in the map \c m_parser_map. As the
    main parser loops through, it reads in xml nodes and fires of parsers from this map to parse each
    of them. Adding a new node to the file format parser is as simple as adding a new node parser function
    (like parsePositionNode()) and adding it to the map in the constructor.

    \ingroup data_structs
*/
class HOOMDBinaryInitializer
    {
    public:
        //! Loads in the file and parses the data
        HOOMDBinaryInitializer(boost::shared_ptr<const ExecutionConfiguration> exec_conf,
                               const std::string &fname);

        //! Returns the timestep of the simulation
        virtual unsigned int getTimeStep() const;

        //! Sets the timestep of the simulation
        virtual void setTimeStep(unsigned int ts);

        //! initializes a snapshot with the particle data
        virtual boost::shared_ptr<SnapshotSystemData> getSnapshot() const;

    private:
        //! Helper function to read the input file
        void readFile(const std::string &fname);

        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Execution configuration

        BoxDim m_box;   //!< Simulation box read from the file

        std::vector< unsigned int > m_tag_array;     //!< tags of all particles loaded
        std::vector< unsigned int > m_rtag_array;    //!< inverse tags of all particles loaded

        std::vector< Scalar > m_x_array;            //!< x position of all particles loaded
        std::vector< Scalar > m_y_array;            //!< y position of all particles loaded
        std::vector< Scalar > m_z_array;            //!< z position of all particles loaded
        std::vector< Scalar > m_vx_array;           //!< x velocity of all particles loaded
        std::vector< Scalar > m_vy_array;           //!< y velocity of all particles loaded
        std::vector< Scalar > m_vz_array;           //!< z velocity of all particles loaded
        std::vector< Scalar > m_ax_array;           //!< x acceleration of all particles loaded
        std::vector< Scalar > m_ay_array;           //!< y acceleration of all particles loaded
        std::vector< Scalar > m_az_array;           //!< z acceleration of all particles loaded

        std::vector< int > m_ix_array;              //!< x image of all particles loaded
        std::vector< int > m_iy_array;              //!< y image of all particles loaded
        std::vector< int > m_iz_array;              //!< z image of all particles loaded

        std::vector< Scalar > m_mass_array;         //!< masses of all particles loaded
        std::vector< Scalar > m_diameter_array;     //!< diameters of all particles loaded
        std::vector< unsigned int > m_type_array;   //!< type values for all particles loaded
        std::vector< Scalar > m_charge_array;       //!< charge of the particles loaded
        std::vector< Wall > m_walls;                //!< walls loaded from the file
        std::vector< BondData::members_t > m_bonds;   //!< Bonds read in from the file
        std::vector< unsigned int > m_bond_types;   //!< Bonds types read in from the file
        std::vector< AngleData::members_t > m_angles; //!< Angle read in from the file
        std::vector< unsigned int>  m_angle_types;  //!< Angle types read in from the file
        std::vector< Dihedral > m_dihedrals;        //!< Dihedral read in from the file
        std::vector< Dihedral > m_impropers;        //!< Improper read in from the file
        std::vector< unsigned int > m_body_array;   //!< Body flag of the particles loaded

        unsigned int m_timestep;                    //!< The time stamp
        unsigned int m_num_dimensions;              //!< Number of dimensions
        std::vector<IntegratorVariables> m_integrator_variables; //!< Integrator variables read in from file

        std::vector<std::string> m_type_mapping;          //!< The created mapping between particle types and ids
        std::vector<std::string> m_bond_type_mapping;     //!< The created mapping between bond types and ids
        std::vector<std::string> m_angle_type_mapping;    //!< The created mapping between angle types and ids
        std::vector<std::string> m_dihedral_type_mapping; //!< The created mapping between dihedral types and ids
        std::vector<std::string> m_improper_type_mapping; //!< The created mapping between improper types and ids

        std::vector< Scalar4 > m_com;                    //!< n_bodies length 1D array of center of mass positions
        std::vector< Scalar4 > m_vel;                    //!< n_bodies length 1D array of body velocities
        std::vector< Scalar4 > m_angmom;                 //!< n_bodies length 1D array of angular momenta in the space frame
        std::vector< int3 > m_body_image;                //!< n_bodies length 1D array of the body image
    };

//! Exports HOOMDBinaryInitializer to python
void export_HOOMDBinaryInitializer();

#endif
