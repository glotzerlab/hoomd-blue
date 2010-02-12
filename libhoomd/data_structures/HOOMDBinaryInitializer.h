/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

/*! \file HOOMDBinaryInitializer.h
    \brief Declares the HOOMDBinaryInitializer class
*/

#include "ParticleData.h"
#include "WallData.h"
#include "BondData.h"
#include "AngleData.h"
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
class HOOMDBinaryInitializer : public ParticleDataInitializer
    {
    public:
        //! Loads in the file and parses the data
        HOOMDBinaryInitializer(const std::string &fname);

        //! Returns the number of dimensions
        virtual unsigned int getNumDimensions() const;
        
        //! Returns the number of particles to be initialized
        virtual unsigned int getNumParticles() const;
        
        //! Returns the number of particle types to be initialized
        virtual unsigned int getNumParticleTypes() const;
        
        //! Returns the timestep of the simulation
        virtual unsigned int getTimeStep() const;
        
        //! Sets the timestep of the simulation
        virtual void setTimeStep(unsigned int ts);
        
        //! Returns the box the particles will sit in
        virtual BoxDim getBox() const;
        
        //! Initializes the particle data arrays
        virtual void initArrays(const ParticleDataArrays &pdata) const;
        
        //! Initialize the walls
        virtual void initWallData(boost::shared_ptr<WallData> wall_data) const;
        
        //! Initialize the type name mapping
        std::vector<std::string> getTypeMapping() const;
        
        //! Returns the number of bond types to be created
        virtual unsigned int getNumBondTypes() const;
        
        //! Returns the number of angle types to be created
        virtual unsigned int getNumAngleTypes() const;
        
        //! Returns the number of dihedral types to be created
        virtual unsigned int getNumDihedralTypes() const;
        
        //! Returns the number of improper types to be created
        virtual unsigned int getNumImproperTypes() const;
        
        //! Initialize the bond data
        virtual void initBondData(boost::shared_ptr<BondData> bond_data) const;
        
        //! Initialize the angle data
        virtual void initAngleData(boost::shared_ptr<AngleData> angle_data) const;
        
        //! Initialize the dihedral data
        virtual void initDihedralData(boost::shared_ptr<DihedralData> dihedral_data) const;
        
        //! Initialize the improper data
        virtual void initImproperData(boost::shared_ptr<DihedralData> improper_data) const;
        
        //! Initialize the rigid data
        virtual void initRigidData(boost::shared_ptr<RigidData> rigid_data) const;
        
        //! Initialize the integrator data
        virtual void initIntegratorData(boost::shared_ptr<IntegratorData> integrator_data ) const;

    private:
        //! Helper function to read the input file
        void readFile(const std::string &fname);        
                
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
        std::vector< Bond > m_bonds;                //!< Bonds read in from the file
        std::vector< Angle > m_angles;              //!< Angle read in from the file
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
        
        GPUArray<Scalar4> m_com;                    //!< n_bodies length 1D array of center of mass positions
        GPUArray<Scalar4> m_vel;                    //!< n_bodies length 1D array of body velocities
        GPUArray<Scalar4> m_angmom;                 //!< n_bodies length 1D array of angular momenta in the space frame
        GPUArray<int> m_body_imagex;                //!< n_bodies length 1D array of the body image in x direction
        GPUArray<int> m_body_imagey;                //!< n_bodies length 1D array of the body image in y direction
        GPUArray<int> m_body_imagez;                //!< n_bodies length 1D array of the body image in z direction
    };

//! Exports HOOMDBinaryInitializer to python
void export_HOOMDBinaryInitializer();

#endif



