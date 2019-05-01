// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file HOOMDInitializer.h
    \brief Declares the HOOMDInitializer class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/ParticleData.h"
#include "hoomd/BondedGroupData.h"
#include "hoomd/deprecated/xmlParser.h"

#include <string>
#include <vector>
#include <map>
#include <functional>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __HOOMD_INITIALIZER_H__
#define __HOOMD_INITIALIZER_H__

//! Forward declarations
class ExecutionConfiguation;
template <class Real> struct SnapshotSystemData;

//! Initializes particle data from a Hoomd input file
/*! The input XML file format is identical to the output XML file format that HOOMDDumpWriter writes.
    For more information on the XML file format design see \ref page_dev_info. Although, HOOMD's
    user guide probably has a more up to date documentation on the format.

    When HOOMDInitializer is instantiated, it reads in the XML file specified in the constructor
    and parses it into internal data structures. The initializer is then ready to be passed
    to ParticleData which will then make the needed calls to copy the data into its representation.

    HOOMD's XML file format and this class are designed to be very extensible. Parsers for inidividual
    XML nodes are written in separate functions and stored by name in the map \c m_parser_map. As the
    main parser loops through, it reads in xml nodes and fires of parsers from this map to parse each
    of them. Adding a new node to the file format parser is as simple as adding a new node parser function
    (like parsePositionNode()) and adding it to the map in the constructor.

    \ingroup data_structs
*/
class PYBIND11_EXPORT HOOMDInitializer
    {
    public:
        //! Loads in the file and parses the data
        HOOMDInitializer(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                         const std::string &fname,
                         bool wrap_coordinates = false);

        virtual ~HOOMDInitializer() { }

        //! Returns the timestep of the simulation
        virtual unsigned int getTimeStep() const;

        //! Sets the timestep of the simulation
        virtual void setTimeStep(unsigned int ts);

        //! initializes a snapshot with the particle data
        virtual std::shared_ptr< SnapshotSystemData<Scalar> > getSnapshot() const;

        //! simple vec for storing particle data
        struct vec
            {
            //! Default construtor
            vec() : x(0.0), y(0.0), z(0.0)
                {
                }
            //! Constructs a vec with given components
            /*! \param xp x-component
                \param yp y-component
                \param zp z-component
            */
            vec(Scalar xp, Scalar yp, Scalar zp) : x(xp), y(yp), z(zp)
                {
                }
            Scalar x;   //!< x-component
            Scalar y;   //!< y-component
            Scalar z;   //!< z-component
            };

        //! simple integer vec for storing particle data
        struct vec_int
            {
            //! Default construtor
            vec_int() : x(0), y(0), z(0)
                {
                }
            //! Constructs a vec with given components
            /*! \param xp x-component
                \param yp y-component
                \param zp z-component
            */
            vec_int(int xp, int yp, int zp) : x(xp), y(yp), z(zp)
                {
                }
            int x;  //!< x-component
            int y;  //!< y-component
            int z;  //!< z-component
            };

        //! Access the read particle positions
        const std::vector< vec >& getPos() { return m_pos_array; }

        //! Access the read images
        const std::vector< vec_int >& getImage() { return m_image_array; }

    private:
        //! Helper function to read the input file
        void readFile(const std::string &fname);
        //! Helper function to parse the box node
        void parseBoxNode(const XMLNode& node);
        //! Helper function to parse the position node
        void parsePositionNode(const XMLNode& node);
        //! Helper function to parse the image node
        void parseImageNode(const XMLNode& node);
        //! Helper function to parse the velocity node
        void parseVelocityNode(const XMLNode& node);
        //! Helper function to parse the mass node
        void parseMassNode(const XMLNode& node);
        //! Helper function to parse diameter node
        void parseDiameterNode(const XMLNode& node);
        //! Helper function to parse the type node
        void parseTypeNode(const XMLNode& node);
        //! Helper function to parse the body node
        void parseBodyNode(const XMLNode& node);
        //! Helper function to parse the bonds node
        void parseBondNode(const XMLNode& node);
        //! Helper function to parse the pairs node
        void parsePairNode(const XMLNode& node);
        //! Helper function to parse the angle node
        void parseAngleNode(const XMLNode& node);
        //! Helper function to parse the dihedral node
        void parseDihedralNode(const XMLNode& node);
        //! Helper function to parse the improper node
        void parseImproperNode(const XMLNode& node);
        //! Helper function to parse the constraint node
        void parseConstraintsNode(const XMLNode& node);
        //! Parse charge node
        void parseChargeNode(const XMLNode& node);
        //! Parse orientation node
        void parseOrientationNode(const XMLNode& node);
        //! Parse moment inertia node
        void parseMomentInertiaNode(const XMLNode& node);
        //! Parse orientation node
        void parseAngularMomentumNode(const XMLNode& node);

        //! Helper function for identifying the particle type id
        unsigned int getTypeId(const std::string& name);
        //! Helper function for identifying the bond type id
        unsigned int getBondTypeId(const std::string& name);
        //! Helper function for identifying the angle type id
        unsigned int getAngleTypeId(const std::string& name);
        //! Helper function for identifying the dihedral type id
        unsigned int getDihedralTypeId(const std::string& name);
        //! Helper function for identifying the improper type id
        unsigned int getImproperTypeId(const std::string& name);
        //! Helper function for identifying the pair type id
        unsigned int getPairTypeId(const std::string& name);

        std::map< std::string, std::function< void (const XMLNode&) > > m_parser_map; //!< Map for dispatching parsers based on node type

        BoxDim m_box;   //!< Simulation box read from the file
        bool m_box_read;    //!< Stores the box we read in

        unsigned int m_num_dimensions;              //!< number of spatial dimensions
        std::vector< vec > m_pos_array;             //!< positions of all particles loaded
        std::vector< vec_int > m_image_array;       //!< images of all particles loaded
        std::vector< vec > m_vel_array;             //!< velocities of all particles loaded
        std::vector< Scalar > m_mass_array;         //!< masses of all particles loaded
        std::vector< Scalar > m_diameter_array;     //!< diameters of all particles loaded
        std::vector< unsigned int > m_type_array;   //!< type values for all particles loaded
        std::vector< unsigned int > m_body_array;   //!< body values for all particles loaded
        std::vector< Scalar > m_charge_array;       //!< charge of the particles loaded
        std::vector< BondData::members_t > m_bonds; //!< Bonds read in from the file
        std::vector< unsigned int> m_bond_types;    //!< Bond types read in from the file
        std::vector< AngleData::members_t > m_angles; //!< Angle read in from the file
        std::vector< unsigned int > m_angle_types;  //!< Angle types read in from the file
        std::vector< DihedralData::members_t > m_dihedrals; //!< Dihedral read in from the file
        std::vector< unsigned int > m_dihedral_types; //!< Dihedral types read in from the file
        std::vector< ImproperData::members_t > m_impropers;  //!< Improper read in from the file
        std::vector< unsigned int > m_improper_types; //!< Improper read in from the file
        std::vector< PairData::members_t > m_pairs;  //!< Pair read in from the file
        std::vector< unsigned int > m_pair_types; //!< Pair read in from the file
        std::vector< ConstraintData::members_t > m_constraints;  //!< Constraint read in from the file
        std::vector< Scalar > m_constraint_distances;//!< Constraint distances
        unsigned int m_timestep;                     //!< The time stamp

        std::vector<std::string> m_type_mapping;          //!< The created mapping between particle types and ids
        std::vector<std::string> m_bond_type_mapping;     //!< The created mapping between bond types and ids
        std::vector<std::string> m_angle_type_mapping;    //!< The created mapping between angle types and ids
        std::vector<std::string> m_dihedral_type_mapping; //!< The created mapping between dihedral types and ids
        std::vector<std::string> m_improper_type_mapping; //!< The created mapping between improper types and ids
        std::vector<std::string> m_pair_type_mapping; //!< The created mapping between pair types and ids

        std::vector<Scalar4> m_orientation;             //!< Orientation of each particle
        std::vector<Scalar3> m_moment_inertia;       //!< Moments of inertia for each particle
        std::vector<Scalar4> m_angmom;               //!< Angular momenta
        std::string m_xml_version;                  //!< Version of XML file

        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< The execution configuration
        bool m_wrap;                                     //!< If true, wrap input coordinates into box
    };

//! Exports HOOMDInitializer to python
void export_HOOMDInitializer(pybind11::module& m);

#endif
