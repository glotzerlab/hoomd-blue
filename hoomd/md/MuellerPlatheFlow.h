// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


/*! \file MuellerPlatheFlow.h

    \brief Declares a class to exchange velocities of
           different spatial region, to create a flow.
*/


//!Indicate a direction in a simulation box.
#include "hoomd/HOOMDMath.h"

#ifndef __MUELLER_PLATHE_FLOW_H__
#define __MUELLER_PLATHE_FLOW_H__

extern const unsigned int INVALID_TAG;
extern const Scalar INVALID_VEL;

//! Dummy struct to keep the enums out of global scope
struct flow_enum
    {
        //! Enum for dimensions
        enum Direction
            {
            X=0,//!< X-direction
            Y,//!< Y-direction
            Z//!< Z-direction
            };
    };

//Above this line shared constructs can be declared.
#ifndef NVCC
#include "hoomd/ParticleGroup.h"
#include "hoomd/Updater.h"
#include "hoomd/Variant.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#include <cfloat>
#include <memory>

//! By exchanging velocities based on their spatial position a flow is created.
/*! \ingroup computes
*/
class PYBIND11_EXPORT MuellerPlatheFlow : public Updater
    {
    public:
        //! Constructs the compute
        //!
        //! \param direction Indicates the normal direction of the slabs.
        //! \param N_slabs Number of total slabs in the simulation box.
        //! \param min_slabs Index of slabs, where the min velocity is searched.
        //! \param max_slabs Index of slabs, where the max velocity is searched.
        //! \note N_slabs should be a multiple of the DomainDecomposition boxes in that direction.
        //! If it is not, the number is rescaled and the user is informed.
        MuellerPlatheFlow(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<ParticleGroup> group,
                          std::shared_ptr<Variant> flow_target,
                          const flow_enum::Direction slab_direction,
                          const flow_enum::Direction flow_direction,
                          const unsigned int N_slabs,
                          const unsigned int min_slab,
                          const unsigned int max_slab);

        //! Destructor
        virtual ~MuellerPlatheFlow(void);

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Returns a list of log quantities this compute calculates
        virtual std::vector< std::string > getProvidedLogQuantities(void);

        //! Calculates the requested log value and returns it
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        Scalar summed_exchanged_momentum(void) const{return m_exchanged_momentum;}

        unsigned int get_N_slabs(void)const{return m_N_slabs;}
        unsigned int get_min_slab(void)const{return m_min_slab;}
        unsigned int get_max_slab(void)const{return m_max_slab;}

        void set_min_slab(const unsigned int slab_id);
        void set_max_slab(const unsigned int slab_id);

        //! Determine, whether this part of the domain decomposition
        //! has particles in the min slab.
        bool has_min_slab(void)const{return m_has_min_slab;}
        //! Determine, whether this part of the domain decomposition
        //! has particles in the max slab.
        bool has_max_slab(void)const{return m_has_max_slab;}

        //! Call function, if the domain decomposition has changed.
        void update_domain_decomposition(void);
        //! Get the ignored variance between flow target and summed flow.
        Scalar get_flow_epsilon(void)const{return m_flow_epsilon;}
        //! Get the ignored variance between flow target and summed flow.
        void set_flow_epsilon(const Scalar flow_epsilon){m_flow_epsilon=flow_epsilon;}
        //! Trigger checks for orthorhombic checks.
        void force_orthorhombic_box_check(void){m_needs_orthorhombic_check=true;}
    protected:
        //! Swap min and max slab for a reverse flow.
        //! More efficient than separate calls of set_min_slab() and set_max_slab(),
        //! especially in MPI runs.
        void swap_min_max_slab(void);

        //! Group of particles, which are searched for the velocity exchange
        std::shared_ptr<ParticleGroup> m_group;

        virtual void search_min_max_velocity(void);
        virtual void update_min_max_velocity(void);

        //!Temporary variables to store last found min vel info.
        //!
        //! x: velocity y: mass z: tag as scalar.
        //! \note Transferring the mass is only necessary if velocities are updated in the ghost layer. This is only
        //! sometimes the case, but for the sake of simplicity it will be update here always. The performance loss
        //! should be only minimal.
        Scalar3 m_last_min_vel;

        //!Temporary variables to store last found max vel info
        //!
        //! x: velocity y: mass z: tag as scalar.
        //! \note Transferring the mass is only necessary if velocities are updated in the ghost layer. This is only
        //! sometimes the case, but for the sake of simplicity it will be update here always. The performance loss
        //! should be only minimal.

        Scalar3 m_last_max_vel;

        //! Direction perpendicular to the slabs.
        enum flow_enum::Direction m_slab_direction;
        //! Direction of the induced flow.
        enum flow_enum::Direction m_flow_direction;

    private:
        std::shared_ptr<Variant> m_flow_target;
        Scalar m_flow_epsilon;
        unsigned int m_N_slabs;
        unsigned int m_min_slab;
        unsigned int m_max_slab;

        Scalar m_exchanged_momentum;

        bool m_has_min_slab;
        bool m_has_max_slab;
        bool m_needs_orthorhombic_check;
        //! Verify that the box is orthorhombic.
        //!
        //! Returns if box is orthorhombic, but throws a runtime_error, if the box is not orthorhombic.
        void verify_orthorhombic_box(void);
#ifdef ENABLE_MPI
        struct MPI_SWAP{
            MPI_Comm comm;
            int rank;
            int size;
            int gbl_rank; //!< global rank of zero in the comm.
            bool initialized; //!< initialized struct, manually set.
            MPI_SWAP()
                :
                comm(MPI_COMM_NULL),
                rank(MPI_UNDEFINED),
                size(MPI_UNDEFINED),
                gbl_rank(MPI_UNDEFINED),
                initialized(false)
                {}
            };
        struct MPI_SWAP m_min_swap;
        struct MPI_SWAP m_max_swap;
        void init_mpi_swap(struct MPI_SWAP* ms,const int color);
        void bcast_vel_to_all(struct MPI_SWAP*ms,Scalar3*vel,const MPI_Op op);
        void mpi_exchange_velocity(void);
#endif//ENABLE_MPI
    };

//! Exports the MuellerPlatheFlow class to python
void export_MuellerPlatheFlow(pybind11::module& m);

#endif//NVCC
#endif//__MUELLER_PLATHE_FLOW_H__
