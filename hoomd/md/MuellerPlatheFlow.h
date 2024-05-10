// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "MuellerPlatheFlowEnum.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/Updater.h"
#include "hoomd/Variant.h"
#include <pybind11/pybind11.h>

#include <cfloat>
#include <memory>

namespace hoomd
    {
namespace md
    {
extern const unsigned int INVALID_TAG;
extern const Scalar INVALID_VEL;

//! By exchanging velocities based on their spatial position a flow is created.
/*! \ingroup computes
 */
class PYBIND11_EXPORT MuellerPlatheFlow : public Updater
    {
    public:
    //! Constructs the compute
    //!
    //! \param direction Indicates the normal direction of the slabs.
    //! \param trigger Trigger to determine when to run updater.
    //! \param N_slabs Number of total slabs in the simulation box.
    //! \param min_slabs Index of slabs, where the min velocity is searched.
    //! \param max_slabs Index of slabs, where the max velocity is searched.
    //! \note N_slabs should be a multiple of the DomainDecomposition boxes in that direction.
    //! If it is not, the number is rescaled and the user is informed.
    MuellerPlatheFlow(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<Trigger> trigger,
                      std::shared_ptr<ParticleGroup> group,
                      std::shared_ptr<Variant> flow_target,
                      std::string slab_direction_str,
                      std::string flow_direction_str,
                      const unsigned int N_slabs,
                      const unsigned int min_slab,
                      const unsigned int max_slab,
                      Scalar flow_epsilon);

    //! Destructor
    virtual ~MuellerPlatheFlow(void);

    //! Take one timestep forward
    virtual void update(uint64_t timestep);

    Scalar getSummedExchangedMomentum(void) const
        {
        return m_exchanged_momentum;
        }

    unsigned int getNSlabs(void) const
        {
        return m_N_slabs;
        }
    unsigned int getMinSlab(void) const
        {
        return m_min_slab;
        }
    unsigned int getMaxSlab(void) const
        {
        return m_max_slab;
        }
    std::shared_ptr<Variant> getFlowTarget(void) const
        {
        return m_flow_target;
        }
    std::string getSlabDirectionPython(void) const
        {
        return getStringFromDirection(m_slab_direction);
        }
    std::string getFlowDirectionPython(void) const
        {
        return getStringFromDirection(m_flow_direction);
        }

    static std::string getStringFromDirection(const enum flow_enum::Direction direction)
        {
        if (direction == flow_enum::Direction::X)
            {
            return "x";
            }
        else if (direction == flow_enum::Direction::Y)
            {
            return "y";
            }
        else if (direction == flow_enum::Direction::Z)
            {
            return "z";
            }
        else
            {
            throw std::runtime_error("Direction must be x, y, or z");
            }
        }

    static enum flow_enum::Direction getDirectionFromString(std::string direction_str)
        {
        if (direction_str == "x")
            {
            return flow_enum::Direction::X;
            }
        else if (direction_str == "y")
            {
            return flow_enum::Direction::Y;
            }
        else if (direction_str == "z")
            {
            return flow_enum::Direction::Z;
            }
        else
            {
            throw std::runtime_error("Direction must be x, y, or z");
            }
        }

    void setMinSlab(const unsigned int slab_id);
    void setMaxSlab(const unsigned int slab_id);

    //! Determine, whether this part of the domain decomposition
    //! has particles in the min slab.
    bool hasMinSlab(void) const
        {
        return m_has_min_slab;
        }
    //! Determine, whether this part of the domain decomposition
    //! has particles in the max slab.
    bool hasMaxSlab(void) const
        {
        return m_has_max_slab;
        }

    //! Call function, if the domain decomposition has changed.
    void updateDomainDecomposition(void);
    //! Get the ignored variance between flow target and summed flow.
    Scalar getFlowEpsilon(void) const
        {
        return m_flow_epsilon;
        }
    //! Set the ignored variance between flow target and summed flow.
    void setFlowEpsilon(const Scalar flow_epsilon)
        {
        m_flow_epsilon = flow_epsilon;
        }
    //! Trigger checks for orthorhombic checks.
    void forceOrthorhombicBoxCheck(void)
        {
        m_needs_orthorhombic_check = true;
        }

    protected:
    //! Swap min and max slab for a reverse flow.
    //! More efficient than separate calls of setMinSlab() and setMaxSlab(),
    //! especially in MPI runs.
    void swapMinMaxSlab(void);

    //! Group of particles, which are searched for the velocity exchange
    std::shared_ptr<ParticleGroup> m_group;

    virtual void searchMinMaxVelocity(void);
    virtual void updateMinMaxVelocity(void);

    //! Temporary variables to store last found min vel info.
    //!
    //! x: velocity y: mass z: tag as scalar.
    //! \note Transferring the mass is only necessary if velocities are updated in the ghost layer.
    //! This is only sometimes the case, but for the sake of simplicity it will be update here
    //! always. The performance loss should be only minimal.
    Scalar3 m_last_min_vel;

    //! Temporary variables to store last found max vel info
    //!
    //! x: velocity y: mass z: tag as scalar.
    //! \note Transferring the mass is only necessary if velocities are updated in the ghost layer.
    //! This is only sometimes the case, but for the sake of simplicity it will be update here
    //! always. The performance loss should be only minimal.

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
    void verifyOrthorhombicBox(void);
#ifdef ENABLE_MPI
    struct MPI_SWAP
        {
        MPI_Comm comm;
        int rank;
        int size;
        int gbl_rank;     //!< global rank of zero in the comm.
        bool initialized; //!< initialized struct, manually set.
        MPI_SWAP()
            : comm(MPI_COMM_NULL), rank(MPI_UNDEFINED), size(MPI_UNDEFINED),
              gbl_rank(MPI_UNDEFINED), initialized(false)
            {
            }
        };
    struct MPI_SWAP m_min_swap;
    struct MPI_SWAP m_max_swap;
    void initMPISwap(struct MPI_SWAP* ms, const int color);
    void bcastVelToAll(struct MPI_SWAP* ms, Scalar3* vel, const MPI_Op op);
    void mpiExchangeVelocity(void);
#endif // ENABLE_MPI
    };

    } // end namespace md
    } // end namespace hoomd
