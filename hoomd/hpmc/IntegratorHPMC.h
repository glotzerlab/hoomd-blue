// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _INTEGRATOR_HPMC_H_
#define _INTEGRATOR_HPMC_H_

/*! \file IntegratorHPMC.h
    \brief Declaration of IntegratorHPMC
*/


#include "hoomd/Integrator.h"
#include "hoomd/CellList.h"

#include "HPMCCounters.h"
#include "ExternalField.h"

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace hpmc
{

//! Integrator that implements the HPMC approach
/*! **Overview** <br>
    IntegratorHPMC is an non-templated base class that implements the basic methods that all HPMC integrators have.
    This provides a base interface that any other code can use when given a shared pointer to an IntegratorHPMC.

    The move ratio is stored as an unsigned int (0xffff = 100%) to avoid numerical issues when the move ratio is exactly
    at 100%.

    \ingroup hpmc_integrators
*/

class PatchEnergy
    {
    public:
        PatchEnergy() { }
        virtual ~PatchEnergy() { }

    //! Returns the cut-off radius
    virtual Scalar getRCut()
        {
        return 0;
        }

    //! Returns the geometric extent, per type
    virtual Scalar getAdditiveCutoff(unsigned int type)
        {
        return 0;
        }

    //! evaluate the energy of the patch interaction
    /*! \param r_ij Vector pointing from particle i to j
        \param type_i Integer type index of particle i
        \param d_i Diameter of particle i
        \param charge_i Charge of particle i
        \param q_i Orientation quaternion of particle i
        \param type_j Integer type index of particle j
        \param q_j Orientation quaternion of particle j
        \param d_j Diameter of particle j
        \param charge_j Charge of particle j
        \returns Energy of the patch interaction.
    */
    virtual float energy(const vec3<float>& r_ij,
        unsigned int type_i,
        const quat<float>& q_i,
        float d_i,
        float charge_i,
        unsigned int type_j,
        const quat<float>& q_j,
        float d_j,
        float charge_j)
        {
        return 0;
        }

    };

class PYBIND11_EXPORT IntegratorHPMC : public Integrator
    {
    public:
        //! Constructor
        IntegratorHPMC(std::shared_ptr<SystemDefinition> sysdef,
                       unsigned int seed);

        virtual ~IntegratorHPMC();

        //! Take one timestep forward
        virtual void update(unsigned int timestep)
            {
            ArrayHandle<hpmc_counters_t> h_counters(m_count_total, access_location::host, access_mode::read);
            m_count_step_start = h_counters.data[0];
            }

        //! Change maximum displacement
        /*! \param d new d to set
         *! \param typ type to which d will be set
        */
        void setD(Scalar d,unsigned int typ)
            {
                {
                ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::readwrite);
                h_d.data[typ] = d;
                }
            updateCellWidth();
            }

        //! Get maximum displacement (by type)
        inline Scalar getD(unsigned int typ)
            {
            ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);
            return h_d.data[typ];
            }

        //! Get array of translation move sizes
        const GPUArray<Scalar>& getDArray() const
            {
            return m_d;
            }

        //! Get the maximum particle translational move size
        virtual Scalar getMaxTransMoveSize()
            {
            // access the type parameters
            ArrayHandle<Scalar> h_d(m_d, access_location::host, access_mode::read);

            // for each type, create a temporary shape and return the maximum diameter
            Scalar maxD = Scalar(0.0);
            for (unsigned int typ = 0; typ < this->m_pdata->getNTypes(); typ++)
                {
                maxD = std::max(maxD, h_d.data[typ]);
                }

            return maxD;
            }

        //! Change maximum rotation
        /*! \param a new a to set
         *! \param type type to which d will be set
        */
        void setA(Scalar a,unsigned int typ)
            {
            ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::readwrite);
            h_a.data[typ] = a;
            }

        //! Get maximum rotation
        inline Scalar getA(unsigned int typ)
            {
            ArrayHandle<Scalar> h_a(m_a, access_location::host, access_mode::read);
            return h_a.data[typ];
            }

        //! Get array of rotation move sizes
        const GPUArray<Scalar>& getAArray() const
            {
            return m_a;
            }

        //! Change move ratio
        /*! \param move_ratio new move_ratio to set
        */
        void setMoveRatio(Scalar move_ratio)
            {
            m_move_ratio = unsigned(move_ratio*65536);
            }

        //! Get move ratio
        //! \returns ratio of translation versus rotation move attempts
        inline double getMoveRatio()
            {
            return m_move_ratio/65536.0;
            }

        //! Set nselect
        /*! \param nselect new nselect value to set
        */
        void setNSelect(unsigned int nselect)
            {
            m_nselect = nselect;
            updateCellWidth();
            }

        //! Get nselect
        //! \returns current value of nselect parameter
        inline unsigned int getNSelect()
            {
            return m_nselect;
            }

        //! Print statistics about the hmc steps taken
        virtual void printStats()
            {
            hpmc_counters_t counters = getCounters(1);
            m_exec_conf->msg->notice(2) << "-- HPMC stats:" << "\n";
            m_exec_conf->msg->notice(2) << "Average translate acceptance: " << counters.getTranslateAcceptance() << "\n";
            if (counters.rotate_accept_count + counters.rotate_reject_count != 0)
                {
                m_exec_conf->msg->notice(2) << "Average rotate acceptance:    " << counters.getRotateAcceptance() << "\n";
                }

            // elapsed time
            double cur_time = double(m_clock.getTime()) / Scalar(1e9);
            uint64_t total_moves = counters.getNMoves();
            m_exec_conf->msg->notice(2) << "Trial moves per second:        " << double(total_moves) / cur_time << std::endl;
            m_exec_conf->msg->notice(2) << "Overlap checks per second:     " << double(counters.overlap_checks) / cur_time << std::endl;
            m_exec_conf->msg->notice(2) << "Overlap checks per trial move: " << double(counters.overlap_checks) / double(total_moves) << std::endl;
            m_exec_conf->msg->notice(2) << "Number of overlap errors:      " << double(counters.overlap_err_count) << std::endl;
            }

        //! Get performance in moves per second
        virtual double getMPS()
            {
            hpmc_counters_t counters = getCounters(1);
            double cur_time = double(m_clock.getTime()) / Scalar(1e9);
            return double(counters.getNMoves()) / cur_time;
            }

        //! Reset statistics counters
        virtual void resetStats()
            {
            ArrayHandle<hpmc_counters_t> h_counters(m_count_total, access_location::host, access_mode::read);
            m_count_run_start = h_counters.data[0];
            m_clock = ClockSource();
            }

        //! Get the diameter of the largest circumscribing sphere for objects handled by this integrator
        virtual Scalar getMaxCoreDiameter()
            {
            return 1.0;
            }

        //! Count the number of particle overlaps
        /*! \param timestep current step
            \param early_exit exit at first overlap found if true
            \returns number of overlaps if early_exit=false, 1 if early_exit=true
        */
        virtual unsigned int countOverlaps(unsigned int timestep, bool early_exit)
            {
            return 0;
            }

        //! Get the number of degrees of freedom granted to a given group
        /*! \param group Group over which to count degrees of freedom.
            \return a non-zero dummy value to suppress warnings.

            MC does not integrate with the MD computations that use this value.
        */
        virtual unsigned int getNDOF(std::shared_ptr<ParticleGroup> group)
            {
            return 1;
            }

        //! Get a list of logged quantities
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Get the value of a logged quantity
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Check the particle data for non-normalized orientations
        virtual bool checkParticleOrientations();

        //! Get the current counter values
        hpmc_counters_t getCounters(unsigned int mode=0);

        //! Communicate particles
        /*! \param migrate Set to true to both migrate and exchange, set to false to only exchange

            This method exists so that the python API can force MPI communication when needed, e.g. before a
            count_overlaps call to ensure that particle data is up to date.

            The base class does nothing and leaves derived classes to implement.
        */
        virtual void communicate(bool migrate)
            {
            }

        //! Set extra ghost width
        /*! \param extra Extra width to add to the ghost layer

            This method is called by AnalyzerSDF when needed to note that an extra padding on the ghost layer is needed
        */
        void setExtraGhostWidth(Scalar extra)
            {
            m_extra_ghost_width = extra;
            updateCellWidth();

            }
        //! Method to scale the box
        virtual bool attemptBoxResize(unsigned int timestep, const BoxDim& new_box);

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange();

        ExternalField* getExternalField()
            {
            return m_external_base;
            }

        //! Returns the patch energy interaction
        std::shared_ptr<PatchEnergy> getPatchInteraction()
            {
            if (!m_patch_log)
                return m_patch;
            else
                return std::shared_ptr<PatchEnergy>();
            }

        //! Compute the energy due to patch interactions
        /*! \param timestep the current time step
         * \returns the total patch energy
         */
        virtual float computePatchEnergy(unsigned int timestep)
            {
            // base class method returns 0
            return 0.0;
            }

        //! Enable deterministic simulations
        virtual void setDeterministic(bool deterministic) {};

        //! Prepare for the run
        virtual void prepRun(unsigned int timestep)
            {
            m_past_first_run = true;
            }

        //! Set the patch energy
        void setPatchEnergy(std::shared_ptr< PatchEnergy > patch)
            {
            m_patch = patch;
            }

        //! Enable the patch energy only for logging
        /*! \param log if True, only enabled for logging purposes
         */
        void disablePatchEnergyLogOnly(bool log)
            {
            m_patch_log = log;
            }

    protected:
        unsigned int m_seed;                        //!< Random number seed
        unsigned int m_move_ratio;                  //!< Ratio of translation to rotation move attempts (*65535)
        unsigned int m_nselect;                     //!< Number of particles to select for trial moves

        GPUVector<Scalar> m_d;                      //!< Maximum move displacement by type
        GPUVector<Scalar> m_a;                      //!< Maximum angular displacement by type

        GPUArray< hpmc_counters_t > m_count_total;  //!< Accept/reject total count

        Scalar m_nominal_width;                      //!< nominal cell width
        Scalar m_extra_ghost_width;                  //!< extra ghost width to add
        ClockSource m_clock;                           //!< Timer for self-benchmarking

        ExternalField* m_external_base; //! This is a cast of the derived class's m_external that can be used in a more general setting.

        std::shared_ptr< PatchEnergy > m_patch;     //!< Patchy Interaction
        bool m_patch_log;                           //!< If true, only use patch energy for logging

        bool m_past_first_run;                      //!< Flag to test if the first run() has started
        //! Update the nominal width of the cells
        /*! This method is virtual so that derived classes can set appropriate widths
            (for example, some may want max diameter while others may want a buffer distance).
        */
        virtual void updateCellWidth()
            {
            }

        //! Return the requested ghost layer width
        virtual Scalar getGhostLayerWidth(unsigned int)
            {
            return Scalar(0.0);
            }

        #ifdef ENABLE_MPI
        //! Return the requested communication flags for ghost particles
        virtual CommFlags getCommFlags(unsigned int)
            {
            return CommFlags(0);
            }

        //! Set the MPI communicator
        /*! \param comm the communicator
            This method is overridden so that we can register with the signal to set the ghost layer width.
        */
        virtual void setCommunicator(std::shared_ptr<Communicator> comm)
            {
            if (! m_communicator_ghost_width_connected)
                {
                // only add the migrate request on the first call
                assert(comm);
                comm->getGhostLayerWidthRequestSignal().connect<IntegratorHPMC, &IntegratorHPMC::getGhostLayerWidth>(this);
                m_communicator_ghost_width_connected = true;
                }
            if (! m_communicator_flags_connected)
                {
                // only add the migrate request on the first call
                assert(comm);
                comm->getCommFlagsRequestSignal().connect<IntegratorHPMC, &IntegratorHPMC::getCommFlags>(this);
                m_communicator_flags_connected = true;
                }

            // set the member variable
            Integrator::setCommunicator(comm);
            }
        #endif

    private:
        hpmc_counters_t m_count_run_start;             //!< Count saved at run() start
        hpmc_counters_t m_count_step_start;            //!< Count saved at the start of the last step

        #ifdef ENABLE_MPI
        bool m_communicator_ghost_width_connected;     //!< True if we have connected to Communicator's ghost layer width signal
        bool m_communicator_flags_connected;           //!< True if we have connected to Communicator's communication flags signal
        #endif
    };

//! Export the IntegratorHPMC class to python
void export_IntegratorHPMC(pybind11::module& m);

} // end namespace hpmc

#endif // _INTEGRATOR_HPMC_H_
