#ifndef __UPDATER_MUVT_H__
#define __UPDATER_MUVT_H__


#include "hoomd/Updater.h"
#include "hoomd/VectorMath.h"
#include "hoomd/Variant.h"
#include "hoomd/HOOMDMPI.h"

#include "Moves.h"
#include "IntegratorHPMCMono.h"

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

namespace hpmc
{

/*!
 * This class implements an Updater for simulations in the grand-canonical ensemble (mu-V-T).
 *
 * Gibbs ensemble integration between two MPI partitions is also supported.
 */
template<class Shape>
class UpdaterMuVT : public Updater
    {
    public:
        //! Constructor
        UpdaterMuVT(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<IntegratorHPMCMono<Shape> > mc,
            unsigned int seed,
            unsigned int npartition);
        virtual ~UpdaterMuVT();

        //! The entry method for this updater
        /*! \param timestep Current simulation step
         */
        virtual void update(unsigned int timestep);

        //! Set the fugacity of a particle type
        /*! \param type The type id for which to set the fugacity
         * \param fugacity The value of the fugacity (variant)
         */
        void setFugacity(unsigned int type, std::shared_ptr<Variant> fugacity)
            {
            assert(type < m_pdata->getNTypes());
            m_fugacity[type] = fugacity;
            }

        //! Set maximum factor for volume rescaling (Gibbs ensemble only)
        void setMaxVolumeRescale(Scalar fac)
            {
            m_max_vol_rescale = fac;
            }

        //! Set ratio of volume moves to exchange/transfer moves (Gibbs ensemble only)
        void setMoveRatio(Scalar move_ratio)
            {
            if (move_ratio < Scalar(0.0) || move_ratio > Scalar(1.0))
                {
                throw std::runtime_error("Move ratio has to be between 0 and 1.\n");
                }
            m_move_ratio = move_ratio;
            }

        //! Set ratio of transfer moves to exchange moves (Gibbs ensemble only)
        void setTransferRatio(Scalar transfer_ratio)
            {
            if (transfer_ratio < Scalar(0.0) || transfer_ratio > Scalar(1.0))
                {
                throw std::runtime_error("Transfer ratio has to be between 0 and 1.\n");
                }
            m_transfer_ratio = transfer_ratio;
            }

        //! List of types that are inserted/removed/transferred
        void setTransferTypes(std::vector<unsigned int>& transfer_types)
            {
            assert(transfer_types.size() <= m_pdata->getNTypes());
            if (transfer_types.size() == 0)
                {
                throw std::runtime_error("Must transfer at least one type.\n");
                }
            m_transfer_types = transfer_types;
            }


        //! Print statistics about the muVT ensemble
        void printStats()
            {
            hpmc_muvt_counters_t counters = getCounters(1);
            m_exec_conf->msg->notice(2) << "-- HPMC muVT stats:" << std::endl;
            if (counters.insert_accept_count + counters.insert_reject_count > 0)
                {
                m_exec_conf->msg->notice(2) << "Average insert acceptance: " << counters.getInsertAcceptance() << std::endl;
                }
            if (counters.remove_accept_count + counters.remove_reject_count > 0)
                {
                m_exec_conf->msg->notice(2) << "Average remove acceptance: " << counters.getRemoveAcceptance() << "\n";
                }
            if (counters.exchange_accept_count + counters.exchange_reject_count > 0)
                {
                m_exec_conf->msg->notice(2) << "Average exchange acceptance: " << counters.getExchangeAcceptance() << "\n";
                }
            m_exec_conf->msg->notice(2) << "Total transfer/exchange moves attempted: " << counters.getNExchangeMoves() << std::endl;
            if (counters.volume_accept_count + counters.volume_reject_count > 0)
                {
                m_exec_conf->msg->notice(2) << "Average volume acceptance: " << counters.getVolumeAcceptance() << "\n";
                }
            m_exec_conf->msg->notice(2) << "Total volume moves attempted: " << counters.getNVolumeMoves() << std::endl;
            }

        //! Get a list of logged quantities
        virtual std::vector< std::string > getProvidedLogQuantities()
            {
            std::vector< std::string > result;

            result.push_back("hpmc_muvt_insert_acceptance");
            result.push_back("hpmc_muvt_remove_acceptance");
            result.push_back("hpmc_muvt_exchange_acceptance");
            result.push_back("hpmc_muvt_volume_acceptance");

            for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
                {
                result.push_back("hpmc_muvt_N_"+m_pdata->getNameByType(i));
                }
            return result;
            }

        //! Get the value of a logged quantity
        virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep);

        //! Reset statistics counters
        void resetStats()
            {
            m_count_run_start = m_count_total;
            }

        //! Get the current counter values
        hpmc_muvt_counters_t getCounters(unsigned int mode=0);

    protected:
        std::vector<std::shared_ptr<Variant> > m_fugacity;  //!< Reservoir concentration per particle-type
        std::shared_ptr<IntegratorHPMCMono<Shape> > m_mc;   //!< The MC Integrator this Updater is associated with
        unsigned int m_seed;                                  //!< RNG seed
        unsigned int m_npartition;                            //!< The number of partitions to use for Gibbs ensemble
        bool m_gibbs;                                         //!< True if we simulate a Gibbs ensemble

        GPUVector<Scalar4> m_postype_backup;                  //!< Backup of postype array

        Scalar m_max_vol_rescale;                             //!< Maximum volume ratio rescaling factor
        Scalar m_move_ratio;                                  //!< Ratio between exchange/transfer and volume moves
        Scalar m_transfer_ratio;                              //!< Ratio between transfer and exchange moves

        unsigned int m_gibbs_other;                           //!< The root-rank of the other partition

        hpmc_muvt_counters_t m_count_total;          //!< Accept/reject total count
        hpmc_muvt_counters_t m_count_run_start;      //!< Count saved at run() start
        hpmc_muvt_counters_t m_count_step_start;     //!< Count saved at the start of the last step

        std::vector<std::vector<unsigned int> > m_type_map;   //!< Local list of particle tags per type
        std::vector<unsigned int> m_transfer_types;  //!< List of types being insert/removed/transferred between boxes

        GPUVector<Scalar4> m_pos_backup;             //!< Backup of particle positions for volume move
        GPUVector<Scalar4> m_orientation_backup;     //!< Backup of particle orientations for volume move
        GPUVector<Scalar> m_charge_backup;           //!< Backup of particle charges for volume move
        GPUVector<Scalar> m_diameter_backup;         //!< Backup of particle diameters for volume move

        /*! Check for overlaps of a fictitious particle
         * \param timestep Current time step
         * \param type Type of particle to test
         * \param pos Position of fictitious particle
         * \param orientation Orientation of particle
         * \param lnboltzmann Log of Boltzmann weight of insertion attempt (return value)
         * \returns True if boltzmann weight is non-zero
         */
        virtual bool tryInsertParticle(unsigned int timestep, unsigned int type, vec3<Scalar> pos, quat<Scalar> orientation,
            Scalar &lnboltzmann);

        /*! Try removing a particle
            \param timestep Current time step
            \param tag Tag of particle being removed
            \param lnboltzmann Log of Boltzmann weight of removal attempt (return value)
            \returns True if boltzmann weight is non-zero
         */
        virtual bool tryRemoveParticle(unsigned int timestep, unsigned int tag, Scalar &lnboltzmann);

        /*! Try switching particle type
         * \param timestep Current time step
         * \param tag Tag of particle that is considered for switching types
         * \param newtype New type of particle
         * \param lnboltzmann Log of Boltzmann weight of removal attempt (return value)
         * \returns True if boltzmann weight is non-zero
         *
         * \note The method has to check that getNGlobal() > 0, otherwise tag is invalid
         */
        virtual bool trySwitchType(unsigned int timestep, unsigned int tag, unsigned newtype, Scalar &lnboltzmann);

        /*! Rescale box to new dimensions and scale particles
         * \param timestep current timestep
         * \param new_box the old BoxDim
         * \param new_box the new BoxDim
         * \param extra_ndof (return value) extra degrees of freedom added before box resize
         * \param lnboltzmann (return value) exponent of Boltzmann factor (-delta_E)
         * \returns true if no overlaps
         */
        virtual bool boxResizeAndScale(unsigned int timestep, const BoxDim old_box, const BoxDim new_box,
            unsigned int &extra_ndof, Scalar &lnboltzmann);

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange();

        //! Map particles by type
        virtual void mapTypes();

        //! Get the nth particle of a given type
        /*! \param type the requested type of the particle
         *  \param type_offs offset of the particle in the list of particles per type
         */
        virtual unsigned int getNthTypeTag(unsigned int type, unsigned int type_offs);

        //! Get number of particles of a given type
        unsigned int getNumParticlesType(unsigned int type);

    private:
        //! Handle MaxParticleNumberChange signal
        /*! Resize the m_pos_backup array
        */
        void slotMaxNChange()
            {
            unsigned int MaxN = m_pdata->getMaxN();
            m_pos_backup.resize(MaxN);
            }
    };

//! Export the UpdaterMuVT class to python
/*! \param name Name of the class in the exported python module
    \tparam Shape An instantiation of UpdaterMuVT<Shape> will be exported
*/
template < class Shape > void export_UpdaterMuVT(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterMuVT<Shape>, std::shared_ptr< UpdaterMuVT<Shape> > >(m, name.c_str(), pybind11::base<Updater>())
          .def( pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr< IntegratorHPMCMono<Shape> >, unsigned int, unsigned int>())
          .def("setFugacity", &UpdaterMuVT<Shape>::setFugacity)
          .def("setMaxVolumeRescale", &UpdaterMuVT<Shape>::setMaxVolumeRescale)
          .def("setMoveRatio", &UpdaterMuVT<Shape>::setMoveRatio)
          .def("setTransferRatio", &UpdaterMuVT<Shape>::setTransferRatio)
          .def("setTransferTypes", &UpdaterMuVT<Shape>::setTransferTypes)
          ;
    }

/*! Constructor
    \param sysdef The system definition
    \param mc The HPMC integrator
    \param seed RNG seed
    \param npartition How many partitions to use in parallel for Gibbs ensemble (n=1 == grand canonical)
 */
template<class Shape>
UpdaterMuVT<Shape>::UpdaterMuVT(std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<IntegratorHPMCMono< Shape > > mc,
    unsigned int seed,
    unsigned int npartition)
    : Updater(sysdef), m_mc(mc), m_seed(seed), m_npartition(npartition), m_gibbs(false),
      m_max_vol_rescale(0.1), m_move_ratio(0.5), m_transfer_ratio(1.0), m_gibbs_other(0)
    {
    // broadcast the seed from rank 0 to all other ranks.
    #ifdef ENABLE_MPI
        if(this->m_pdata->getDomainDecomposition())
            bcast(m_seed, 0, this->m_exec_conf->getMPICommunicator());
    #endif

    m_fugacity.resize(m_pdata->getNTypes(), std::shared_ptr<Variant>(new VariantConst(0.0)));
    m_type_map.resize(m_pdata->getNTypes());

    m_pdata->getNumTypesChangeSignal().template connect<UpdaterMuVT<Shape>, &UpdaterMuVT<Shape>::slotNumTypesChange>(this);
    m_pdata->getParticleSortSignal().template connect<UpdaterMuVT<Shape>, &UpdaterMuVT<Shape>::mapTypes>(this);

    if (npartition > 1)
        {
        m_gibbs = true;
        }

    #ifdef ENABLE_MPI
    if (m_gibbs)
        {
        if (m_exec_conf->getNPartitions() % npartition)
            {
            m_exec_conf->msg->error() << "Total number of partitions not a multiple of number "
                << "of Gibbs ensemble partitions." << std::endl;
            throw std::runtime_error("Error setting up Gibbs ensemble integration.");
            }

        GPUVector<Scalar4> postype_backup(m_exec_conf);
        m_postype_backup.swap(postype_backup);

        m_exec_conf->msg->notice(5) << "Constructing UpdaterMuVT: Gibbs ensemble with "
            << m_npartition << " partitions" << std::endl;
        }
    else
    #endif
        {
        m_exec_conf->msg->notice(5) << "Constructing UpdaterMuVT" << std::endl;
        }

    #ifndef ENABLE_MPI
    if (m_gibbs)
        {
        throw std::runtime_error("Gibbs ensemble integration only supported with MPI.");
        }
    #endif

    // initialize list of tags per type
    mapTypes();

    // Connect to the MaxParticleNumberChange signal
    m_pdata->getMaxParticleNumberChangeSignal().template connect<UpdaterMuVT<Shape>, &UpdaterMuVT<Shape>::slotMaxNChange>(this);
    }

//! Destructor
template<class Shape>
UpdaterMuVT<Shape>::~UpdaterMuVT()
    {
    m_pdata->getNumTypesChangeSignal().template disconnect<UpdaterMuVT<Shape>, &UpdaterMuVT<Shape>::slotNumTypesChange>(this);
    m_pdata->getParticleSortSignal().template disconnect<UpdaterMuVT<Shape>, &UpdaterMuVT<Shape>::mapTypes>(this);
    m_pdata->getMaxParticleNumberChangeSignal().template disconnect<UpdaterMuVT<Shape>, &UpdaterMuVT<Shape>::slotMaxNChange>(this);
    }

template<class Shape>
void UpdaterMuVT<Shape>::mapTypes()
    {
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    assert(m_pdata->getNTypes() == m_type_map.size());
    for (unsigned int itype = 0; itype < m_pdata->getNTypes(); ++itype)
        {
        m_type_map[itype].clear();
        }

    unsigned int nptl = m_pdata->getN();
    for (unsigned int idx = 0; idx < nptl; idx++)
        {
        unsigned int typei = __scalar_as_int(h_postype.data[idx].w);
        unsigned int tag = h_tag.data[idx];

        // store tag in per-type list
        assert(m_type_map.size() > typei);
        m_type_map[typei].push_back(tag);
        }
    }

template<class Shape>
unsigned int UpdaterMuVT<Shape>::getNthTypeTag(unsigned int type, unsigned int type_offs)
    {
    unsigned int tag = UINT_MAX;

    assert(m_type_map.size() > type);
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // get number of particles of given type
        unsigned int nptl = m_type_map[type].size();

        // have to initialize correctly for prefix sum
        unsigned int begin_offs=0;
        unsigned int end_offs=0;

        // exclusive scan
        MPI_Exscan(&nptl, &begin_offs, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());

        // inclusive scan
        MPI_Scan(&nptl, &end_offs, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());

        bool is_local = type_offs >= begin_offs && type_offs < end_offs;

        unsigned int rank = is_local ? m_exec_conf->getRank() : 0;

        MPI_Allreduce(MPI_IN_PLACE, &rank, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        assert(rank == m_exec_conf->getRank() || !is_local);

        // broadcast the chosen particle tag
        if (is_local)
            {
            assert(type_offs - begin_offs < m_type_map[type].size());
            tag = m_type_map[type][type_offs - begin_offs];
            }

        MPI_Bcast(&tag, 1, MPI_UNSIGNED, rank, m_exec_conf->getMPICommunicator());
        }
    else
    #endif
        {
        assert(type_offs < m_type_map[type].size());
        tag = m_type_map[type][type_offs];
        }

    assert(tag <= m_pdata->getMaximumTag());
    return tag;
    }

template<class Shape>
unsigned int UpdaterMuVT<Shape>::getNumParticlesType(unsigned int type)
    {
    assert(type < m_type_map.size());
    unsigned int nptl_type = m_type_map[type].size();

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &nptl_type, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif
    return nptl_type;
    }

//! Destructor
template<class Shape>
void UpdaterMuVT<Shape>::slotNumTypesChange()
    {
    // resize parameter list
    m_fugacity.resize(m_pdata->getNTypes(), std::shared_ptr<Variant>(new VariantConst(0.0)));
    m_type_map.resize(m_pdata->getNTypes());
    }

/*! Set new box and scale positions
*/
template<class Shape>
bool UpdaterMuVT<Shape>::boxResizeAndScale(unsigned int timestep, const BoxDim old_box, const BoxDim new_box,
    unsigned int &extra_ndof, Scalar& lnboltzmann)
    {
    lnboltzmann = Scalar(0.0);

    unsigned int N_old = m_pdata->getN();

    extra_ndof = 0;

    auto patch = m_mc->getPatchInteraction();

    if (patch)
        {
        // energy of old configuration
        lnboltzmann += m_mc->computePatchEnergy(timestep);
        }

        {
        // Get particle positions
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

        // move the particles to be inside the new box
        for (unsigned int i = 0; i < N_old; i++)
            {
            Scalar3 old_pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);

            // obtain scaled coordinates in the old global box
            Scalar3 f = old_box.makeFraction(old_pos);

            // scale particles
            Scalar3 scaled_pos = new_box.makeCoordinates(f);
            h_pos.data[i].x = scaled_pos.x;
            h_pos.data[i].y = scaled_pos.y;
            h_pos.data[i].z = scaled_pos.z;
            }
        } // end lexical scope

    m_pdata->setGlobalBox(new_box);

    // we have changed particle neighbors, communicate those changes
    m_mc->communicate(false);

    // check for overlaps
    bool overlap = m_mc->countOverlaps(timestep, true);

    if (!overlap && patch)
        {
        // energy of new configuration
        lnboltzmann -= m_mc->computePatchEnergy(timestep);
        }

    return !overlap;
    }

template<class Shape>
void UpdaterMuVT<Shape>::update(unsigned int timestep)
    {
    m_count_step_start = m_count_total;

    if (m_prof) m_prof->push("update muVT");

    m_exec_conf->msg->notice(10) << "UpdaterMuVT update: " << timestep << std::endl;

    // initialize random number generator
    #ifdef ENABLE_MPI
    unsigned int group = (m_exec_conf->getPartition()/m_npartition);
    #else
    unsigned int group = 0;
    #endif

    hoomd::detail::Saru rng(hoomd::RNGIdentifier::UpdaterMuVT, this->m_seed, timestep, group);

    bool active = true;
    unsigned int mod = 0;

    bool volume_move = false;

    bool is_root = (m_exec_conf->getRank() == 0);

    #ifdef ENABLE_MPI
    unsigned int src = 0;
    unsigned int dest = 1;

    // the other MPI partition
    if (m_gibbs)
        {
        unsigned int p = m_exec_conf->getPartition() % m_npartition;

        // choose a random pair of communicating boxes
        src = rand_select(rng, m_npartition-1);
        dest  = rand_select(rng, m_npartition-2);
        if (src <= dest)
            dest++;

        if (p==src)
            {
            m_gibbs_other = (dest+group*m_npartition)*m_exec_conf->getNRanks();
            mod = 0;
            }
        if (p==dest)
            {
            m_gibbs_other = (src+group*m_npartition)*m_exec_conf->getNRanks();
            mod = 1;
            }
        if (p != src && p!= dest)
            {
            active = false;
            }

        // order the expanded ensembles
        volume_move = (rng.f() < m_move_ratio);

        if (active && m_exec_conf->getRank() == 0)
            {
            unsigned int other_timestep = 0;
            // make sure random seeds are equal
            if (mod == 0)
                {
                MPI_Status stat;
                MPI_Recv(&other_timestep, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                MPI_Send(&timestep, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                }
            else
                {
                MPI_Status stat;
                MPI_Send(&timestep, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                MPI_Recv(&other_timestep, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                }

            if (other_timestep != timestep)
                {
                m_exec_conf->msg->error() << "UpdaterMuVT: Boxes are at different time steps " << timestep << " != " << other_timestep << ". Aborting."
                    << std::endl;
                throw std::runtime_error("Error in update.muvt.");
                }
            }
        }
    #endif

    if (active && !volume_move)
        {
        bool transfer_move = (rng.f() <= m_transfer_ratio);

        if (transfer_move)
            {
            #ifdef ENABLE_MPI
            if (m_gibbs)
                {
                m_exec_conf->msg->notice(10) << "UpdaterMuVT: Gibbs ensemble transfer " << src << "->" << dest << " " << timestep
                    << " (Gibbs ensemble partition " << m_exec_conf->getPartition() % m_npartition << ")" << std::endl;
                }
            #endif

            // whether we insert or remove a particle
            bool insert = m_gibbs ? mod : rand_select(rng,1);

            if (insert)
                {
                // Try inserting a particle
                unsigned int type = 0;
                std::string type_name;
                Scalar lnboltzmann(0.0);

                unsigned int nptl_type = 0;

                Scalar V = m_pdata->getGlobalBox().getVolume();

                assert(m_transfer_types.size() > 0);

                if (! m_gibbs)
                    {
                    // choose a random particle type out of those being inserted or removed
                    type = m_transfer_types[rand_select(rng, m_transfer_types.size()-1)];
                    }
                else
                    {
                    if (is_root)
                        {
                        #ifdef ENABLE_MPI
                        MPI_Status stat;

                        // receive type of particle
                        unsigned int n;
                        MPI_Recv(&n, 1, MPI_UNSIGNED, m_gibbs_other,0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                        char s[n];
                        MPI_Recv(s, n, MPI_CHAR, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                        type_name = std::string(s);

                        // resolve type name
                        type = m_pdata->getTypeByName(type_name);
                        #endif
                        }

                    #ifdef ENABLE_MPI
                    if (m_comm)
                        {
                        bcast(type,0,m_exec_conf->getMPICommunicator());
                        }
                    #endif
                    }

                // number of particles of that type
                nptl_type = getNumParticlesType(type);

                    {
                    const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc->getParams();
                    const typename Shape::param_type& param = params[type];

                    // Propose a random position uniformly in the box
                    Scalar3 f;
                    f.x = rng.template s<Scalar>();
                    f.y = rng.template s<Scalar>();
                    if (m_sysdef->getNDimensions() == 2)
                        {
                        f.z = Scalar(0.5);
                        }
                    else
                        {
                        f.z = rng.template s<Scalar>();
                        }
                    vec3<Scalar> pos_test = vec3<Scalar>(m_pdata->getGlobalBox().makeCoordinates(f));

                    Shape shape_test(quat<Scalar>(), param);
                    if (shape_test.hasOrientation())
                        {
                        // set particle orientation
                        if (m_sysdef->getNDimensions() == 2)
                            {
                            shape_test.orientation = generateRandomOrientation2D(rng);
                            }
                        else
                            {
                            shape_test.orientation = generateRandomOrientation(rng);
                            }
                        }

                    if (m_gibbs)
                        {
                        // acceptance probability
                        lnboltzmann = log((Scalar)V/(Scalar)(nptl_type+1));
                        }
                    else
                        {
                        // get fugacity value
                        Scalar fugacity = m_fugacity[type]->getValue(timestep);

                        // sanity check
                        if (fugacity <= Scalar(0.0))
                            {
                            m_exec_conf->msg->error() << "Fugacity has to be greater than zero." << std::endl;
                            throw std::runtime_error("Error in UpdaterMuVT");
                            }

                        // acceptance probability
                        lnboltzmann = log(fugacity*V/(Scalar)(nptl_type+1));
                        }

                    // check if particle can be inserted without overlaps
                    Scalar lnb(0.0);
                    unsigned int nonzero = tryInsertParticle(timestep, type, pos_test, shape_test.orientation, lnb);

                    if (nonzero)
                        {
                        lnboltzmann += lnb;
                        }

                    #ifdef ENABLE_MPI
                    if (m_gibbs && is_root)
                        {
                        // receive Boltzmann factor for removal from other rank
                        MPI_Status stat;
                        Scalar remove_lnb;
                        unsigned int remove_nonzero;
                        MPI_Recv(&remove_lnb, 1, MPI_HOOMD_SCALAR, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                        MPI_Recv(&remove_nonzero, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);

                        // avoid divide/multiply by infinity
                        if (remove_nonzero)
                            {
                            lnboltzmann += remove_lnb;
                            }
                        else
                            {
                            nonzero = 0;
                            }
                        }

                    if (m_comm)
                        {
                        bcast(lnboltzmann, 0, m_exec_conf->getMPICommunicator());
                        bcast(nonzero, 0, m_exec_conf->getMPICommunicator());
                        }
                    #endif

                    // apply acceptance criterion
                    bool accept = false;
                    if (nonzero)
                        {
                        accept = (rng.template s<Scalar>() < exp(lnboltzmann));
                        }

                    #ifdef ENABLE_MPI
                    if (m_gibbs && is_root)
                        {
                        // send result of acceptance test
                        unsigned result = accept;
                        MPI_Send(&result, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                        }
                    #endif

                    if (accept)
                        {
                        // insertion was successful

                        // create a new particle with given type
                        unsigned int tag;

                        tag = m_pdata->addParticle(type);

                        // set the position of the particle

                        // setPosition() takes into account the grid shift, so subtract that one
                        Scalar3 p = vec_to_scalar3(pos_test)-m_pdata->getOrigin();
                        int3 tmp = make_int3(0,0,0);
                        m_pdata->getGlobalBox().wrap(p,tmp);
                        m_pdata->setPosition(tag, p);
                        if (shape_test.hasOrientation())
                            {
                            m_pdata->setOrientation(tag, quat_to_scalar4(shape_test.orientation));
                            }
                        m_count_total.insert_accept_count++;
                        }
                    else
                        {
                        m_count_total.insert_reject_count++;
                        }
                    }
                }
            else
                {
                // try removing a particle
                unsigned int tag = UINT_MAX;

                // in Gibbs ensemble, we should not use correlated random numbers with box 1
                hoomd::detail::Saru rng_local(hoomd::RNGIdentifier::UpdaterMuVTBox1, this->m_seed, timestep, group);

                // choose a random particle type out of those being transferred
                assert(m_transfer_types.size() > 0);
                unsigned int type = m_transfer_types[rand_select(rng_local, m_transfer_types.size()-1)];

                // choose a random particle of that type
                unsigned int nptl_type = getNumParticlesType(type);

                if (nptl_type)
                    {
                    // get random tag of given type
                    unsigned int type_offset = rand_select(rng_local, nptl_type-1);
                    tag = getNthTypeTag(type, type_offset);
                    }

                Scalar V = m_pdata->getGlobalBox().getVolume();
                Scalar lnboltzmann(0.0);

                if (!m_gibbs)
                    {
                    // get fugacity value
                    Scalar fugacity = m_fugacity[type]->getValue(timestep);

                    // sanity check
                    if (fugacity <= Scalar(0.0))
                        {
                        m_exec_conf->msg->error() << "Fugacity has to be greater than zero." << std::endl;
                        throw std::runtime_error("Error in UpdaterMuVT");
                        }

                    lnboltzmann -= log(fugacity);
                    }
                else
                    {
                    if (is_root)
                        {
                        #ifdef ENABLE_MPI
                        // determine type name
                        std::string type_name = m_pdata->getNameByType(type);

                        // send particle type to other rank
                        unsigned int n = type_name.size()+1;
                        MPI_Send(&n, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                        char s[n];
                        memcpy(s,type_name.c_str(),n);
                        MPI_Send(s, n, MPI_CHAR, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                        #endif
                        }
                    }

                // acceptance probability
                unsigned int nonzero = 1;
                if (nptl_type)
                    {
                    lnboltzmann += log((Scalar)nptl_type/V);
                    }
                else
                    {
                    nonzero = 0;
                    }

                bool accept = true;

                // get weight for removal
                Scalar lnb(0.0);
                if (tryRemoveParticle(timestep, tag, lnb))
                    {
                    lnboltzmann += lnb;
                    }
                else
                    {
                    nonzero = 0;
                    }

                if (m_gibbs)
                    {
                    if (is_root)
                        {
                        #ifdef ENABLE_MPI
                        // send result of removal attempt
                        MPI_Send(&lnboltzmann, 1, MPI_HOOMD_SCALAR, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                        MPI_Send(&nonzero, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());

                        // wait for result of insertion on other rank
                        unsigned int result;
                        MPI_Status stat;
                        MPI_Recv(&result, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                        accept = result;
                        #endif
                        }
                    }
                else
                    {
                    // apply acceptance criterion
                    if (nonzero)
                        {
                        accept  = (rng_local.d() < exp(lnboltzmann));
                        }
                    else
                        {
                        accept = false;
                        }
                    }

                #ifdef ENABLE_MPI
                if (m_gibbs && m_comm)
                    {
                    bcast(accept,0,m_exec_conf->getMPICommunicator());
                    }
                #endif

                if (accept)
                    {
                    // remove particle
                    m_pdata->removeParticle(tag);
                    m_count_total.remove_accept_count++;
                    }
                else
                    {
                    m_count_total.remove_reject_count++;
                    }
                } // end remove particle
            } // end transfer move
        else
            {
            // exchange move, try changing identity of a particle to a different one
            #ifdef ENABLE_MPI
            if (m_gibbs)
                {
                m_exec_conf->msg->notice(10) << "UpdaterMuVT: Gibbs ensemble exchange " << src << "<->" << dest << " " << timestep
                    << " (Gibbs ensemble partition " << m_exec_conf->getPartition() % m_npartition << ")" << std::endl;
                }
            #endif
            if (m_pdata->getNTypes() > 1)
                {
                if (!mod)
                    {
                    // master

                    // fugacity not support for now
                    if (! m_gibbs)
                        {
                        throw std::runtime_error("Particle identity changes only supported in Gibbs ensemble.");
                        }

                    // select a particle type at random
                    unsigned int type = rand_select(rng, m_pdata->getNTypes()-1);

                    // select another type
                    unsigned int other_type = rand_select(rng, m_pdata->getNTypes()-2);

                    if (type <= other_type)
                        other_type++;

                    #ifdef ENABLE_MPI
                    if (m_gibbs && is_root)
                        {
                        // communicate type pair to other box
                        MPI_Send(&type, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                        MPI_Send(&other_type, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                        }
                    #endif

                    // get number of particles of both types
                    unsigned int N_old = getNumParticlesType(type);
                    unsigned int N_new = getNumParticlesType(other_type);

                    unsigned int nonzero = 1;
                    Scalar lnboltzmann(0.0);

                    if (N_old > 0)
                        {
                        lnboltzmann = (Scalar)(N_new+1)/(Scalar)(N_old);
                        }
                    else
                        {
                        nonzero = 0;
                        }

                    unsigned int tag = UINT_MAX;

                    if (nonzero)
                        {
                        // select a random particle tag of given type
                        unsigned int type_offs = rand_select(rng, N_old-1);
                        tag = getNthTypeTag(type, type_offs);

                        Scalar lnb(0.0);

                        // try changing particle identity
                        if (trySwitchType(timestep, tag, other_type, lnb))
                            {
                            lnboltzmann += lnb;
                            }
                        else
                            {
                            nonzero = 0;
                            }
                        }

                    #ifdef ENABLE_MPI
                    if (m_gibbs)
                        {
                        unsigned int other_nonzero = 0;
                        Scalar lnb;
                        if (is_root)
                            {
                            // receive result of identity change from other box
                            MPI_Status stat;
                            MPI_Recv(&other_nonzero, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                            MPI_Recv(&lnb, 1, MPI_HOOMD_SCALAR, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                            }
                        if (m_pdata->getDomainDecomposition())
                            {
                            MPI_Bcast(&other_nonzero, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
                            MPI_Bcast(&lnb, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
                            }
                        if (other_nonzero)
                            {
                            lnboltzmann += lnb;
                            }
                        else
                            {
                            nonzero = 0;
                            }
                        }
                    #endif

                    unsigned int accept = 0;
                    if (nonzero)
                        {
                        // apply acceptance criterion
                        accept = rng.f() < exp(lnboltzmann);
                        }

                    #ifdef ENABLE_MPI
                    if (m_gibbs && is_root)
                        {
                        // communicate result to other box
                        MPI_Send(&accept, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                        }
                    #endif

                    if (accept)
                        {
                        // update the type
                        m_pdata->setType(tag, other_type);

                        // we have changed types, notify particle data
                        m_pdata->notifyParticleSort();

                        m_count_total.exchange_accept_count++;
                        }
                    else
                        {
                        m_count_total.exchange_reject_count++;
                        }
                    }
                else
                    {
                    // slave
                    assert(m_gibbs);

                    unsigned int type=0, other_type=0;
                    #ifdef ENABLE_MPI
                    if (is_root)
                        {
                        MPI_Status stat;
                        MPI_Recv(&type, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                        MPI_Recv(&other_type, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                        }

                    if (m_pdata->getDomainDecomposition())
                        {
                        MPI_Bcast(&type, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
                        MPI_Bcast(&other_type, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
                        }
                    #endif

                    // get number of particles of both types
                    unsigned int N_old = getNumParticlesType(other_type);
                    unsigned int N_new = getNumParticlesType(type);

                    unsigned int nonzero = 1;
                    Scalar lnboltzmann(0.0);

                    if (N_old > 0)
                        {
                        lnboltzmann = (Scalar)(N_new+1)/(Scalar)(N_old);
                        }
                    else
                        {
                        nonzero = 0;
                        }

                    unsigned int tag = UINT_MAX;

                    if (nonzero)
                        {
                        // select a random particle tag of given type

                        // make sure we are not using the same random numbers as box 1
                        hoomd::detail::Saru rng_local(hoomd::RNGIdentifier::UpdaterMuVTBox2, this->m_seed, timestep, group);

                        unsigned int type_offs = rand_select(rng_local, N_old-1);
                        tag = getNthTypeTag(other_type, type_offs);

                        Scalar lnb(0.0);

                        // try changing particle identity
                        if (trySwitchType(timestep, tag, type, lnb))
                            {
                            lnboltzmann += lnb;
                            }
                        else
                            {
                            nonzero = 0;
                            }
                        }

                    unsigned int accept = 0;

                    #ifdef ENABLE_MPI
                    if (is_root)
                        {
                        MPI_Send(&nonzero, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                        MPI_Send(&lnboltzmann, 1, MPI_HOOMD_SCALAR, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());

                        // receive result of decision from other box
                        MPI_Status stat;
                        MPI_Recv(&accept, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                        }

                    if (m_pdata->getDomainDecomposition())
                        {
                        MPI_Bcast(&accept, 1, MPI_UNSIGNED, 0, m_exec_conf->getMPICommunicator());
                        }
                    #endif

                    if (accept)
                        {
                        // update the type
                        m_pdata->setType(tag, type);

                        // we have changed types, notify particle data
                        m_pdata->notifyParticleSort();

                        m_count_total.exchange_accept_count++;
                        }
                    else
                        {
                        m_count_total.exchange_reject_count++;
                        }
                    }
                } // ntypes > 1
            } // !transfer
        }
    #ifdef ENABLE_MPI
    if (active && volume_move)
        {
        if (m_gibbs)
            {
            m_exec_conf->msg->notice(10) << "UpdaterMuVT: Gibbs ensemble volume move " << timestep << std::endl;
            }

        // perform volume move

        Scalar V_other = 0;
        const BoxDim global_box_old = m_pdata->getGlobalBox();
        Scalar V = global_box_old.getVolume();
        unsigned int nglobal = m_pdata->getNGlobal();

        Scalar V_new,V_new_other;
        if (is_root)
            {
            if (mod == 0)
                {
                // send volume to other rank
                MPI_Send(&V, 1, MPI_HOOMD_SCALAR, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());

                MPI_Status stat;

                // receive other box volume
                MPI_Recv(&V_other, 1, MPI_HOOMD_SCALAR, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                }
            else
                {
                // receive other box volume
                MPI_Status stat;
                MPI_Recv(&V_other, 1, MPI_HOOMD_SCALAR, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);

                // send volume to other rank
                MPI_Send(&V, 1, MPI_HOOMD_SCALAR, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                }

            if (mod == 0)
                {
                Scalar ln_V_new = log(V/V_other)+(rng.template s<Scalar>()-Scalar(0.5))*m_max_vol_rescale;
                V_new = (V+V_other)*exp(ln_V_new)/(Scalar(1.0)+exp(ln_V_new));
                V_new_other = (V+V_other)*(Scalar(1.0)-exp(ln_V_new)/(Scalar(1.0)+exp(ln_V_new)));
                }
             else
                {
                Scalar ln_V_new = log(V_other/V)+(rng.template s<Scalar>()-Scalar(0.5))*m_max_vol_rescale;
                V_new = (V+V_other)*(Scalar(1.0)-exp(ln_V_new)/(Scalar(1.0)+exp(ln_V_new)));
                }
            }

        if (m_comm)
            {
            bcast(V_new,0,m_exec_conf->getMPICommunicator());
            }

        // apply volume rescale to box
        BoxDim global_box_new = m_pdata->getGlobalBox();
        Scalar3 L_old = global_box_new.getL();
        Scalar3 L_new = global_box_new.getL();
        Scalar power(0.0);
        if (m_sysdef->getNDimensions() == 2)
            {
            power = Scalar(1.0/2.0);
            }
        else
            {
            power = Scalar(1.0/3.0);
            }
        L_new = L_old * pow(V_new/V,power);
        global_box_new.setL(L_new);

        m_postype_backup.resize(m_pdata->getN());

        // Make a backup copy of position data
        unsigned int N_backup = m_pdata->getN();
            {
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_postype_backup(m_postype_backup, access_location::host, access_mode::readwrite);
            memcpy(h_postype_backup.data, h_postype.data, sizeof(Scalar4) * N_backup);
            }

        //  number of degrees of freedom the old volume (it doesn't change during a volume move)
        unsigned int ndof = nglobal;

        unsigned int extra_ndof = 0;

        // set new box and rescale coordinates
        Scalar lnb(0.0);
        bool has_overlaps = !boxResizeAndScale(timestep, global_box_old, global_box_new, extra_ndof, lnb);
        ndof += extra_ndof;

        unsigned int other_result;
        Scalar other_lnb;

        if (is_root)
            {
            if (mod == 0)
                {
                // receive result from other rank
                MPI_Status stat;
                MPI_Recv(&other_result, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                MPI_Recv(&other_lnb, 1, MPI_HOOMD_SCALAR, m_gibbs_other, 1, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                }
            else
                {
                // send result to other rank
                unsigned int result = has_overlaps;
                MPI_Send(&result, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                MPI_Send(&lnb, 1, MPI_HOOMD_SCALAR, m_gibbs_other, 1, m_exec_conf->getHOOMDWorldMPICommunicator());
                }
            }

        bool accept = true;

        if (is_root)
            {
            if (mod == 0)
                {
                // receive number of particles from other rank
                unsigned int other_ndof;
                MPI_Status stat;
                MPI_Recv(&other_ndof, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);

                // apply criterion on rank zero
                Scalar arg = log(V_new/V)*(Scalar)(ndof+1)+log(V_new_other/V_other)*(Scalar)(other_ndof+1)
                    + lnb + other_lnb;

                accept = rng.f() < exp(arg);
                accept &= !(has_overlaps || other_result);

                // communicate if accepted
                unsigned result = accept;
                MPI_Send(&result, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());
                }
            else
                {
                // send number of particles
                MPI_Send(&ndof, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator());

                // wait for result of acceptance criterion
                MPI_Status stat;
                unsigned int result;
                MPI_Recv(&result, 1, MPI_UNSIGNED, m_gibbs_other, 0, m_exec_conf->getHOOMDWorldMPICommunicator(), &stat);
                accept = result;
                }
            }

        if (m_comm)
            {
            bcast(accept,0,m_exec_conf->getMPICommunicator());
            }

        if (! accept)
            {
            // volume move rejected

            // restore particle positions and orientations
                {
                ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
                ArrayHandle<Scalar4> h_postype_backup(m_postype_backup, access_location::host, access_mode::read);
                unsigned int N = m_pdata->getN();
                if (N != N_backup)
                    {
                    this->m_exec_conf->msg->error() << "update.muvt" << ": Number of particles mismatch when rejecting volume move" << std::endl;
                    throw std::runtime_error("Error resizing box");
                    // note, this error should never appear (because particles are not migrated after a box resize),
                    // but is left here as a sanity check
                    }
                memcpy(h_postype.data, h_postype_backup.data, sizeof(Scalar4) * N);
                }

            m_pdata->setGlobalBox(global_box_old);

            // increment counter
            m_count_total.volume_reject_count++;
            }
        else
            {
            // volume move accepted
            m_count_total.volume_accept_count++;
            }
        } // end volume move
    #endif

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // We have inserted or removed particles or changed box volume, so update ghosts
        m_mc->communicate(false);
        }
    #endif

    if (m_prof) m_prof->pop();
    }

template<class Shape>
bool UpdaterMuVT<Shape>::tryRemoveParticle(unsigned int timestep, unsigned int tag, Scalar &lnboltzmann)
    {
    lnboltzmann = Scalar(0.0);

    // guard against trying to modify empty particle data
    if (tag == UINT_MAX) return false;

    bool is_local = this->m_pdata->isParticleLocal(tag);

    // do we have to compute energetic contribution?
    auto patch = m_mc->getPatchInteraction();

    // if not, no overlaps generated, return happily
    if (!patch) return true;

    // type
    unsigned int type = this->m_pdata->getType(tag);

    // read in the current position and orientation
    quat<Scalar> orientation(m_pdata->getOrientation(tag));

    // charge and diameter
    Scalar diameter = m_pdata->getDiameter(tag);
    Scalar charge = m_pdata->getCharge(tag);

    // getPosition() takes into account grid shift, correct for that
    Scalar3 p = m_pdata->getPosition(tag)+m_pdata->getOrigin();
    int3 tmp = make_int3(0,0,0);
    m_pdata->getGlobalBox().wrap(p,tmp);
    vec3<Scalar> pos(p);

    if (is_local)
        {
        // update the aabb tree
        const detail::AABBTree& aabb_tree = m_mc->buildAABBTree();

        // update the image list
        const std::vector<vec3<Scalar> >&image_list = m_mc->updateImageList();

        // check for overlaps
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
        ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

        // Check particle against AABB tree for neighbors
        Scalar r_cut_patch = patch->getRCut() + 0.5*patch->getAdditiveCutoff(type);

        OverlapReal R_query = std::max(0.0,r_cut_patch - m_mc->getMinCoreDiameter()/(OverlapReal)2.0);
        detail::AABB aabb_local = detail::AABB(vec3<Scalar>(0,0,0),R_query);

        Scalar r_cut_self = r_cut_patch + 0.5*patch->getAdditiveCutoff(type);

        const unsigned int n_images = image_list.size();
        for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
            {
            vec3<Scalar> pos_image = pos + image_list[cur_image];

            if (cur_image != 0)
                {
                vec3<Scalar> r_ij = pos - pos_image;
                // self-energy
                if (dot(r_ij,r_ij) <= r_cut_self*r_cut_self)
                    {
                    lnboltzmann += patch->energy(r_ij,
                        type,
                        quat<float>(orientation),
                        diameter,
                        charge,
                        type,
                        quat<float>(orientation),
                        diameter,
                        charge);
                    }
                }

            detail::AABB aabb = aabb_local;
            aabb.translate(pos_image);

            // stackless search
            for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                {
                if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                    {
                    if (aabb_tree.isNodeLeaf(cur_node_idx))
                        {
                        for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                            {
                            // read in its position and orientation
                            unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                            Scalar4 postype_j = h_postype.data[j];
                            Scalar4 orientation_j = h_orientation.data[j];

                            // put particles in coordinate system of particle i
                            vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_image;

                            unsigned int typ_j = __scalar_as_int(postype_j.w);

                            // we computed the self-interaction above
                            if (h_tag.data[j] == tag) continue;

                            Scalar r_cut_ij = r_cut_patch + 0.5*patch->getAdditiveCutoff(typ_j);

                            if (dot(r_ij,r_ij) <= r_cut_ij*r_cut_ij)
                                {
                                lnboltzmann += patch->energy(r_ij,
                                    type,
                                    quat<float>(orientation),
                                    diameter,
                                    charge,
                                    typ_j,
                                    quat<float>(orientation_j),
                                    h_diameter.data[j],
                                    h_charge.data[j]);
                                }
                            }
                        }
                    }
                else
                    {
                    // skip ahead
                    cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                    }
                } // end loop over AABB nodes
            } // end loop over images
        }

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &lnboltzmann, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif

    return true;
    }


template<class Shape>
bool UpdaterMuVT<Shape>::tryInsertParticle(unsigned int timestep, unsigned int type, vec3<Scalar> pos,
    quat<Scalar> orientation, Scalar &lnboltzmann)
    {
    // do we have to compute energetic contribution?
    auto patch = m_mc->getPatchInteraction();

    lnboltzmann = Scalar(0.0);

    unsigned int overlap = 0;

    bool is_local = true;
    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        const BoxDim& global_box = this->m_pdata->getGlobalBox();
        ArrayHandle<unsigned int> h_cart_ranks(this->m_pdata->getDomainDecomposition()->getCartRanks(), access_location::host, access_mode::read);
        is_local = this->m_exec_conf->getRank() == this->m_pdata->getDomainDecomposition()->placeParticle(global_box, vec_to_scalar3(pos), h_cart_ranks.data);
        }
    #endif

    unsigned int nptl_local = m_pdata->getN() + m_pdata->getNGhosts();

    if (is_local)
        {
        // get some data structures from the integrator
        auto& image_list = m_mc->updateImageList();
        const unsigned int n_images = image_list.size();
        auto& params = m_mc->getParams();

        const Index2D& overlap_idx = m_mc->getOverlapIndexer();

        OverlapReal r_cut_patch(0.0);
        Scalar r_cut_self(0.0);

        if (patch)
            {
            r_cut_patch = patch->getRCut() + 0.5*patch->getAdditiveCutoff(type);
            r_cut_self = r_cut_patch + 0.5*patch->getAdditiveCutoff(type);
            }

        unsigned int err_count = 0;

            {
            // check for overlaps
            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

            ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

            // read in the current position and orientation
            Shape shape(orientation, params[type]);

            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_image = pos + image_list[cur_image];

                if (cur_image != 0)
                    {
                    // check for self-overlap with all images except the original
                    vec3<Scalar> r_ij = pos - pos_image;
                    if (h_overlaps.data[overlap_idx(type, type)]
                        && check_circumsphere_overlap(r_ij, shape, shape)
                        && test_overlap(r_ij, shape, shape, err_count))
                        {
                        overlap = 1;
                        break;
                        }

                    // self-energy
                    if (patch && dot(r_ij,r_ij) <= r_cut_self*r_cut_self)
                        {
                        lnboltzmann -= patch->energy(r_ij,
                            type,
                            quat<float>(orientation),
                            1.0, // diameter i
                            0.0, // charge i
                            type,
                            quat<float>(orientation),
                            1.0, // diameter i
                            0.0 // charge i
                            );
                        }
                    }
                }
            }

        // we cannot rely on a valid AABB tree when there are 0 particles
        if (! overlap && nptl_local > 0)
            {
            // Check particle against AABB tree for neighbors
            const detail::AABBTree& aabb_tree = m_mc->buildAABBTree();

            ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

            Shape shape(orientation, params[type]);
            OverlapReal R_query = std::max(shape.getCircumsphereDiameter()/OverlapReal(2.0),
                r_cut_patch - m_mc->getMinCoreDiameter()/(OverlapReal)2.0);
            detail::AABB aabb_local = detail::AABB(vec3<Scalar>(0,0,0),R_query);

            for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                {
                vec3<Scalar> pos_image = pos + image_list[cur_image];


                detail::AABB aabb = aabb_local;
                aabb.translate(pos_image);

                // stackless search
                for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
                    {
                    if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                        {
                        if (aabb_tree.isNodeLeaf(cur_node_idx))
                            {
                            for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                                {
                                // read in its position and orientation
                                unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                Scalar4 postype_j = h_postype.data[j];
                                Scalar4 orientation_j = h_orientation.data[j];

                                // put particles in coordinate system of particle i
                                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_image;

                                unsigned int typ_j = __scalar_as_int(postype_j.w);
                                Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                                Scalar r_cut_ij(0.0);
                                if (patch)
                                    r_cut_ij = r_cut_patch + 0.5*patch->getAdditiveCutoff(typ_j);

                                if (h_overlaps.data[overlap_idx(type, typ_j)]
                                    && check_circumsphere_overlap(r_ij, shape, shape_j)
                                    && test_overlap(r_ij, shape, shape_j, err_count))
                                    {
                                    overlap = 1;
                                    break;
                                    }
                                else if (patch && dot(r_ij,r_ij) <= r_cut_ij*r_cut_ij)
                                    {
                                    lnboltzmann -= patch->energy(r_ij,
                                        type,
                                        quat<float>(orientation),
                                        1.0, // diameter i
                                        0.0, // charge i
                                        typ_j,
                                        quat<float>(orientation_j),
                                        h_diameter.data[j],
                                        h_charge.data[j]);
                                    }
                                }
                            }
                        }
                    else
                        {
                        // skip ahead
                        cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                        }

                    if (overlap)
                        {
                        break;
                        }
                    } // end loop over AABB nodes

                if (overlap)
                    {
                    break;
                    }
                } // end loop over images
            } // end if nptl_local > 0
        } // end if local

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &lnboltzmann, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &overlap, 1, MPI_UNSIGNED, MPI_MAX, m_exec_conf->getMPICommunicator());
        }
    #endif

    return !overlap;
    }

template<class Shape>
bool UpdaterMuVT<Shape>::trySwitchType(unsigned int timestep, unsigned int tag, unsigned int newtype, Scalar &lnboltzmann)
    {
    lnboltzmann = Scalar(0.0);
    unsigned int overlap = 0;

    // guard against trying to modify empty particle data
    if (m_pdata->getNGlobal()==0) return false;

    // update the aabb tree
    const detail::AABBTree& aabb_tree = m_mc->buildAABBTree();

    // update the image list
    const std::vector<vec3<Scalar> >&image_list = m_mc->updateImageList();

    quat<Scalar> orientation(m_pdata->getOrientation(tag));

    // getPosition() takes into account grid shift, correct for that
    Scalar3 p = m_pdata->getPosition(tag)+m_pdata->getOrigin();
    int3 tmp = make_int3(0,0,0);
    m_pdata->getGlobalBox().wrap(p,tmp);
    vec3<Scalar> pos(p);

    // check for overlaps
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    const std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > & params = m_mc->getParams();

    ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);
    const Index2D & overlap_idx = m_mc->getOverlapIndexer();

    // read in the current position and orientation
    Shape shape(orientation, params[newtype]);

    // Check particle against AABB tree for neighbors
    detail::AABB aabb_local = shape.getAABB(vec3<Scalar>(0,0,0));

    unsigned int err_count = 0;

    const unsigned int n_images = image_list.size();
    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
        {
        vec3<Scalar> pos_image = pos + image_list[cur_image];

        detail::AABB aabb = aabb_local;
        aabb.translate(pos_image);

        if (cur_image != 0)
            {
            // check for self-overlap with all images except the original
            vec3<Scalar> r_ij = pos - pos_image;
            if (h_overlaps.data[overlap_idx(newtype,newtype)]
                && check_circumsphere_overlap(r_ij, shape, shape)
                && test_overlap(r_ij, shape, shape, err_count))
                {
                overlap = 1;
                break;
                }
            }

        // stackless search
        for (unsigned int cur_node_idx = 0; cur_node_idx < aabb_tree.getNumNodes(); cur_node_idx++)
            {
            if (detail::overlap(aabb_tree.getNodeAABB(cur_node_idx), aabb))
                {
                if (aabb_tree.isNodeLeaf(cur_node_idx))
                    {
                    for (unsigned int cur_p = 0; cur_p < aabb_tree.getNodeNumParticles(cur_node_idx); cur_p++)
                        {
                        // read in its position and orientation
                        unsigned int j = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                        Scalar4 postype_j = h_postype.data[j];
                        Scalar4 orientation_j = h_orientation.data[j];

                        // put particles in coordinate system of particle i
                        vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_image;

                        unsigned int typ_j = __scalar_as_int(postype_j.w);
                        Shape shape_j(quat<Scalar>(orientation_j), params[typ_j]);

                        if (h_overlaps.data[overlap_idx(typ_j, newtype)]
                            && check_circumsphere_overlap(r_ij, shape, shape_j)
                            && test_overlap(r_ij, shape, shape_j, err_count))
                            {
                            // do not count self overlap
                            if (h_tag.data[j] != tag)
                                {
                                overlap = 1;
                                break;
                                }
                            }
                        }
                    }
                }
            else
                {
                // skip ahead
                cur_node_idx += aabb_tree.getNodeSkip(cur_node_idx);
                }

            if (overlap)
                {
                break;
                }
            } // end loop over AABB nodes

        if (overlap)
            {
            break;
            }
        } // end loop over images

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &overlap, 1, MPI_UNSIGNED, MPI_MAX, m_exec_conf->getMPICommunicator());
        }
    #endif

    return !overlap;
    }

template<class Shape>
Scalar UpdaterMuVT<Shape>::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    hpmc_muvt_counters_t counters = getCounters(1);

    for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
        {
        std::string q = "hpmc_muvt_N_"+m_pdata->getNameByType(i);
        if (quantity == q)
            {
            return getNumParticlesType(i);
            }
        }
    if (quantity == "hpmc_muvt_insert_acceptance")
        {
        return counters.getInsertAcceptance();
        }
    else if (quantity == "hpmc_muvt_remove_acceptance")
        {
        return counters.getRemoveAcceptance();
        }
    else if (quantity == "hpmc_muvt_exchange_acceptance")
        {
        return counters.getExchangeAcceptance();
        }
    else if (quantity == "hpmc_muvt_volume_acceptance")
        {
        return counters.getVolumeAcceptance();
        }
    else
        {
        m_exec_conf->msg->error() << "UpdaterMuVT: Log quantity " << quantity
            << " is not supported by this Updater." << std::endl;
        throw std::runtime_error("Error querying log value.");
        }
    }

/*! \param mode 0 -> Absolute count, 1 -> relative to the start of the run, 2 -> relative to the last executed step
    \return The current state of the acceptance counters

    UpdaterMuVT maintains a count of the number of accepted and rejected moves since instantiation. getCounters()
    provides the current value. The parameter *mode* controls whether the returned counts are absolute, relative
    to the start of the run, or relative to the start of the last executed step.
*/
template<class Shape>
hpmc_muvt_counters_t UpdaterMuVT<Shape>::getCounters(unsigned int mode)
    {
    hpmc_muvt_counters_t result;

    if (mode == 0)
        result = m_count_total;
    else if (mode == 1)
        result = m_count_total - m_count_run_start;
    else
        result = m_count_total - m_count_step_start;

    // don't MPI_AllReduce counters because all ranks count the same thing
    return result;
    }
}
#endif
