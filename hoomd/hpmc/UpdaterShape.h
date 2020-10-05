#ifndef _UPDATER_SHAPE_H
#define _UPDATER_SHAPE_H

#include <numeric>
#include <algorithm>
#include "hoomd/Updater.h"
#include "IntegratorHPMCMono.h"
#include "hoomd/HOOMDMPI.h"
#include "ShapeUtils.h"
#include "ShapeMoves.h"
#include <pybind11/pybind11.h>

namespace hpmc {


template< typename Shape >
class UpdaterShape  : public Updater
{
public:
    UpdaterShape(   std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr< IntegratorHPMCMono<Shape> > mc,
                    Scalar move_ratio,
                    unsigned int seed,
                    unsigned int tselect,
                    unsigned int nsweeps,
                    bool pretend,
                    bool multiphase,
                    unsigned int numphase);

    ~UpdaterShape();

    std::vector< std::string > getProvidedLogQuantities();

    //! Calculates the requested log value and returns it
    Scalar getLogValue(const std::string& quantity, unsigned int timestep);

    void update(unsigned int timestep);

    void initialize();

    unsigned int getAcceptedCount(unsigned int ndx) { return m_count_accepted[ndx]; }

    unsigned int getTotalCount(unsigned int ndx) { return m_count_total[ndx]; }

    unsigned int getAcceptedBox(unsigned int ndx) { return m_box_accepted[ndx]; }

    unsigned int getTotalBox(unsigned int ndx) { return m_box_total[ndx]; }

    void resetStatistics()
        {
        std::fill(m_count_accepted.begin(), m_count_accepted.end(), 0);
        std::fill(m_count_total.begin(), m_count_total.end(), 0);
        std::fill(m_box_accepted.begin(), m_box_accepted.end(), 0);
        std::fill(m_box_total.begin(), m_box_total.end(), 0);
        }

    void registerLogBoltzmannFunction(std::shared_ptr< ShapeLogBoltzmannFunction<Shape> >  lbf);

    void registerShapeMove(std::shared_ptr< ShapeMoveBase<Shape> > move);

    Scalar getStepSize(unsigned int typ)
        {
        if(m_move_function) return m_move_function->getStepSize(typ);
        return 0.0;
        }

    void setStepSize(unsigned int typ, Scalar stepsize)
        {
        if(m_move_function) m_move_function->setStepSize(typ, stepsize);
        }


    void countTypes();

    //! Method that is called whenever the GSD file is written if connected to a GSD file.
    int slotWriteGSD(gsd_handle&, std::string name) const;

    //! Method that is called to connect to the gsd write state signal
    void connectGSDStateSignal(std::shared_ptr<GSDDumpWriter> writer, std::string name);

    //! Method that is called to connect to the gsd write state signal
    bool restoreStateGSD(std::shared_ptr<GSDReader> reader, std::string name);

private:
    unsigned int                m_seed;               //!< Random number seed
    int                         m_global_partition;   //!< Random number seed
    unsigned int                n_type_select;
    unsigned int                m_nsweeps;
    std::vector<unsigned int>   m_count_accepted;
    std::vector<unsigned int>   m_count_total;
    std::vector<unsigned int>   m_box_accepted;
    std::vector<unsigned int>   m_box_total;
    unsigned int                m_move_ratio;

    std::shared_ptr< ShapeMoveBase<Shape> >   m_move_function;
    std::shared_ptr< IntegratorHPMCMono<Shape> >          m_mc;
    std::shared_ptr< ShapeLogBoltzmannFunction<Shape> >   m_log_boltz_function;

    GPUArray< Scalar >          m_determinant;
    GPUArray< unsigned int >    m_ntypes;

    std::vector< std::string >  m_provided_quantities;
    size_t                      m_num_params;
    bool                        m_pretend;
    bool                        m_initialized;
    bool                        m_multi_phase;
    unsigned int                m_num_phase;
    detail::UpdateOrder         m_update_order;         //!< Update order


    static constexpr Scalar m_tol = 0.00001; //!< The minimum move size required not to be ignored.

};

template < class Shape >
UpdaterShape<Shape>::UpdaterShape(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr< IntegratorHPMCMono<Shape> > mc,
                                 Scalar move_ratio,
                                 unsigned int seed,
                                 unsigned int tselect,
                                 unsigned int nsweeps,
                                 bool pretend,
                                 bool multiphase,
                                 unsigned int numphase)
    : Updater(sysdef), m_seed(seed), m_global_partition(0), n_type_select(tselect), m_nsweeps(nsweeps),
      m_move_ratio(move_ratio*65535), m_mc(mc),
      m_determinant(m_pdata->getNTypes(), m_exec_conf),
      m_ntypes(m_pdata->getNTypes(), m_exec_conf), m_num_params(0),
      m_pretend(pretend),m_initialized(false), m_multi_phase(multiphase),
      m_num_phase(numphase), m_update_order(seed)
    {
    m_count_accepted.resize(m_pdata->getNTypes(), 0);
    m_count_total.resize(m_pdata->getNTypes(), 0);
    m_box_accepted.resize(m_pdata->getNTypes(), 0);
    m_box_total.resize(m_pdata->getNTypes(), 0);
    n_type_select = (m_pdata->getNTypes() < n_type_select) ? m_pdata->getNTypes() : n_type_select;
    m_provided_quantities.push_back("shape_move_acceptance_ratio");
    m_provided_quantities.push_back("shape_move_particle_volume");
    m_provided_quantities.push_back("shape_move_multi_phase_box");

    ArrayHandle<Scalar> h_det(m_determinant, access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);
    m_provided_quantities.push_back("shape_move_energy");
    for(size_t i = 0; i < m_pdata->getNTypes(); i++)
        {
        h_det.data[i] = 0.0;
        h_ntypes.data[i] = 0;
        }
    // TODO: connect to ntypes change/particle changes to resize arrays and count them up again.
    countTypes();
    //TODO: add a sanity check to makesure that MPI is setup correctly
    if(m_multi_phase)
    {
    #ifdef ENABLE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &m_global_partition);
    assert(m_global_partition < 2);
    #endif
    }

    }

template< class Shape >
UpdaterShape<Shape>::~UpdaterShape()
    {
    m_exec_conf->msg->notice(5) << "Destroying UpdaterShape " << std::endl;
    }

/*! hpmc::UpdaterShape provides:
\returns a list of provided quantities
*/
template < class Shape >
std::vector< std::string > UpdaterShape<Shape>::getProvidedLogQuantities()
    {
    return m_provided_quantities;
    }

//! Calculates the requested log value and returns it
template < class Shape >
Scalar UpdaterShape<Shape>::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if(m_move_function->isProvidedQuantity(quantity))
        {
        return m_move_function->getLogValue(quantity, timestep);
        }
    else if(m_log_boltz_function->isProvidedQuantity(quantity))
        {
        return m_log_boltz_function->getLogValue(quantity, timestep);
        }
    else if(quantity == "shape_move_acceptance_ratio")
        {
        unsigned int ctAccepted = 0, ctTotal = 0;
        ctAccepted = std::accumulate(m_count_accepted.begin(), m_count_accepted.end(), 0);
        ctTotal = std::accumulate(m_count_total.begin(), m_count_total.end(), 0);
        return ctTotal ? Scalar(ctAccepted)/Scalar(ctTotal) : 0;
        }
    else if(quantity == "shape_move_particle_volume")
        {
        ArrayHandle< unsigned int > h_ntypes(m_ntypes, access_location::host, access_mode::read);
        auto params = m_mc->getParams();
        double volume = 0.0;
        for(size_t i = 0; i < m_pdata->getNTypes(); i++)
            {
            detail::MassProperties<Shape> mp(params[i]);
            volume += mp.getVolume()*Scalar(h_ntypes.data[i]);
            }
		return volume;
		}
    else if(quantity == "shape_move_multi_phase_box")
        {
        unsigned int boxAccepted = 0, boxTotal = 0;
        boxAccepted = std::accumulate(m_box_accepted.begin(), m_box_accepted.end(), 0);
        boxTotal = std::accumulate(m_box_total.begin(), m_box_total.end(), 0);
        return boxTotal ? Scalar(boxAccepted)/Scalar(boxTotal) : 0;
        }
    else if(quantity == "shape_move_energy")
        {
        Scalar energy = 0.0;
        ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_det(m_determinant, access_location::host, access_mode::readwrite);
        for(unsigned int i = 0; i < m_pdata->getNTypes(); i++)
            {
            energy += m_log_boltz_function->computeEnergy(
                    timestep, h_ntypes.data[i], i, m_mc->getParams()[i], h_det.data[i]);
            }
        return energy;
        }
    else if(quantity.compare(0, 11, "shape_param") == 0)
        {
        return m_move_function->getLogValue(quantity, timestep);
        }
    else
	    {
        m_exec_conf->msg->error() << "update.shape: " << quantity << " is not a valid log quantity" << std::endl;
		throw std::runtime_error("Error getting log value");
		}
    }

/*! Perform Metropolis Monte Carlo shape deformations
\param timestep Current time step of the simulation
*/
template < class Shape >
void UpdaterShape<Shape>::update(unsigned int timestep)
    {
    typedef std::vector<typename Shape::param_type, managed_allocator<typename Shape::param_type> > param_vector;
    m_exec_conf->msg->notice(4) << "UpdaterShape update: " << timestep << ", initialized: "
                                << std::boolalpha << m_initialized << " @ "
                                << std::hex << this << std::dec << std::endl;
    bool warn = !m_initialized;
    if(!m_initialized)
        initialize();
    if(!m_move_function || !m_log_boltz_function)
        {
    	if(warn) m_exec_conf->msg->warning() << "update.shape: running without a move function! " << std::endl;
    	return;
        }

    hoomd::RandomGenerator rng(hoomd::RNGIdentifier::UpdaterShapeUpdate, m_move_ratio, m_seed, timestep);
    unsigned int move_type_select = hoomd::UniformIntDistribution(0xffff)(rng);
    bool move = (move_type_select < m_move_ratio);
    if (!move)
        return;
    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "UpdaterShape update");

    m_update_order.resize(m_pdata->getNTypes());
    for(unsigned int sweep=0; sweep < m_nsweeps; sweep++)
        {
        if (this->m_prof)
            this->m_prof->push(this->m_exec_conf, "UpdaterShape setup");
        // Shuffle the order of particles for this sweep
        // TODO: should these be better random numbers?
        m_update_order.setRandomDirection(timestep+40591); // order of the list doesn't matter the probability of each combination is the same.
        if (this->m_prof)
            this->m_prof->pop();

        Scalar log_boltz = 0.0;
        m_exec_conf->msg->notice(6) << "UpdaterShape copying data" << std::endl;
        if (this->m_prof)
            this->m_prof->push(this->m_exec_conf, "UpdaterShape copy param");

        param_vector& params = m_mc->getParams();
        param_vector param_copy(n_type_select);
        for (unsigned int i = 0; i < n_type_select; i++)
            {
            param_copy[i] = params[m_update_order[i]];
            }
        if (this->m_prof)
            this->m_prof->pop();

        if (this->m_prof)
            this->m_prof->push(this->m_exec_conf, "UpdaterShape move");
        GPUArray< Scalar > determinant_backup(m_determinant);
        m_move_function->prepare(timestep);

        for (unsigned int cur_type = 0; cur_type < n_type_select; cur_type++)
            {
            // make a trial move for i
            int typ_i = m_update_order[cur_type];
            // Skip move if step size is smaller than tolerance
            if (m_move_function->getStepSize(typ_i) < m_tol)
                {
                m_exec_conf->msg->notice(5) << " Skipping moves for particle typeid=" << typ_i << ", " << cur_type << std::endl;
                continue;
                }
            else
                {
                m_exec_conf->msg->notice(5) << " UpdaterShape making trial move for typeid=" << typ_i << ", " << cur_type << std::endl;
                }
            m_count_total[typ_i]++;
            // access parameters
            typename Shape::param_type param;
            param = params[typ_i];
            ArrayHandle<Scalar> h_det(m_determinant, access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_det_backup(determinant_backup, access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);

            hoomd::RandomGenerator rng_i(hoomd::RNGIdentifier::UpdaterShapeConstruct, m_seed, timestep, typ_i, sweep);
            m_move_function->construct(timestep, typ_i, param, rng_i);
            h_det.data[typ_i] = m_move_function->getDeterminant(); // new determinant
            m_exec_conf->msg->notice(5) << " UpdaterShape I=" << h_det.data[typ_i] << ", " << h_det_backup.data[typ_i] << std::endl;
            // energy and moment of interia change.
            assert(h_det.data[typ_i] != 0 && h_det_backup.data[typ_i] != 0);
            log_boltz += (*m_log_boltz_function)(   timestep,
                                                    h_ntypes.data[typ_i],           // number of particles of type typ_i,
                                                    typ_i,                          // the type id
                                                    param,                          // new shape parameter
                                                    h_det.data[typ_i],              // new determinant
                                                    param_copy[cur_type],           // old shape parameter
                                                    h_det_backup.data[typ_i]        // old determinant
                                                );
            m_mc->setParam(typ_i, param);
            }  // end loop over particle types
        if (this->m_prof)
            this->m_prof->pop();

        if (this->m_prof)
            this->m_prof->push(this->m_exec_conf, "UpdaterShape cleanup");
        // calculate boltzmann factor.
        bool accept = false, reject=true; // looks redundant but it is not because of the pretend mode.

        Scalar p = hoomd::detail::generate_canonical<Scalar>(rng);
        Scalar Z = fast::exp(log_boltz);
        m_exec_conf->msg->notice(5) << " UpdaterShape p=" << p << ", z=" << Z << std::endl;

    if(m_multi_phase)
        {
        #ifdef ENABLE_MPI
        std::vector<Scalar> Zs;
        all_gather_v(Z, Zs, MPI_COMM_WORLD);
        Z = std::accumulate(Zs.begin(), Zs.end(), 1, std::multiplies<Scalar>());
        #endif
        }
    if(p < Z)
        {
        bool early_exit = true;
        unsigned int overlap_count = 0;
        unsigned int err_count = 0;

        m_exec_conf->msg->notice(10) << "HPMCMono count overlaps: " << timestep << std::endl;

        // build an up to date AABB tree
        const detail::AABBTree& aabb_tree = m_mc->buildAABBTree();
        // update the image list
        std::vector<vec3<Scalar> > image_list = m_mc->updateImageList();

        if (this->m_prof) this->m_prof->push(this->m_exec_conf, "HPMC count overlaps");

        const Index2D& overlap_idx = m_mc->getOverlapIndexer();
        // access particle data and system box
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

        // access parameters and interaction matrix
        ArrayHandle<unsigned int> h_overlaps(m_mc->getInteractionMatrix(), access_location::host, access_mode::read);

        // Loop over particles corresponding to n_type_select
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            Scalar4 postype_i = h_postype.data[i];
            int typ_i = __scalar_as_int(postype_i.w);
            for (unsigned int cur_type = 0; cur_type < n_type_select; cur_type++)
                {
                // Only check overlaps for particles of types specified by n_type_select
                if (typ_i == m_update_order[cur_type])
                    {
                    // read in the current position and orientation
                    Scalar4 orientation_i = h_orientation.data[i];
                    Shape shape_i(quat<Scalar>(orientation_i), m_mc->getParams()[typ_i]);
                    vec3<Scalar> pos_i = vec3<Scalar>(postype_i);
                    // Check particle against AABB tree for neighbors
                    detail::AABB aabb_i_local = shape_i.getAABB(vec3<Scalar>(0,0,0));

                    const unsigned int n_images = image_list.size();
                    for (unsigned int cur_image = 0; cur_image < n_images; cur_image++)
                        {
                        vec3<Scalar> pos_i_image = pos_i + image_list[cur_image];
                        detail::AABB aabb = aabb_i_local;
                        aabb.translate(pos_i_image);

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
                                        unsigned int k = aabb_tree.getNodeParticle(cur_node_idx, cur_p);

                                        // skip i==j in the 0 image
                                        if (cur_image == 0 && i == k)
                                            continue;

                                        Scalar4 postype_k = h_postype.data[k];
                                        Scalar4 orientation_k = h_orientation.data[k];

                                        // put particles in coordinate system of particle i
                                        vec3<Scalar> r_ik = vec3<Scalar>(postype_k) - pos_i_image;

                                        unsigned int typ_k = __scalar_as_int(postype_k.w);
                                        Shape shape_k(quat<Scalar>(orientation_k), m_mc->getParams()[typ_k]);

                                        if (h_overlaps.data[overlap_idx(typ_i,typ_k)]
                                            && check_circumsphere_overlap(r_ik, shape_i, shape_k)
                                            && test_overlap(r_ik, shape_i, shape_k, err_count)
                                            && test_overlap(-r_ik, shape_k, shape_i, err_count))
                                            {
                                            overlap_count++;
                                            if (early_exit)
                                                {
                                                // exit early from loop over neighbor particles
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

                            if (overlap_count && early_exit)
                                {
                                break;
                                }
                            } // end loop over AABB nodes

                        if (overlap_count && early_exit)
                            {
                            break;
                            }
                        } // end loop over images

                    if (overlap_count && early_exit)
                        {
                        break;
                        }
                    }
                }

            } // end loop over particles

        if (this->m_prof) this->m_prof->pop(this->m_exec_conf);

        #ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            MPI_Allreduce(MPI_IN_PLACE, &overlap_count, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
            if (early_exit && overlap_count > 1)
                overlap_count = 1;
            }
        #endif

        accept = !overlap_count;

        m_exec_conf->msg->notice(5) << " UpdaterShape counted " << overlap_count << " overlaps" << std::endl;
        if(m_multi_phase)
            {
            #ifdef ENABLE_MPI
            // make sure random seeds are equal
            if(accept)
                {
                for (unsigned int cur_type = 0; cur_type < n_type_select; cur_type++)
                    {
                    int typ_i = m_update_order[cur_type];
                    m_box_accepted[typ_i]++;
                    }
                }
            std::vector<int> all_a;
            all_gather_v((int)accept, all_a, MPI_COMM_WORLD);
            accept = std::accumulate(all_a.begin(), all_a.end(), 1, std::multiplies<int>());
            #endif
            }
        m_exec_conf->msg->notice(5) << " UpdaterShape p=" << p << ", z=" << Z << std::endl;
        }

        if( !accept ) // categorically reject the move.
            {
            m_exec_conf->msg->notice(5) << " UpdaterShape move retreating" << std::endl;
            m_move_function->retreat(timestep);
            }
        else if( m_pretend ) // pretend to accept the move but actually reject it.
            {
            m_exec_conf->msg->notice(5) << " UpdaterShape move accepted -- pretend mode" << std::endl;
            m_move_function->retreat(timestep);
            for (unsigned int cur_type = 0; cur_type < n_type_select; cur_type++)
                {
                int typ_i = m_update_order[cur_type];
                m_count_accepted[typ_i]++;
                }
            }
        else    // actually accept the move.
            {
            m_exec_conf->msg->notice(5) << " UpdaterShape move accepted" << std::endl;
            for (unsigned int cur_type = 0; cur_type < n_type_select; cur_type++)
                {
                int typ_i = m_update_order[cur_type];
                m_count_accepted[typ_i]++;
                }
            reject = false;
            }

        if(reject)
            {
            m_exec_conf->msg->notice(5) << " UpdaterShape move rejected" << std::endl;
            m_determinant.swap(determinant_backup);
            // m_mc->swapParams(param_copy);
            // ArrayHandle<typename Shape::param_type> h_param_copy(param_copy, access_location::host, access_mode::readwrite);
            for(size_t typ = 0; typ < n_type_select; typ++)
                {
                m_mc->setParam(m_update_order[typ], param_copy[typ]); // set the params.
                }
            }
        if (this->m_prof)
            this->m_prof->pop();
        }  // end loop over n_sweeps
    if (this->m_prof)
        this->m_prof->pop();
    m_exec_conf->msg->notice(4) << " UpdaterShape update done" << std::endl;
    }  // end UpdaterShape<Shape>::update(unsigned int timestep)

template< typename Shape>
void UpdaterShape<Shape>::initialize()
    {
    ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_det(m_determinant, access_location::host, access_mode::readwrite);
    // ArrayHandle<typename Shape::param_type> h_params(m_mc->getParams(), access_location::host, access_mode::readwrite);
    auto params = m_mc->getParams();
    for(size_t i = 0; i < m_pdata->getNTypes(); i++)
        {
        detail::MassProperties<Shape> mp(params[i]);
        h_det.data[i] = mp.getDeterminant();
        }
    m_initialized = true;
    }

template< typename Shape>
void UpdaterShape<Shape>::registerLogBoltzmannFunction(std::shared_ptr< ShapeLogBoltzmannFunction<Shape> >  lbf)
    {
    if(m_log_boltz_function)
        return;
    m_log_boltz_function = lbf;
    std::vector< std::string > quantities(m_log_boltz_function->getProvidedLogQuantities());
    m_provided_quantities.reserve( m_provided_quantities.size() + quantities.size() );
    m_provided_quantities.insert(m_provided_quantities.end(), quantities.begin(), quantities.end());
    }

template< typename Shape>
void UpdaterShape<Shape>::registerShapeMove(std::shared_ptr<ShapeMoveBase<Shape> > move)
    {
    if(m_move_function) // if it exists I do not want to reset it.
        return;
    m_move_function = move;
    std::vector< std::string > quantities(m_move_function->getProvidedLogQuantities());
    m_provided_quantities.reserve( m_provided_quantities.size() + quantities.size() );
    m_provided_quantities.insert(m_provided_quantities.end(), quantities.begin(), quantities.end());
    }

template< typename Shape>
void UpdaterShape<Shape>::countTypes()
    {
    // zero the array.
    ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);
    for(size_t i = 0; i < m_pdata->getNTypes(); i++)
        {
        h_ntypes.data[i] = 0;
        }

    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    for(size_t j= 0; j < m_pdata->getN(); j++)
        {
        int typ_j = __scalar_as_int(h_postype.data[j].w);
        h_ntypes.data[typ_j]++;
        }

    #ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, h_ntypes.data, m_pdata->getNTypes(), MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif
    }

template< typename Shape>
int UpdaterShape<Shape>::slotWriteGSD(gsd_handle& handle, std::string name) const
    {
    m_exec_conf->msg->notice(2) << "UpdaterShape writing to GSD File to name: "<< name << std::endl;
    #ifdef ENABLE_MPI
    bool mpi=(bool)m_pdata->getDomainDecomposition();
    #else
    bool mpi=false;
    #endif
    int retval = m_move_function->writeGSD(handle, name+"move/", m_exec_conf, mpi);
    return retval;
    }

template< typename Shape>
void UpdaterShape<Shape>::connectGSDStateSignal(std::shared_ptr<GSDDumpWriter> writer, std::string name)
    {
    typedef hoomd::detail::SharedSignalSlot<int(gsd_handle&)> SlotType;
    auto func = std::bind(&UpdaterShape<Shape>::slotWriteGSD, this, std::placeholders::_1, name);
    std::shared_ptr<hoomd::detail::SignalSlot> pslot( new SlotType(writer->getWriteSignal(), func));
    addSlot(pslot);
    }

template< typename Shape>
bool UpdaterShape<Shape>::restoreStateGSD(std::shared_ptr<GSDReader> reader, std::string name)
    {
    m_exec_conf->msg->notice(2) << "UpdaterShape from GSD File to name: "<< name << std::endl;
    #ifdef ENABLE_MPI
    bool mpi=(bool)m_pdata->getDomainDecomposition();
    #else
    bool mpi=false;
    #endif
    bool success = m_move_function->restoreStateGSD(reader, name+"move/", m_exec_conf, mpi);
    return success;
    }

template< typename Shape >
void export_UpdaterShape(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterShape<Shape>, std::shared_ptr< UpdaterShape<Shape> >, Updater >(m, name.c_str())
    .def( pybind11::init<   std::shared_ptr<SystemDefinition>,
                            std::shared_ptr< IntegratorHPMCMono<Shape> >,
                            Scalar,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            bool,
                            bool,
                            unsigned int>())
    .def("getAcceptedCount", &UpdaterShape<Shape>::getAcceptedCount)
    .def("getTotalCount", &UpdaterShape<Shape>::getTotalCount)
    .def("registerShapeMove", &UpdaterShape<Shape>::registerShapeMove)
    .def("registerLogBoltzmannFunction", &UpdaterShape<Shape>::registerLogBoltzmannFunction)
    .def("resetStatistics", &UpdaterShape<Shape>::resetStatistics)
    .def("getStepSize", &UpdaterShape<Shape>::getStepSize)
    .def("setStepSize", &UpdaterShape<Shape>::setStepSize)
    .def("connectGSDStateSignal", &UpdaterShape<Shape>::connectGSDStateSignal)
    .def("restoreStateGSD", &UpdaterShape<Shape>::restoreStateGSD)
    ;
    }


} // namespace



#endif
