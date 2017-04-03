#ifndef _UPDATER_SHAPE_H
#define _UPDATER_SHAPE_H

#include <numeric>
#include <algorithm>
#include "hoomd/Updater.h"
#include "hoomd/extern/saruprng.h"
#include "IntegratorHPMCMono.h"

#include "ShapeUtils.h"
#include "ShapeMoves.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace hpmc {


template< typename Shape >
class UpdaterShape  : public Updater
{
public:
    UpdaterShape(   std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr< IntegratorHPMCMono<Shape> > mc,
                    Scalar move_ratio,
                    unsigned int seed,
                    unsigned int nselect,
                    unsigned int nsweeps,
                    bool pretend);

    ~UpdaterShape();

    std::vector< std::string > getProvidedLogQuantities();

    //! Calculates the requested log value and returns it
    Scalar getLogValue(const std::string& quantity, unsigned int timestep);

    void update(unsigned int timestep);

    void initialize();

    unsigned int getAcceptedCount(unsigned int ndx) { return m_count_accepted[ndx]; }

    unsigned int getTotalCount(unsigned int ndx) { return m_count_total[ndx]; }

    void resetStatistics()
        {
        std::fill(m_count_accepted.begin(), m_count_accepted.end(), 0);
        std::fill(m_count_total.begin(), m_count_total.end(), 0);
        }

    void registerLogBoltzmannFunction(std::shared_ptr< ShapeLogBoltzmannFunction<Shape> >  lbf);

    void registerShapeMove(std::shared_ptr<shape_move_function<Shape, Saru> > move);

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

protected:
    static std::string getParamName(size_t i)
        {
        std::stringstream ss;
        std::string snum;
        ss << i;
        ss>>snum;
        return "shape_param-" + snum;
        }


private:
    unsigned int                m_seed;           //!< Random number seed
    unsigned int                m_nselect;
    unsigned int                m_nsweeps;
    std::vector<unsigned int>   m_count_accepted;
    std::vector<unsigned int>   m_count_total;
    unsigned int                m_move_ratio;

    std::shared_ptr< shape_move_function<Shape, Saru> >   m_move_function;
    std::shared_ptr< IntegratorHPMCMono<Shape> >          m_mc;
    std::shared_ptr< ShapeLogBoltzmannFunction<Shape> >   m_log_boltz_function;

    GPUArray< Scalar >          m_determinant;
    GPUArray< unsigned int >    m_ntypes;

    std::vector< std::string >  m_ProvidedQuantities;
    size_t                      m_num_params;
    bool                        m_pretend;
    bool                        m_initialized;
    detail::UpdateOrder         m_update_order;         //!< Update order
};

template < class Shape >
UpdaterShape<Shape>::UpdaterShape(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr< IntegratorHPMCMono<Shape> > mc,
                                 Scalar move_ratio,
                                 unsigned int seed,
                                 unsigned int nselect,
                                 unsigned int nsweeps,
                                 bool pretend)
    : Updater(sysdef), m_seed(seed), m_nselect(nselect), m_nsweeps(nsweeps),
      m_move_ratio(move_ratio*65535), m_mc(mc),
      m_determinant(m_pdata->getNTypes(), m_exec_conf),
      m_ntypes(m_pdata->getNTypes(), m_exec_conf), m_num_params(0),
      m_pretend(pretend), m_initialized(false), m_update_order(seed)
    {
    m_count_accepted.resize(m_pdata->getNTypes(), 0);
    m_count_total.resize(m_pdata->getNTypes(), 0);
    m_nselect = (m_pdata->getNTypes() < m_nselect) ? m_pdata->getNTypes() : m_nselect;
    m_ProvidedQuantities.push_back("shape_move_acceptance_ratio");
    m_ProvidedQuantities.push_back("shape_move_particle_volume");
    m_ProvidedQuantities.push_back("shape_move_energy");
        {
        ArrayHandle<Scalar> h_det(m_determinant, access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);
        for(size_t i = 0; i < m_pdata->getNTypes(); i++)
            {
            h_det.data[i] = 0.0;
            h_ntypes.data[i] = 0;
            }
        }
    countTypes(); // TODO: connect to ntypes change/particle changes to resize arrays and count them up again.
    }

template< class Shape >
UpdaterShape<Shape>::~UpdaterShape() { m_exec_conf->msg->notice(5) << "Destroying UpdaterShape "<< std::endl; }

/*! hpmc::UpdaterShape provides:
\returns a list of provided quantities
*/
template < class Shape >
std::vector< std::string > UpdaterShape<Shape>::getProvidedLogQuantities()
    {
    return m_ProvidedQuantities;
    }
//! Calculates the requested log value and returns it
template < class Shape >
Scalar UpdaterShape<Shape>::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if(quantity == "shape_move_acceptance_ratio")
        {
        unsigned int ctAccepted = 0, ctTotal = 0;
        ctAccepted = std::accumulate(m_count_accepted.begin(), m_count_accepted.end(), 0);
        ctTotal = std::accumulate(m_count_total.begin(), m_count_total.end(), 0);
        return ctTotal ? Scalar(ctAccepted)/Scalar(ctTotal) : 0;
        }
    else if(quantity == "shape_move_particle_volume")
        {
        ArrayHandle< unsigned int > h_ntypes(m_ntypes, access_location::host, access_mode::read);
        ArrayHandle<typename Shape::param_type> h_params(m_mc->getParams(), access_location::host, access_mode::readwrite);
        double volume = 0.0;
        for(size_t i = 0; i < m_pdata->getNTypes(); i++)
            {
            detail::mass_properties<Shape> mp(h_params.data[i]);
            volume += mp.getVolume()*Scalar(h_ntypes.data[i]);
            }
		return volume;
		}
    else if(quantity == "shape_move_energy")
        {
        Scalar energy = 0.0;
        ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_det(m_determinant, access_location::host, access_mode::readwrite);
        ArrayHandle<typename Shape::param_type> h_params(m_mc->getParams(), access_location::host, access_mode::readwrite);
        for(unsigned int i = 0; i < m_pdata->getNTypes(); i++)
            {
            energy += m_log_boltz_function->computeEnergy(h_ntypes.data[i], i, h_params.data[i], h_det.data[i]);
            }
        return energy;
        }
    else
		{
		for(size_t i = 0; i < m_num_params; i++)
		    {
		    if(quantity == getParamName(i))
    			{
    			return m_move_function->getParam(i);
    			}
		    }

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
    m_exec_conf->msg->notice(4) << "UpdaterShape update: " << timestep << ", initialized: "<< std::boolalpha << m_initialized << " @ " << std::hex << this << std::dec << std::endl;
    bool warn = !m_initialized;
    if(!m_initialized)
        initialize();
    if(!m_move_function || !m_log_boltz_function)
        {
    	if(warn) m_exec_conf->msg->warning() << "update.shape: runing without a move function! " << std::endl;
    	return;
        }

    Saru rng(m_move_ratio, m_seed, timestep);
    unsigned int move_type_select = rng.u32() & 0xffff;
    bool move = (move_type_select < m_move_ratio);
    if (!move)
        return;
    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "ElasticShape update");

    m_update_order.resize(m_pdata->getNTypes());
    for(unsigned int sweep=0; sweep < m_nsweeps; sweep++)
        {
        if (this->m_prof)
            this->m_prof->push(this->m_exec_conf, "ElasticShape setup");
        // Shuffle the order of particles for this sweep
        m_update_order.choose(timestep+40591, m_nselect, sweep+91193); // order of the list doesn't matter the probability of each combination is the same.

        Scalar log_boltz = 0.0;
        m_exec_conf->msg->notice(6) << "UpdaterShape copying data" << std::endl;
        if (this->m_prof)
            this->m_prof->push(this->m_exec_conf, "ElasticShape copy param");
        GPUArray< typename Shape::param_type > param_copy(m_nselect, m_exec_conf);
        {
        ArrayHandle<typename Shape::param_type> h_params(m_mc->getParams(), access_location::host, access_mode::readwrite);
        ArrayHandle<typename Shape::param_type> h_param_backup(param_copy, access_location::host, access_mode::readwrite);
        for (unsigned int i = 0; i < m_nselect; i++)
            {
            h_param_backup.data[i] = h_params.data[m_update_order[i]];
            }
        }
        if (this->m_prof)
            this->m_prof->pop();

        GPUArray< Scalar > determinant_backup(m_determinant);
        m_move_function->prepare(timestep);

        if (this->m_prof)
            this->m_prof->pop();
        if (this->m_prof)
            this->m_prof->push(this->m_exec_conf, "ElasticShape move");
        for (unsigned int cur_type = 0; cur_type < m_nselect; cur_type++)
            {
            // make a trial move for i
            int typ_i = m_update_order[cur_type];
            m_exec_conf->msg->notice(5) << " UpdaterShape making trial move for typeid=" << typ_i << ", " << cur_type << std::endl;
            m_count_total[typ_i]++;
            // access parameters
            typename Shape::param_type param;
                { // need to scope because we set at the end of loop
                ArrayHandle<typename Shape::param_type> h_params(m_mc->getParams(), access_location::host, access_mode::readwrite);
                param = h_params.data[typ_i];
                }
            ArrayHandle<typename Shape::param_type> h_param_backup(param_copy, access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_det(m_determinant, access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_det_backup(determinant_backup, access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);

            Saru rng_i(m_seed + m_nselect + sweep + m_nsweeps, typ_i+1046527, timestep+7919);
            m_move_function->construct(timestep, typ_i, param, rng_i);
            h_det.data[typ_i] = m_move_function->getDeterminant(); // new determinant
            m_exec_conf->msg->notice(5) << " UpdaterShape I=" << h_det.data[typ_i] << ", " << h_det_backup.data[typ_i] << std::endl;
            // energy and moment of interia change.
            assert(h_det.data[typ_i] != 0 && h_det_backup.data[typ_i] != 0);
            log_boltz += (*m_log_boltz_function)(
                                                    h_ntypes.data[typ_i],           // number of particles of type typ_i,
                                                    typ_i,                          // the type id
                                                    param,                          // new shape parameter
                                                    h_det.data[typ_i],              // new determinant
                                                    h_param_backup.data[cur_type],     // old shape parameter
                                                    h_det_backup.data[typ_i]        // old determinant
                                                );
            m_mc->setParam(typ_i, param, cur_type == (m_nselect-1));
            }
        if (this->m_prof)
            this->m_prof->pop();

        if (this->m_prof)
            this->m_prof->push(this->m_exec_conf, "ElasticShape cleanup");
        // calculate boltzmann factor.
        bool accept = false, reject=true; // looks redundant but it is not because of the pretend mode.
        Scalar p = rng.s(Scalar(0.0),Scalar(1.0)), Z = fast::exp(log_boltz);
        m_exec_conf->msg->notice(5) << " UpdaterShape p=" << p << ", z=" << Z << std::endl;
        if( p < Z)
            {
            unsigned int overlaps = 1;
            if(m_pdata->getNTypes() == m_pdata->getNGlobal())
                {
                overlaps = m_mc->countOverlapsEx(timestep, true, m_update_order.begin(), m_update_order.begin()+m_nselect);
                }
            else
                {
                overlaps = m_mc->countOverlaps(timestep, true);
                }
            accept = !overlaps;
            m_exec_conf->msg->notice(5) << " UpdaterShape counted " << overlaps << " overlaps" << std::endl;
            }

        if( !accept ) // catagorically reject the move.
            {
            m_exec_conf->msg->notice(5) << " UpdaterShape move retreating" << std::endl;
            m_move_function->retreat(timestep);
            }
        else if( m_pretend ) // pretend to accept the move but actually reject it.
            {
            m_exec_conf->msg->notice(5) << " UpdaterShape move accepted -- pretend mode" << std::endl;
            m_move_function->retreat(timestep);
            for (unsigned int cur_type = 0; cur_type < m_nselect; cur_type++)
                {
                int typ_i = m_update_order[cur_type];
                m_count_accepted[typ_i]++;
                }
            }
        else    // actually accept the move.
            {
            m_exec_conf->msg->notice(5) << " UpdaterShape move accepted" << std::endl;
            for (unsigned int cur_type = 0; cur_type < m_nselect; cur_type++)
                {
                int typ_i = m_update_order[cur_type];
                m_count_accepted[typ_i]++;
                }
            // m_move_function->advance(timestep);
            reject = false;
            }

        if(reject)
            {
            m_exec_conf->msg->notice(5) << " UpdaterShape move rejected" << std::endl;
            m_determinant.swap(determinant_backup);
            // m_mc->swapParams(param_copy);
            ArrayHandle<typename Shape::param_type> h_param_copy(param_copy, access_location::host, access_mode::readwrite);
            for(size_t typ = 0; typ < m_nselect; typ++)
                {
                m_mc->setParam(m_update_order[typ], h_param_copy.data[typ], typ == (m_nselect-1)); // set the params.
                }
            }
        if (this->m_prof)
            this->m_prof->pop();
        }
    if (this->m_prof)
        this->m_prof->pop();
    m_exec_conf->msg->notice(4) << " UpdaterShape update done" << std::endl;
    }

template< typename Shape>
void UpdaterShape<Shape>::initialize()
    {
    ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_det(m_determinant, access_location::host, access_mode::readwrite);
    ArrayHandle<typename Shape::param_type> h_params(m_mc->getParams(), access_location::host, access_mode::readwrite);
    for(size_t i = 0; i < m_pdata->getNTypes(); i++)
        {
        detail::mass_properties<Shape> mp(h_params.data[i]);
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
    }

template< typename Shape>
void UpdaterShape<Shape>::registerShapeMove(std::shared_ptr<shape_move_function<Shape, Saru> > move)
    {
    if(m_move_function) // if it exists I do not want to reset it.
        return;
    m_move_function = move;
    m_num_params = m_move_function->getNumParam();
    for(size_t i = 0; i < m_num_params; i++)
        {
        m_ProvidedQuantities.push_back(getParamName(i));
        }
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

template< typename Shape >
void export_UpdaterShape(pybind11::module& m, const std::string& name)
    {
    pybind11::class_< UpdaterShape<Shape>, std::shared_ptr< UpdaterShape<Shape> > >(m, name.c_str(), pybind11::base<Updater>())
    .def( pybind11::init<   std::shared_ptr<SystemDefinition>,
                            std::shared_ptr< IntegratorHPMCMono<Shape> >,
                            Scalar,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            bool >())
    .def("getAcceptedCount", &UpdaterShape<Shape>::getAcceptedCount)
    .def("getTotalCount", &UpdaterShape<Shape>::getTotalCount)
    .def("registerShapeMove", &UpdaterShape<Shape>::registerShapeMove)
    .def("registerLogBoltzmannFunction", &UpdaterShape<Shape>::registerLogBoltzmannFunction)
    .def("resetStatistics", &UpdaterShape<Shape>::resetStatistics)
    .def("getStepSize", &UpdaterShape<Shape>::getStepSize)
    .def("setStepSize", &UpdaterShape<Shape>::setStepSize)
    ;
    }



} // namespace



#endif
