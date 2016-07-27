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
                    bool pretend);

    std::vector< std::string > getProvidedLogQuantities();

    //! Calculates the requested log value and returns it
    Scalar getLogValue(const std::string& quantity, unsigned int timestep);

    void update(unsigned int timestep);

    void initialize()
        {
        if(!m_move_function)
            return;
        m_initialized = true;
        ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_det(m_determinant, access_location::host, access_mode::readwrite);
        ArrayHandle<typename Shape::param_type> h_params(m_mc->getParams(), access_location::host, access_mode::readwrite);
        for(size_t i = 0; i < m_pdata->getNTypes(); i++)
            {
            detail::mass_properties<Shape> mp(h_params.data[i]);
            h_det.data[i] = mp.getDeterminant();
            }
        }

    unsigned int getAcceptedCount(unsigned int ndx) { return m_count_accepted[ndx]; }

    unsigned int getTotalCount(unsigned int ndx) { return m_count_total[ndx]; }

    void resetStatistics()
        {
        std::fill(m_count_accepted.begin(), m_count_accepted.end(), 0);
        std::fill(m_count_total.begin(), m_count_total.end(), 0);
        }

    void registerLogBoltzmannFunction(std::shared_ptr< ShapeLogBoltzmannFunction<Shape> >  lbf)
        {
        if(m_log_boltz_function)
            return;
        m_log_boltz_function = lbf;
        }

    void registerShapeMove(std::shared_ptr<shape_move_function<Shape, Saru> > move)
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

    // void registerPythonCallback(boost::python::object pyfun, boost::python::list pyparams, Scalar stepsize, Scalar mixratio, bool normalized)
    //     {
    //     if(m_move_function) // if it exists I do not want to reset it.
    //         return;
    //     boost::python::stl_input_iterator<Scalar> begin(pyparams), end;
    //     std::vector<Scalar> params(begin, end);
    //     registerShapeMove(std::shared_ptr<shape_move_function<Shape, Saru> >( new python_callback_parameter_shape_move<Shape, Saru>(pyfun, params, stepsize, mixratio, m_seed, normalized)));
    //     }
    //
    // void registerConstantShapeMove(const typename Shape::param_type& shape, Scalar detI, const typename Shape::param_type& shape_move, Scalar detI_move)
    //     {
    //     if(m_move_function) // if it exists I do not want to reset it.
    //         return;
    //     registerShapeMove(std::shared_ptr<shape_move_function<Shape, Saru> >( new constant_shape_move<Shape, Saru>(shape, detI, shape_move, detI_move) ));
    //     }

    // void registerConvexPolyhedronShapeMove()
    //     {
    //     if(m_move_function) // if it exists I do not want to reset it.
    //         return;
    //     ArrayHandle<typename Shape::param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
    //     registerShapeMove(std::shared_ptr<shape_move_function<Shape, Saru> >( new convex_polyhedron_generalized_shape_move<Shape, Saru>(h_params.data[0], 1.0, 1.0, m_seed) ));
    //     }

    // boost::python::dict getShapeParams(unsigned int type_id)
    //     {
    //     // validate input
    //     if (type_id >= this->m_pdata->getNTypes())
    //         {
    //         this->m_exec_conf->msg->error() << "update.shape_updater." << /*evaluator::getName() <<*/ ": Trying to set pair params for a non existant type! " << type_id << std::endl;
    //         throw std::runtime_error("Error setting parameters in IntegratorHPMCMono");
    //         }
    //
    //     ArrayHandle<typename Shape::param_type> h_params(m_mc->getParams(), access_location::host, access_mode::read);
    //     detail::shape_param_to_python<Shape> converter;
    //     return converter(h_params.data[type_id]);
    //     }

    Scalar getStepSize(unsigned int typ)
        {
        if(m_move_function) return m_move_function->getStepSize(typ);
        return 0.0;
        }

    void setStepSize(unsigned int typ, Scalar stepsize)
        {
        if(m_move_function) m_move_function->setStepSize(typ, stepsize);
        }

    void countTypes()
        {
        ArrayHandle<unsigned int> h_ntypes(m_ntypes, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
        for(size_t i = 0; i < m_pdata->getNTypes(); i++)
            {
            unsigned int ct = 0;
            for(size_t j= 0; j < m_pdata->getN(); j++)
                {
                int typ_j = __scalar_as_int(h_postype.data[j].w);
                if(typ_j == i) ct++;
                }
            h_ntypes.data[i] = ct;
            }

        }

protected:
    std::string getParamName(size_t i)
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
                                 bool pretend)
    : Updater(sysdef), m_seed(seed), m_nselect(nselect),
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
    ArrayHandle<Scalar> h_det(m_determinant, access_location::host, access_mode::readwrite);
    for(size_t i = 0; i < m_pdata->getNTypes(); i++)
        {
        h_det.data[i] = 0.0;
        }
    countTypes(); // TODO: connect to ntypes change/particle changes to resize arrays and count them up again.
    }

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
    if(!m_initialized)
        initialize();

    if(!m_move_function || !m_log_boltz_function)
        return;

    Saru rng(m_move_ratio, m_seed, timestep); // TODO: better way to seed the rng?
    unsigned int move_type_select = rng.u32() & 0xffff;
    bool move = (move_type_select < m_move_ratio);

    if (!move)
        return;



    m_exec_conf->msg->notice(10) << "ElasticShape update: " << timestep << std::endl;

    // Shuffle the order of particles for this step
    m_update_order.resize(m_pdata->getNTypes());
    m_update_order.randomize(timestep);

    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "ElasticShape update");
    Scalar log_boltz = 0.0;

    GPUArray< typename Shape::param_type > param_copy(m_mc->getParams());
    GPUArray< Scalar > determinant_backup(m_determinant);
    m_move_function->prepare(timestep);
    for (unsigned int cur_type = 0; cur_type < m_nselect; cur_type++)
        {
        // make a trial move for i
        int typ_i = m_update_order[cur_type];
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

        Saru rng_i(typ_i, m_seed + m_exec_conf->getRank()*m_nselect + typ_i, timestep); //TODO: think about the seed for MPI.
        m_move_function->construct(timestep, typ_i, param, rng_i);
        h_det.data[typ_i] = m_move_function->getDeterminant(); // new determinant
        // energy and moment of interia change.
        log_boltz += (*m_log_boltz_function)(
                                                h_ntypes.data[typ_i],           // number of particles of type typ_i
                                                param,                          // new shape parameter
                                                h_det.data[typ_i],              // new determinant
                                                h_param_backup.data[typ_i],     // old shape parameter
                                                h_det_backup.data[typ_i]        // old determinant
                                            );
        m_mc->setParam(typ_i, param);
        }
    // calculate boltzmann factor.
    bool accept = false, reject=true; // looks redundant but it is not because of the pretend mode.
    if(rng.s(Scalar(0.0),Scalar(1.0)) < fast::exp(log_boltz))
        {
        accept = ! m_mc->countOverlaps(timestep, true);
        }

    if( !accept ) // catagorically reject the move.
        {
        m_move_function->retreat(timestep);
        }
    else if( m_pretend ) // pretend to accept the move but actually reject it.
        {
        m_move_function->retreat(timestep);
        for (unsigned int cur_type = 0; cur_type < m_nselect; cur_type++)
            {
            int typ_i = m_update_order[cur_type];
            m_count_accepted[typ_i]++;
            }
        }
    else    // actually accept the move.
        {
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
        m_determinant.swap(determinant_backup);
        ArrayHandle<typename Shape::param_type> h_param_copy(param_copy, access_location::host, access_mode::readwrite);
        for(size_t typ = 0; typ < m_pdata->getNTypes(); typ++)
            {
            m_mc->setParam(typ, h_param_copy.data[typ]); // set the params.
            }
        }
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
