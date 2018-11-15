#ifndef _SHAPE_MOVES_H
#define _SHAPE_MOVES_H

#include "hoomd/Saru.h"
#include "ShapeUtils.h"
#include <hoomd/Variant.h>
#include "Moves.h"
#include "hoomd/GSDState.h"
#include <hoomd/extern/Eigen/Eigen/Dense>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace hpmc {


class not_implemented_error : std::exception {
  const char* what() const noexcept {return "Error: Function called that has not been implemented.\n";}
};

template < typename Shape, typename RNG>
class shape_move_function
{
public:
    shape_move_function(unsigned int ntypes) :
        m_determinantInertiaTensor(0),
        m_step_size(ntypes)
        {
        }

    shape_move_function(const shape_move_function& src) :
        m_determinantInertiaTensor(src.getDeterminantInertiaTensor()),
        m_step_size(src.getStepSizeArray())
        {
        }

    //! prepare is called at the beginning of every update()
    virtual void prepare(unsigned int timestep)
        {
        throw not_implemented_error();
        }

    //! construct is called for each particle type that will be changed in update()
    virtual void construct(const unsigned int&, const unsigned int&, typename Shape::param_type&, RNG&)
        {
        throw not_implemented_error();
        }

    //! retreat whenever the proposed move is rejected.
    virtual void retreat(const unsigned int)
        {
        throw not_implemented_error();
        }

    // TODO: remove this?
    Scalar getDeterminant() const
        {
        return m_determinantInertiaTensor;
        }

    //! Get the stepsize for \param type_id
    Scalar getStepSize(const unsigned int& type_id) const
        {
        return m_step_size[type_id];
        }

    //! Get all of the stepsizes
    const std::vector<Scalar>& getStepSizeArray() const
        {
        return m_step_size;
        }

    //! Set the step size for the \param type_id to \param stepsize
    void setStepSize(const unsigned int& type_id, const Scalar& stepsize)
        {
        m_step_size[type_id] = stepsize;
        }

    //! Method that is called whenever the GSD file is written if connected to a GSD file.
    virtual int writeGSD(gsd_handle& handle, std::string name, const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) const
        {
        if(!exec_conf->isRoot())
            return 0;
        int retval = 0;
        std::string path = name + "stepsize";
        exec_conf->msg->notice(2) << "shape_move writint to GSD File to name: "<< name << std::endl;
        std::vector<float> d;
        d.resize(m_step_size.size());
        std::transform(m_step_size.begin(), m_step_size.end(), d.begin(), [](const Scalar& s)->float{ return s; });
        retval |= gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_FLOAT, d.size(), 1, 0, (void *)&d[0]);
        return retval;
        }

    //! Method that is called to connect to the gsd write state signal
    virtual bool restoreStateGSD(   std::shared_ptr<GSDReader> reader,
                                    uint64_t frame,
                                    std::string name,
                                    unsigned int Ntypes,
                                    const std::shared_ptr<const ExecutionConfiguration> exec_conf,
                                    bool mpi)
        {
        bool success = true;
        std::string path = name + "stepsize";
        std::vector<float> d;
        if(exec_conf->isRoot())
            {
            d.resize(Ntypes, 0.0);
            exec_conf->msg->notice(2) << "shape_move reading from GSD File from name: "<< name << std::endl;
            exec_conf->msg->notice(2) << "stepsize: "<< d[0] << " success: " << std::boolalpha << success << std::endl;

            success = reader->readChunk((void *)&d[0], frame, path.c_str(), Ntypes*gsd_sizeof_type(GSD_TYPE_FLOAT), Ntypes) && success;
            exec_conf->msg->notice(2) << "stepsize: "<< d[0] << " success: " << std::boolalpha << success << std::endl;
            }

        #ifdef ENABLE_MPI
        if(mpi)
            {
            bcast(d, 0, exec_conf->getMPICommunicator()); // broadcast the data
            }
        #endif
        if(!d.size() || d.size() != m_step_size.size()) // adding this sanity check but can remove.
            throw std::runtime_error("Error occured while attempting to restore from gsd file.");
        for(unsigned int i = 0; i < d.size(); i++)
            m_step_size[i] = d[i];
        return success;
        }

    //! Returns all of the provided log quantities for the shape move.
    std::vector< std::string > getProvidedLogQuantities()
        {
        return m_ProvidedQuantities;
        }

    //! Calculates the requested log value and returns true if the quantity was
    //! provided by this class.
    virtual bool getLogValue(const std::string& quantity, unsigned int timestep, Scalar& value)
        {
        return false;
        }


protected:
    std::vector< std::string >      m_ProvidedQuantities;
    Scalar                          m_determinantInertiaTensor;     // TODO: REMOVE?
    std::vector<Scalar>             m_step_size;                    // maximum stepsize. input/output
};


// TODO: make this class more general and make python function a spcialization.
template < typename Shape, typename RNG >
class python_callback_parameter_shape_move : public shape_move_function<Shape, RNG>
{
    using shape_move_function<Shape, RNG>::m_determinantInertiaTensor;
    using shape_move_function<Shape, RNG>::m_step_size;
    using shape_move_function<Shape, RNG>::m_ProvidedQuantities;
public:
    python_callback_parameter_shape_move(   unsigned int ntypes,
                                            pybind11::object python_function,
                                            std::vector< std::vector<Scalar> > params,
                                            std::vector<Scalar> stepsize,
                                            Scalar mixratio
                                        )
        :  shape_move_function<Shape, RNG>(ntypes), m_num_params(0), m_params(params), m_python_callback(python_function)
        {
        if(m_step_size.size() != stepsize.size())
            throw std::runtime_error("must provide a stepsize for each type");

        m_step_size = stepsize;
        m_select_ratio = fmin(mixratio, 1.0)*65535;
        m_determinantInertiaTensor = 0.0;
        for(size_t i = 0; i < getNumParam(); i++)
            {
            m_ProvidedQuantities.push_back(getParamName(i));
            }
        }

    void prepare(unsigned int timestep)
        {
        m_params_backup = m_params;
        // m_step_size_backup = m_step_size;
        }

    void construct(const unsigned int& timestep, const unsigned int& type_id, typename Shape::param_type& shape, RNG& rng)
        {
        for(size_t i = 0; i < m_params[type_id].size(); i++)
            {
            Scalar x = ((rng.u32() & 0xffff) < m_select_ratio) ? rng.s(fmax(-m_step_size[type_id], -(m_params[type_id][i])), fmin(m_step_size[type_id], (1.0-m_params[type_id][i]))) : 0.0;
            m_params[type_id][i] += x;
            }
        pybind11::object shape_data = m_python_callback(m_params[type_id]);
        shape = pybind11::cast< typename Shape::param_type >(shape_data);
        detail::mass_properties<Shape> mp(shape);
        m_determinantInertiaTensor = mp.getDeterminant();
        }

    void retreat(unsigned int timestep)
        {
        // move has been rejected.
        std::swap(m_params, m_params_backup);
        }

    Scalar getParam(size_t k)
        {
        size_t n = 0;
        for (size_t i = 0; i < m_params.size(); i++)
            {
            size_t next = n + m_params[i].size();
            if(k < next)
                return m_params[i][k - n];
            n = next;
            }
        throw std::out_of_range("Error: Could not get parameter, index out of range.\n");// out of range.
        return Scalar(0.0);
        }

    size_t getNumParam()
        {
        if(m_num_params > 0 )
            return m_num_params;
        m_num_params = 0;
        for (size_t i = 0; i < m_params.size(); i++)
            m_num_params += m_params[i].size();
        return m_num_params;
        }

    static std::string getParamName(size_t i)
        {
        std::stringstream ss;
        std::string snum;
        ss << i;
        ss>>snum;
        return "shape_param-" + snum;
        }

    //! Calculates the requested log value and returns true if the quantity was
    //! provided by this class.
    bool getLogValue(const std::string& quantity, unsigned int timestep, Scalar& value)
        {
        for(size_t i = 0; i < m_num_params; i++)
            {
            if(quantity == getParamName(i))
                {
                value = getParam(i);
                return true;
                }
            }
        return false;
        }

private:
    std::vector<Scalar>                     m_step_size_backup;
    unsigned int                            m_select_ratio;     // fraction of parameters to change in each move. internal use
    unsigned int                            m_num_params;       // cache the number of parameters.
    Scalar                                  m_scale;            // the scale needed to keep the particle at constant volume. internal use
    std::vector< std::vector<Scalar> >      m_params_backup;    // all params are from 0,1
    std::vector< std::vector<Scalar> >      m_params;           // all params are from 0,1
    pybind11::object                        m_python_callback;  // callback that takes m_params as an argiment and returns (shape, det(I))
    // bool                                    m_normalized;       // if true all parameters are restricted to (0,1)
};

template< typename Shape, typename RNG >
class constant_shape_move : public shape_move_function<Shape, RNG>
{
    using shape_move_function<Shape, RNG>::m_determinantInertiaTensor;
public:
    constant_shape_move(const unsigned int& ntypes, const std::vector< typename Shape::param_type >& shape_move) : shape_move_function<Shape, RNG>(ntypes), m_shapeMoves(shape_move)
        {
        if(ntypes != m_shapeMoves.size())
            throw std::runtime_error("Must supply a shape move for each type");
        for(size_t i = 0; i < m_shapeMoves.size(); i++)
            {
            detail::mass_properties<Shape> mp(m_shapeMoves[i]);
            m_determinants.push_back(mp.getDeterminant());
            }
        }

    void prepare(unsigned int timestep) {}

    void construct(const unsigned int& timestep, const unsigned int& type_id, typename Shape::param_type& shape, RNG& rng)
        {
        shape = m_shapeMoves[type_id];
        m_determinantInertiaTensor = m_determinants[type_id];
        }

    void retreat(unsigned int timestep)
        {
        // move has been rejected.
        }

private:
    std::vector< typename Shape::param_type >   m_shapeMoves;
    std::vector< Scalar >                       m_determinants;
};

template < typename ShapeConvexPolyhedronType, typename RNG >
class convex_polyhedron_generalized_shape_move : public shape_move_function<ShapeConvexPolyhedronType, RNG>
{
    using shape_move_function<ShapeConvexPolyhedronType, RNG>::m_determinantInertiaTensor;
    using shape_move_function<ShapeConvexPolyhedronType, RNG>::m_step_size;
public:
    convex_polyhedron_generalized_shape_move(
                                            unsigned int ntypes,
                                            Scalar stepsize,
                                            Scalar mixratio,
                                            Scalar volume
                                        ) : shape_move_function<ShapeConvexPolyhedronType, RNG>(ntypes), m_volume(volume)
        {
        // if(m_step_size.size() != stepsize.size())
        //     throw std::runtime_error("must provide a stepsize for each type");

        m_determinantInertiaTensor = 1.0;
        m_scale = 1.0;
        std::fill(m_step_size.begin(), m_step_size.end(), stepsize);
        m_calculated.resize(ntypes, false);
        m_centroids.resize(ntypes, vec3<Scalar>(0,0,0));
        m_select_ratio = fmin(mixratio, 1.0)*65535;
        m_step_size_backup = m_step_size;
        }

    void prepare(unsigned int timestep)
        {
        m_step_size_backup = m_step_size;
        }

    void construct(const unsigned int& timestep, const unsigned int& type_id, typename ShapeConvexPolyhedronType::param_type& shape, RNG& rng)
        {
        if(!m_calculated[type_id])
            {
            detail::ConvexHull convex_hull(shape); // compute the convex_hull.
            convex_hull.compute();
            detail::mass_properties<ShapeConvexPolyhedronType> mp(convex_hull.getPoints(), convex_hull.getFaces());
            m_centroids[type_id] = mp.getCenterOfMass();
            m_calculated[type_id] = true;
            }
        // mix the shape.
        for(size_t i = 0; i < shape.N; i++)
            {
            if( (rng.u32()& 0xffff) < m_select_ratio )
                {
                vec3<Scalar> vert(shape.x[i], shape.y[i], shape.z[i]);
                move_translate(vert, rng,  m_step_size[type_id], 3);
                shape.x[i] = vert.x;
                shape.y[i] = vert.y;
                shape.z[i] = vert.z;
                }
            }

        detail::ConvexHull convex_hull(shape); // compute the convex_hull.
        convex_hull.compute();
        detail::mass_properties<ShapeConvexPolyhedronType> mp(convex_hull.getPoints(), convex_hull.getFaces());
        Scalar volume = mp.getVolume();
        vec3<Scalar> dr = m_centroids[type_id] - mp.getCenterOfMass();
        m_scale = pow(m_volume/volume, 1.0/3.0);
        Scalar rsq = 0.0;
        std::vector< vec3<Scalar> > points(shape.N);
        for(size_t i = 0; i < shape.N; i++)
            {
            shape.x[i] += dr.x;
            shape.x[i] *= m_scale;
            shape.y[i] += dr.y;
            shape.y[i] *= m_scale;
            shape.z[i] += dr.z;
            shape.z[i] *= m_scale;
            vec3<Scalar> vert(shape.x[i], shape.y[i], shape.z[i]);
            rsq = fmax(rsq, dot(vert, vert));
            points[i] = vert;
            }
        detail::mass_properties<ShapeConvexPolyhedronType> mp2(points, convex_hull.getFaces());
        m_determinantInertiaTensor = mp2.getDeterminant();
        shape.diameter = 2.0*sqrt(rsq);
        m_step_size[type_id] *= m_scale; // only need to scale if the parameters are not normalized
        }

    void retreat(unsigned int timestep)
        {
        // move has been rejected.
        std::swap(m_step_size, m_step_size_backup);
        }

private:
    std::vector<Scalar>     m_step_size_backup;
    unsigned int            m_select_ratio;
    Scalar                  m_scale;
    Scalar                  m_volume;
    std::vector< vec3<Scalar> > m_centroids;
    std::vector<bool>       m_calculated;
};

//TODO: put the following functions in a class
inline bool isIn(Scalar x, Scalar y, Scalar alpha)
    {
    const Scalar one = 1.0;
    if(x < one && y > one/(alpha*x)) return true;
    else if(x >= one && y < alpha/x) return true;
    return false;
    }


template <class RNG>
inline void generate_scale_R(Scalar& x, Scalar& y, RNG& rng, Scalar alpha)
    {
    do
        {
        x = rng.s(Scalar(1)/alpha, alpha);
        y = rng.s(Scalar(1)/alpha, alpha);
        }while(!isIn(x,y,alpha));
    }
template <class RNG>
inline void generate_scale_S(Scalar& x, Scalar& y, RNG& rng, Scalar alpha)
    {
    Scalar sigma_max = 0.0, sigma = 0.0, U = 0.0;
    sigma_max = sqrt(pow(alpha, 4) + pow(alpha, 2) + 1);
    do
        {
        generate_scale_R(x,y,rng,alpha);
        sigma = sqrt((1.0/(x*x*x*x*y*y)) + (1.0/(x*x*y*y*y*y)) + 1);
        U = rng.s(0.0,1.0);
        }while(U > sigma/sigma_max);
    }

template <class RNG>
inline void generate_scale(Eigen::Matrix3d& S, RNG& rng, Scalar alpha)
    {
    Scalar x = 0.0, y = 0.0, z = 0.0;
    generate_scale_S(x, y, rng, alpha);
    z = 1.0/x/y;
    S << x, 0.0, 0.0,
        0.0, y, 0.0,
        0.0, 0.0, z;
    }

template<class Shape, class RNG>
class elastic_shape_move_function : public shape_move_function<Shape, RNG>
{  // Derived class from shape_move_function base class
    using shape_move_function<Shape, RNG>::m_determinantInertiaTensor;
    using shape_move_function<Shape, RNG>::m_step_size;
    std::vector <Eigen::Matrix3d> m_Fbar_last;
    std::vector <Eigen::Matrix3d> m_Fbar;
public:
    elastic_shape_move_function(
                                    unsigned int ntypes,
                                    const Scalar& stepsize,
                                    Scalar move_ratio
                                ) : shape_move_function<Shape, RNG>(ntypes), m_mass_props(ntypes)
        {
        m_select_ratio = fmin(move_ratio, 1.0)*65535;
        m_step_size.resize(ntypes, stepsize);
        m_Fbar.resize(ntypes, Eigen::Matrix3d::Identity());
        m_Fbar_last.resize(ntypes, Eigen::Matrix3d::Identity());
        std::fill(m_step_size.begin(), m_step_size.end(), stepsize);
        m_determinantInertiaTensor = 1.0;
        }

    void prepare(unsigned int timestep)
        {
        m_Fbar_last = m_Fbar;
        }

    //! construct is called at the beginning of every update()                                            # param was shape - Luis
    void construct(const unsigned int& timestep, const unsigned int& type_id, typename Shape::param_type& param, RNG& rng)
        {
        using Eigen::Matrix3d;
        Matrix3d transform;
        if( (rng.u32()& 0xffff) < m_select_ratio ) // perform a scaling move
            {
            generate_scale(transform, rng, m_step_size[type_id]+1.0);
            }
        else                                        // perform a rotation-scale-rotation move
            {
            quat<Scalar> q(1.0,vec3<Scalar>(0.0,0.0,0.0));
            move_rotate(q, rng, 0.5, 3);
            Matrix3d rot, rot_inv, scale;
            Eigen::Quaternion<double> eq(q.s, q.v.x, q.v.y, q.v.z);
            rot = eq.toRotationMatrix();
            rot_inv = rot.transpose();
            generate_scale(scale, rng, m_step_size[type_id]+1.0);
            transform = rot*scale*rot_inv;
            }

        m_Fbar[type_id] = transform*m_Fbar[type_id];
        Scalar dsq = 0.0;
        for(unsigned int i = 0; i < param.N; i++)
            {
            vec3<Scalar> vert(param.x[i], param.y[i], param.z[i]);
            param.x[i] = transform(0,0)*vert.x + transform(0,1)*vert.y + transform(0,2)*vert.z;
            param.y[i] = transform(1,0)*vert.x + transform(1,1)*vert.y + transform(1,2)*vert.z;
            param.z[i] = transform(2,0)*vert.x + transform(2,1)*vert.y + transform(2,2)*vert.z;
            vert = vec3<Scalar>( param.x[i], param.y[i], param.z[i]);
            dsq = fmax(dsq, dot(vert, vert));
            }
        param.diameter = 2.0*sqrt(dsq);
        m_mass_props[type_id].updateParam(param, false); // update allows caching since for some shapes a full compute is not necessary.
        m_determinantInertiaTensor = m_mass_props[type_id].getDeterminant();
        #ifdef DEBUG
            detail::mass_properties<Shape> mp(param);
            m_determinantInertiaTensor = mp.getDeterminant();
            assert(fabs(m_determinantInertiaTensor-mp.getDeterminant()) < 1e-5);
        #endif
        }

    Eigen::Matrix3d getEps(unsigned int type_id)
        {
        return 0.5*((m_Fbar[type_id].transpose()*m_Fbar[type_id]) - Eigen::Matrix3d::Identity());
        }

    Eigen::Matrix3d getEpsLast(unsigned int type_id)
        {
        return 0.5*((m_Fbar_last[type_id].transpose()*m_Fbar_last[type_id]) - Eigen::Matrix3d::Identity());
        }

    //! retreat whenever the proposed move is rejected.
    void retreat(unsigned int timestep)
        {
        m_Fbar.swap(m_Fbar_last); // we can swap because m_Fbar_last will be reset on the next prepare
        }

protected:
    unsigned int            m_select_ratio;
    std::vector< detail::mass_properties<Shape> > m_mass_props;
};

template<class Shape>
class ShapeLogBoltzmannFunction
{
  public:
    ShapeLogBoltzmannFunction(){};
    virtual Scalar operator()(
                                const unsigned int& timestep,
                                const unsigned int& N,
                                const unsigned int type_id,
                                const typename Shape::param_type& shape_new,
                                const Scalar& inew,
                                const typename Shape::param_type& shape_old,
                                const Scalar& iold)
        {
        throw std::runtime_error("not implemented");
        return 0.0;
        }

    virtual Scalar computeEnergy(
                                    const unsigned int& timestep,
                                    const unsigned int& N,
                                    const unsigned int type_id,
                                    const typename Shape::param_type& shape,
                                    const Scalar& inertia)
        {
        return 0.0;
        }

    //! Returns all of the provided log quantities for the shape move.
    std::vector< std::string > getProvidedLogQuantities()
        {
        return m_ProvidedQuantities;
        }

    //! Calculates the requested log value and returns true if the quantity was
    //! provided by this class.
    virtual bool getLogValue(const std::string& quantity, unsigned int timestep, Scalar& value)
        {
        return false;
        }

protected:
    std::vector< std::string >      m_ProvidedQuantities;
};

template<class Shape>
class AlchemyLogBoltzmannFunction : public ShapeLogBoltzmannFunction<Shape>
{
public:
    virtual Scalar operator()(const unsigned int& timestep, const unsigned int& N,const unsigned int type_id, const typename Shape::param_type& shape_new, const Scalar& inew, const typename Shape::param_type& shape_old, const Scalar& iold)
        {
        return (Scalar(N)/Scalar(2.0))*log(inew/iold);
        }
};

#define SHAPE_SPRING_STIFFNESS "shape_move_stiffness"

template< class Shape >
class ShapeSpringBase : public ShapeLogBoltzmannFunction<Shape>
{
protected:
    Scalar m_volume;
    std::unique_ptr<typename Shape::param_type> m_reference_shape;
    std::shared_ptr<Variant> m_k;
    using ShapeLogBoltzmannFunction<Shape>::m_ProvidedQuantities;
public:

    ShapeSpringBase(std::shared_ptr<Variant> k, typename Shape::param_type shape) : m_reference_shape(new typename Shape::param_type), m_k(k)
        {
        (*m_reference_shape) = shape;
        detail::mass_properties<Shape> mp(*m_reference_shape);
        m_volume = mp.getVolume();
        m_ProvidedQuantities.push_back(SHAPE_SPRING_STIFFNESS);
        }

    void setStiffness(std::shared_ptr<Variant> stiff)
        {
        m_k = stiff;
        }

    std::shared_ptr<Variant> getStiffness() const
        {
        return m_k;
        }

    //! Calculates the requested log value and returns true if the quantity was
    //! provided by this class.
    virtual bool getLogValue(const std::string& quantity, unsigned int timestep, Scalar& value)
        {
        if(quantity == SHAPE_SPRING_STIFFNESS)
            {
            value = m_k->getValue(timestep);
            return true;
            }
        return false;
        }
};

template<class Shape>
class ShapeSpring : public ShapeSpringBase< Shape >
{
    using ShapeSpringBase< Shape >::m_reference_shape;
    using ShapeSpringBase< Shape >::m_volume;
    //using elastic_shape_move_function<Shape, hoomd::detail::Saru>;
    std::shared_ptr<elastic_shape_move_function<Shape, hoomd::detail::Saru> > m_shape_move;
public:
    using ShapeSpringBase< Shape >::m_k;
    ShapeSpring(std::shared_ptr<Variant> k, typename Shape::param_type ref, std::shared_ptr<elastic_shape_move_function<Shape, hoomd::detail::Saru> > P) : ShapeSpringBase <Shape> (k, ref ) , m_shape_move(P)
        {
        }

    Scalar operator()(const unsigned int& timestep, const unsigned int& N, const unsigned int type_id ,const typename Shape::param_type& shape_new, const Scalar& inew, const typename Shape::param_type& shape_old, const Scalar& iold)
        {
        Scalar stiff = this->m_k->getValue(timestep);
        Eigen::Matrix3d eps = m_shape_move->getEps(type_id);
        Eigen::Matrix3d eps_last = m_shape_move->getEpsLast(type_id);
        AlchemyLogBoltzmannFunction< Shape > fn;
        Scalar e_ddot_e = 0.0, e_ddot_e_last = 0.0;
        e_ddot_e = eps(0,0)*eps(0,0) + eps(0,1)*eps(1,0) + eps(0,2)*eps(2,0) +
                 eps(1,0)*eps(0,1) + eps(1,1)*eps(1,1) + eps(1,2)*eps(2,1) +
                 eps(2,0)*eps(0,2) + eps(2,1)*eps(1,2) + eps(2,2)*eps(2,2);

        e_ddot_e_last = eps_last(0,0)*eps_last(0,0) + eps_last(0,1)*eps_last(1,0) + eps_last(0,2)*eps_last(2,0) +
                 eps_last(1,0)*eps_last(0,1) + eps_last(1,1)*eps_last(1,1) + eps_last(1,2)*eps_last(2,1) +
                 eps_last(2,0)*eps_last(0,2) + eps_last(2,1)*eps_last(1,2) + eps_last(2,2)*eps_last(2,2) ;
        // TODO: To make this more correct we need to calculate the previous volume and multiply accodingly.
        return N*stiff*(e_ddot_e_last-e_ddot_e)*m_volume + fn(timestep, N, type_id, shape_new, inew, shape_old, iold); // -\beta dH
        }

    Scalar computeEnergy(const unsigned int &timestep, const unsigned int& N, const unsigned int type_id, const typename Shape::param_type& shape, const Scalar& inertia)
        {
        Scalar stiff = m_k->getValue(timestep);
        Eigen::Matrix3d eps = m_shape_move->getEps(type_id);
        Scalar e_ddot_e = 0.0;
        e_ddot_e = eps(0,0)*eps(0,0) + eps(0,1)*eps(1,0) + eps(0,2)*eps(2,0) +
                 eps(1,0)*eps(0,1) + eps(1,1)*eps(1,1) + eps(1,2)*eps(2,1) +
                 eps(2,0)*eps(0,2) + eps(2,1)*eps(1,2) + eps(2,2)*eps(2,2);
        return N*stiff*e_ddot_e*m_volume;
        }
};

template<class Shape>
void export_ShapeMoveInterface(pybind11::module& m, const std::string& name);

template<class Shape>
void export_ScaleShearShapeMove(pybind11::module& m, const std::string& name);

template< typename Shape >
void export_ShapeLogBoltzmann(pybind11::module& m, const std::string& name);

template<class Shape>
void export_ShapeSpringLogBoltzmannFunction(pybind11::module& m, const std::string& name);

template<class Shape>
void export_AlchemyLogBoltzmannFunction(pybind11::module& m, const std::string& name);

template<class Shape>
void export_ConvexPolyhedronGeneralizedShapeMove(pybind11::module& m, const std::string& name);

template<class Shape>
void export_PythonShapeMove(pybind11::module& m, const std::string& name);

template<class Shape>
void export_ConstantShapeMove(pybind11::module& m, const std::string& name);

}

#endif
