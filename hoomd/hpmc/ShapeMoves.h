#ifndef _SHAPE_MOVES_H
#define _SHAPE_MOVES_H

#include "ShapeUtils.h"
#include <hoomd/Variant.h>
#include "Moves.h"
#include "GSDHPMCSchema.h"
#include <hoomd/extern/Eigen/Eigen/Dense>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace hpmc {


template <typename Shape>
class ShapeMoveBase
{
public:
    ShapeMoveBase(unsigned int ntypes) :
        m_det_inertia_tensor(0),
        m_step_size(ntypes)
        {
        }

    ShapeMoveBase(const ShapeMoveBase& src) :
        m_det_inertia_tensor(src.getDeterminantInertiaTensor()),
        m_step_size(src.getStepSizeArray())
        {
        }

    //! prepare is called at the beginning of every update()
    virtual void prepare(unsigned int timestep)
        {
        throw std::runtime_error("Shape move function not implemented.");
        }

    //! construct is called for each particle type that will be changed in update()
    virtual void construct(const unsigned int&, const unsigned int&, typename Shape::param_type&, hoomd::RandomGenerator&)
        {
        throw std::runtime_error("Shape move function not implemented.");
        }

    //! retreat whenever the proposed move is rejected.
    virtual void retreat(const unsigned int)
        {
        throw std::runtime_error("Shape move function not implemented.");
        }

    // TODO: remove this?
    Scalar getDeterminant() const
        {
        return m_det_inertia_tensor;
        }

    // Get the isoperimetric quotient of the shape
    Scalar getIsoperimetricQuotient() const
        {
        return m_isoperimetric_quotient;
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
        std::string path = name + "stepsize";
        exec_conf->msg->notice(2) << "shape_move writing to GSD File to name: "<< name << std::endl;
        std::vector<float> d;
        d.resize(m_step_size.size());
        std::transform(m_step_size.begin(), m_step_size.end(), d.begin(), [](const Scalar& s)->float{ return s; });
        int retval = gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_FLOAT, d.size(), 1, 0, (void *)&d[0]);
        return retval;
        }

    //! Method that is called to connect to the gsd write state signal
    virtual bool restoreStateGSD(   std::shared_ptr<GSDReader> reader,
                                    std::string name,
                                    const std::shared_ptr<const ExecutionConfiguration> exec_conf,
                                    bool mpi)
        {
        bool success;
        std::string path = name + "stepsize";
        std::vector<float> d;
        unsigned int Ntypes = this->m_step_size.size();
        uint64_t frame = reader->getFrame();
        if(exec_conf->isRoot())
            {
            d.resize(Ntypes, 0.0);
            exec_conf->msg->notice(2) << "shape_move reading from GSD File from name: "<< name << std::endl;
            success = reader->readChunk((void *)&d[0], frame, path.c_str(), Ntypes*gsd_sizeof_type(GSD_TYPE_FLOAT), Ntypes);
            exec_conf->msg->notice(2) << "stepsize: "<< d[0] << " success: " << std::boolalpha << success << std::endl;
            }

        #ifdef ENABLE_MPI
        if(mpi)
            {
            bcast(d, 0, exec_conf->getMPICommunicator()); // broadcast the data
            }
        #endif

        for(unsigned int i = 0; i < d.size(); i++)
            m_step_size[i] = Scalar(d[i]);

        return success;

        }

    //! Returns all of the provided log quantities for the shape move.
    std::vector< std::string > getProvidedLogQuantities()
        {
        return m_provided_quantities;
        }

    //! Calculates the requested log value and returns it
    virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep)
        {
        return 0.0;
        }

    //! Checks if the requested log value is provided
    virtual bool isProvidedQuantity(const std::string& quantity)
        {
        return false;
        }

protected:
    std::vector< std::string >      m_provided_quantities;
    Scalar                          m_det_inertia_tensor;     // TODO: REMOVE?
    Scalar                          m_isoperimetric_quotient;
    std::vector<Scalar>             m_step_size;                    // maximum stepsize. input/output
};   // end class ShapeMoveBase


// TODO: make this class more general and make python function a spcialization.
template < typename Shape >
class PythonShapeMove : public ShapeMoveBase<Shape>
{
public:
    PythonShapeMove(unsigned int ntypes,
                    pybind11::object python_function,
                    std::vector< std::vector<Scalar> > params,
                    std::vector<Scalar> stepsize,
                    Scalar mixratio)
        :  ShapeMoveBase<Shape>(ntypes), m_num_params(0), m_params(params), m_python_callback(python_function)
        {
        if(this->m_step_size.size() != stepsize.size())
            throw std::runtime_error("must provide a stepsize for each type");

        this->m_step_size = stepsize;
        m_select_ratio = fmin(mixratio, 1.0)*65535;
        this->m_det_inertia_tensor = 0.0;
        for(size_t i = 0; i < getNumParam(); i++)
            {
            this->m_provided_quantities.push_back(getParamName(i));
            }
        }

    void prepare(unsigned int timestep)
        {
        m_params_backup = m_params;
        }

    void construct(const unsigned int& timestep,
                   const unsigned int& type_id,
                   typename Shape::param_type& shape,
                   hoomd::RandomGenerator& rng)
        {
        for(size_t i = 0; i < m_params[type_id].size(); i++)
            {
            hoomd::UniformDistribution<Scalar> uniform(
                    fmax(-this->m_step_size[type_id], -(m_params[type_id][i])),
                    fmin(this->m_step_size[type_id], (1.0-m_params[type_id][i])));
            Scalar x = (hoomd::UniformIntDistribution(0xffff)(rng) < m_select_ratio) ? uniform(rng) : 0.0;
            m_params[type_id][i] += x;
            }
        pybind11::object shape_data = m_python_callback(m_params[type_id]);
        shape = pybind11::cast< typename Shape::param_type >(shape_data);
        detail::mass_properties<Shape> mp(shape);
        this->m_det_inertia_tensor = mp.getDeterminant();
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

    //! Calculates the requested log value and returns it
    virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep)
        {
        for(size_t i = 0; i < m_num_params; i++)
            {
            if(quantity == getParamName(i))
                {
                return getParam(i);
                }
            }
        }

private:
    std::vector<Scalar>                     m_step_size_backup;
    unsigned int                            m_select_ratio;     // fraction of parameters to change in each move. internal use
    unsigned int                            m_num_params;       // cache the number of parameters.
    Scalar                                  m_scale;            // the scale needed to keep the particle at constant volume. internal use
    std::vector< std::vector<Scalar> >      m_params_backup;    // all params are from 0,1
    std::vector< std::vector<Scalar> >      m_params;           // all params are from 0,1
    pybind11::object                        m_python_callback;  // callback that takes m_params as an argiment and returns (shape, det(I))
};

template< typename Shape >
class ConstantShapeMove : public ShapeMoveBase<Shape>
{
public:
    ConstantShapeMove(const unsigned int& ntypes,
                        const std::vector< typename Shape::param_type >& shape_move)
        : ShapeMoveBase<Shape>(ntypes), m_shape_moves(shape_move)
        {
        if(ntypes != m_shape_moves.size())
            throw std::runtime_error("Must supply a shape move for each type");
        for(size_t i = 0; i < m_shape_moves.size(); i++)
            {
            detail::mass_properties<Shape> mp(m_shape_moves[i]);
            m_determinants.push_back(mp.getDeterminant());
            }
        }

    void prepare(unsigned int timestep) {}

    void construct(const unsigned int& timestep,
                   const unsigned int& type_id,
                   typename Shape::param_type& shape,
                   hoomd::RandomGenerator& rng)
        {
        shape = m_shape_moves[type_id];
        this->m_det_inertia_tensor = m_determinants[type_id];
        }

    void retreat(unsigned int timestep)
        {
        // move has been rejected.
        }

private:
    std::vector< typename Shape::param_type >   m_shape_moves;
    std::vector< Scalar >                       m_determinants;
};


class ConvexPolyhedronVertexShapeMove : public ShapeMoveBase<ShapeConvexPolyhedron>
{
public:
    ConvexPolyhedronVertexShapeMove(unsigned int ntypes,
                                             Scalar stepsize,
                                             Scalar mixratio,
                                             Scalar volume)
        : ShapeMoveBase<ShapeConvexPolyhedron>(ntypes), m_volume(volume)
        {
        this->m_det_inertia_tensor = 1.0;
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

    void construct(const unsigned int& timestep,
                   const unsigned int& type_id,
                   typename ShapeConvexPolyhedron::param_type& shape,
                   hoomd::RandomGenerator& rng)
        {
        if(!m_calculated[type_id])
            {
            detail::ConvexHull convex_hull(shape); // compute the convex_hull.
            convex_hull.compute();
            detail::mass_properties<ShapeConvexPolyhedron> mp(convex_hull.getPoints(), convex_hull.getFaces());
            m_centroids[type_id] = mp.getCenterOfMass();
            m_calculated[type_id] = true;
            }
        // mix the shape.
        for(size_t i = 0; i < shape.N; i++)
            {
            if( hoomd::UniformIntDistribution(0xffff)(rng) < m_select_ratio )
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
        detail::mass_properties<ShapeConvexPolyhedron> mp(convex_hull.getPoints(), convex_hull.getFaces());
        Scalar volume = mp.getVolume();
        vec3<Scalar> dr = m_centroids[type_id] - mp.getCenterOfMass();
        m_scale = fast::pow(m_volume/volume, 1.0/3.0);
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
        detail::mass_properties<ShapeConvexPolyhedron> mp2(points, convex_hull.getFaces());
        this->m_det_inertia_tensor = mp2.getDeterminant();
        m_isoperimetric_quotient = mp2.getIsoperimetricQuotient();
        shape.diameter = 2.0*fast::sqrt(rsq);
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
};   // end class ConvexPolyhedronVertexShapeMove

template<class Shape>
class ElasticShapeMove : public ShapeMoveBase<Shape>
    {

    public:
        ElasticShapeMove(
                         unsigned int ntypes,
                         const Scalar& stepsize,
                         Scalar move_ratio
                        )
            : ShapeMoveBase<Shape>(ntypes), m_mass_props(ntypes)
            {
            m_select_ratio = fmin(move_ratio, 1.0)*65535;
            this->m_step_size.resize(ntypes, stepsize);
            m_Fbar.resize(ntypes, Eigen::Matrix3d::Identity());
            m_Fbar_last.resize(ntypes, Eigen::Matrix3d::Identity());
            std::fill(this->m_step_size.begin(), this->m_step_size.end(), stepsize);
            this->m_det_inertia_tensor = 1.0;
            }

        void prepare(unsigned int timestep)
            {
            m_Fbar_last = m_Fbar;
            }

        //! construct is called at the beginning of every update()
        void construct(const unsigned int& timestep,
                const unsigned int& type_id,
                typename Shape::param_type& param,
                hoomd::RandomGenerator& rng)
            {
            using Eigen::Matrix3d;
            Matrix3d transform;
            if( hoomd::UniformIntDistribution(0xffff)(rng) < m_select_ratio ) // perform a scaling move
                {
                generateExtentional(transform, rng, this->m_step_size[type_id]+1.0);
                }
            else                                        // perform a rotation-scale-rotation move
                {
                quat<Scalar> q(1.0,vec3<Scalar>(0.0,0.0,0.0));
                move_rotate(q, rng, 0.5, 3);
                Matrix3d rot, rot_inv, scale;
                Eigen::Quaternion<double> eq(q.s, q.v.x, q.v.y, q.v.z);
                rot = eq.toRotationMatrix();
                rot_inv = rot.transpose();
                generateExtentional(scale, rng, this->m_step_size[type_id]+1.0);
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
            param.diameter = 2.0*fast::sqrt(dsq);
            m_mass_props[type_id].updateParam(param, false); // update allows caching since for some shapes a full compute is not necessary.
            this->m_det_inertia_tensor = m_mass_props[type_id].getDeterminant();
            #ifdef DEBUG
                detail::mass_properties<Shape> mp(param);
                this->m_det_inertia_tensor = mp.getDeterminant();
                assert(fabs(this->m_det_inertia_tensor-mp.getDeterminant()) < 1e-5);
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

        //! Method that is called whenever the GSD file is written if connected to a GSD file.
        int writeGSD(gsd_handle& handle, std::string name, const std::shared_ptr<const ExecutionConfiguration> exec_conf, bool mpi) const
            {

            if(!exec_conf->isRoot())
                return 0;

            // Call base method for stepsize
            int retval = ShapeMoveBase<Shape>::writeGSD(handle, name, exec_conf, mpi);
            // flatten deformation matrix before writting to GSD
            unsigned int Ntypes = this->m_step_size.size();
            int rows = Ntypes*3;
            std::vector<float> data(rows*3);
            size_t count = 0;
            for(unsigned int i = 0; i < Ntypes; i++)
                {
                for (unsigned int j = 0; j < 3; j++)
                    {
                    data[count*3+0] = float(m_Fbar[i](0,j));
                    data[count*3+1] = float(m_Fbar[i](1,j));
                    data[count*3+2] = float(m_Fbar[i](2,j));
                    count++;
                  };
                };
            std::string path = name + "defmat";
            exec_conf->msg->notice(2) << "shape_move writing to GSD File to name: "<< name << std::endl;
            retval |= gsd_write_chunk(&handle, path.c_str(), GSD_TYPE_FLOAT, rows, 3, 0, (void *)&data[0]);
            return retval;
            };

        //! Method that is called to connect to the gsd write state signal
        virtual bool restoreStateGSD(   std::shared_ptr<GSDReader> reader,
                                        std::string name,
                                        const std::shared_ptr<const ExecutionConfiguration> exec_conf,
                                        bool mpi)
            {
            // Call base method for stepsize
            bool success = ShapeMoveBase<Shape>::restoreStateGSD(reader, name, exec_conf, mpi);
            unsigned int Ntypes = this->m_step_size.size();
            uint64_t frame = reader->getFrame();
            std::vector<float> defmat(Ntypes*3*3,0.0);
            if(exec_conf->isRoot())
                {
                std::string path = name + "defmat";
                exec_conf->msg->notice(2) << "shape_move reading from GSD File from name: "<< name << std::endl;
                success = reader->readChunk((void *)&defmat[0], frame, path.c_str(), 3*3*Ntypes*gsd_sizeof_type(GSD_TYPE_FLOAT), 3*Ntypes) && success;
                exec_conf->msg->notice(2) << "defmat success: " << std::boolalpha << success << std::endl;
                }

            #ifdef ENABLE_MPI
            if(mpi)
                {
                bcast(defmat, 0, exec_conf->getMPICommunicator());
                }
            #endif

            if(defmat.size() != (this->m_Fbar).size()*3*3)
                {
                throw std::runtime_error("Error occured while attempting to restore from gsd file.");
                }

            size_t count = 0;
            for(unsigned int i = 0; i < (this->m_Fbar).size(); i++)
                {
                for (unsigned int j = 0; j < 3; j++)
                    {
                    this->m_Fbar[i](0,j) = defmat[count*3+0];
                    this->m_Fbar[i](1,j) = defmat[count*3+1];
                    this->m_Fbar[i](2,j) = defmat[count*3+2];
                    count++;
                    }
                }

            return success;
            };

        protected:
            unsigned int m_select_ratio;
            std::vector< detail::mass_properties<Shape> > m_mass_props;
            std::vector <Eigen::Matrix3d> m_Fbar_last;
            std::vector <Eigen::Matrix3d> m_Fbar;

        private:

            // These are ElasticShapeMove specific helper functions to randomly
            // sample point on the XYZ=1 surface from a uniform distribution

            //! Check if a point (x,y) lies in the projection of xyz=1 surface
            //! on the xy plane
            inline bool inInSurfaceProjection(Scalar x, Scalar y, Scalar alpha)
                {
                if(x < Scalar(1.0) && y > Scalar(1.0)/(alpha*x))
                    return true;
                else if(x >= Scalar(1.0) && y < alpha/x)
                    return true;
                else
                    return false;
                }

             //! Sample points on the projection of xyz=1
            inline void sampleOnSurfaceProjection(Scalar& x,
                                                  Scalar& y,
                                                  hoomd::RandomGenerator& rng,
                                                  Scalar alpha)
                {
                hoomd::UniformDistribution<Scalar> uniform(Scalar(1)/alpha, alpha);
                do
                    {
                    x = uniform(rng);
                    y = uniform(rng);
                    }while(!inInSurfaceProjection(x,y,alpha));
                }

            //! Sample points on the projection of xyz=1 surface
            inline void sampleOnSurface(Scalar& x, Scalar& y, hoomd::RandomGenerator& rng, Scalar alpha)
                {
                Scalar sigma_max = 0.0, sigma = 0.0, U = 0.0;
                Scalar alpha2 = alpha*alpha;
                Scalar alpha4 = alpha2*alpha2;
                sigma_max = fast::sqrt(alpha4 + alpha2 + 1);
                do
                    {
                    sampleOnSurfaceProjection(x,y,rng,alpha);
                    sigma = fast::sqrt((1.0/(x*x*x*x*y*y)) + (1.0/(x*x*y*y*y*y)) + 1);
                    U = hoomd::detail::generate_canonical<Scalar>(rng);
                    }while(U > sigma/sigma_max);
                }

            //! Generate an volume conserving extentional deformation matrix
            inline void generateExtentional(Eigen::Matrix3d& S, hoomd::RandomGenerator& rng, Scalar alpha)
                {
                Scalar x = 0.0, y = 0.0, z = 0.0;
                sampleOnSurface(x, y, rng, alpha);
                z = Scalar(1.0)/x/y;
                S << x, 0.0, 0.0,
                     0.0, y, 0.0,
                     0.0, 0.0, z;
                }
    };

template <>
class ElasticShapeMove<ShapeEllipsoid> : public ShapeMoveBase<ShapeEllipsoid>
    {

    public:

        ElasticShapeMove(unsigned int ntypes,
                         Scalar stepsize,
                         Scalar move_ratio)
                         : ShapeMoveBase<ShapeEllipsoid>(ntypes),
                         m_mass_props(ntypes), m_move_ratio(move_ratio)
            {
            this->m_step_size.resize(ntypes, stepsize);
            std::fill(m_step_size.begin(), m_step_size.end(), stepsize);
            }

        void construct(const unsigned int& timestep, const unsigned int& type_id,
                       typename ShapeEllipsoid::param_type& param, hoomd::RandomGenerator& rng)
            {
            Scalar lnx = log(param.x/param.y);
            Scalar dlnx = hoomd::UniformDistribution<Scalar>(-m_step_size[type_id], m_step_size[type_id])(rng);
            Scalar x = fast::exp(lnx+dlnx);
            m_mass_props[type_id].updateParam(param);
            Scalar volume = m_mass_props[type_id].getVolume();
            Scalar vol_factor = detail::mass_properties<ShapeEllipsoid>::m_vol_factor;
            Scalar b = fast::pow(volume/vol_factor/x, 1.0/3.0);
            param.x = x*b;
            param.y = b;
            param.z = b;
            }

        void prepare(unsigned int timestep)
            {
            }

        void retreat(unsigned int timestep)
            {
            }

    private:
        std::vector< detail::mass_properties<ShapeEllipsoid> > m_mass_props;
        Scalar m_move_ratio;
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
        return m_provided_quantities;
        }

    //! Calculates the requested log value and returns it
    virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep)
        {
        return 0.0;
        }

    virtual bool isProvidedQuantity(const std::string& quantity)
        {
        return false;
        }

protected:
    std::vector< std::string >      m_provided_quantities;
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

template< class Shape >
class ShapeSpringBase : public ShapeLogBoltzmannFunction<Shape>
{
protected:
    Scalar m_volume;
    std::unique_ptr<typename Shape::param_type> m_reference_shape;
    std::shared_ptr<Variant> m_k;
    using ShapeLogBoltzmannFunction<Shape>::m_provided_quantities;
public:

    ShapeSpringBase(std::shared_ptr<Variant> k, typename Shape::param_type shape) : m_reference_shape(new typename Shape::param_type), m_k(k)
        {
        (*m_reference_shape) = shape;
        detail::mass_properties<Shape> mp(*m_reference_shape);
        m_volume = mp.getVolume();
        m_provided_quantities.push_back("shape_move_stiffness");
        }

    void setStiffness(std::shared_ptr<Variant> stiff)
        {
        m_k = stiff;
        }

    std::shared_ptr<Variant> getStiffness() const
        {
        return m_k;
        }

    //! Calculates the requested log value and returns it
    virtual Scalar getLogValue(const std::string& quantity, unsigned int timestep)
        {
        if(quantity == "shape_move_stiffness")
            {
            return m_k->getValue(timestep);
            }
        }

    //! Checks if the requested log value is provided
    virtual bool isProvidedQuantity(const std::string& quantity)
        {
        if(std::find(m_provided_quantities.begin(), m_provided_quantities.end(), quantity)
           != m_provided_quantities.end())
            {
            return true;
            }
        return false;
        }
};

template<class Shape>
class ShapeSpring : public ShapeSpringBase< Shape >
{
    std::shared_ptr<ElasticShapeMove<Shape> > m_shape_move;
public:
    ShapeSpring(std::shared_ptr<Variant> k,
                typename Shape::param_type ref,
                std::shared_ptr<ElasticShapeMove<Shape> > P)
        : ShapeSpringBase <Shape> (k, ref ) , m_shape_move(P)
        {
        }

    Scalar operator()(const unsigned int& timestep, const unsigned int& N, const unsigned int type_id ,const typename Shape::param_type& shape_new, const Scalar& inew, const typename Shape::param_type& shape_old, const Scalar& iold)
        {
        Scalar stiff = this->m_k->getValue(timestep);
        Eigen::Matrix3d eps = m_shape_move->getEps(type_id);
        Eigen::Matrix3d eps_last = m_shape_move->getEpsLast(type_id);
        AlchemyLogBoltzmannFunction< Shape > fn;
        Scalar e_ddot_e = (eps*eps.transpose()).trace();
        Scalar e_ddot_e_last = (eps_last*eps_last.transpose()).trace();
        // TODO: To make this more correct we need to calculate the previous volume and multiply accodingly.
        return N*stiff*(e_ddot_e_last-e_ddot_e)*this->m_volume
               + fn(timestep, N, type_id, shape_new, inew, shape_old, iold);
        }

    Scalar computeEnergy(const unsigned int &timestep, const unsigned int& N, const unsigned int type_id, const typename Shape::param_type& shape, const Scalar& inertia)
        {
        Scalar stiff = this->m_k->getValue(timestep);
        Eigen::Matrix3d eps = m_shape_move->getEps(type_id);
        Scalar e_ddot_e = (eps*eps.transpose()).trace();
        return N*stiff*e_ddot_e*this->m_volume;
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
