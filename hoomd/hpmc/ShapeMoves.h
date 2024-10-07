// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _SHAPE_MOVES_H
#define _SHAPE_MOVES_H

#include "GSDHPMCSchema.h"
#include "Moves.h"
#include "ShapeUtils.h"
#include "hoomd/extern/quickhull/QuickHull.hpp"
#include <Eigen/Dense>
#include <hoomd/Variant.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace hoomd
    {

namespace hpmc
    {

template<typename Shape> class ShapeMoveBase
    {
    public:
    ShapeMoveBase(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<IntegratorHPMCMono<Shape>> mc)
        : m_mc(mc), m_sysdef(sysdef)
        {
        m_ntypes = this->m_sysdef->getParticleData()->getNTypes();
        m_volume.resize(m_ntypes, 0);
        m_step_size.resize(m_ntypes, 0);
        }

    virtual ~ShapeMoveBase() { };

    //! prepare is called at the beginning of every update()
    virtual void prepare(uint64_t timestep) { }

    //! construct is called for each particle type that will be changed in update()
    virtual void update_shape(uint64_t,
                              const unsigned int&,
                              typename Shape::param_type&,
                              hoomd::RandomGenerator&,
                              bool managed)
        {
        }

    //! retreat whenever the proposed move is rejected.
    virtual void retreat(uint64_t timestep, unsigned int type) { }

    Scalar getStepSize(std::string typ)
        {
        unsigned int typid = getValidateType(typ);
        return this->m_step_size[typid];
        }

    void setStepSize(std::string typ, Scalar volume)
        {
        unsigned int typid = getValidateType(typ);
        this->m_step_size[typid] = volume;
        }

    Scalar getVolume(std::string typ)
        {
        unsigned int typid = getValidateType(typ);
        return this->m_volume[typid];
        }

    void setVolume(std::string typ, Scalar volume)
        {
        unsigned int typid = getValidateType(typ);
        this->m_volume[typid] = volume;
        }

    unsigned int getValidateType(std::string typ)
        {
        unsigned int typid = this->m_sysdef->getParticleData()->getTypeByName(typ);
        if (typid >= this->m_ntypes)
            {
            throw std::runtime_error("Invalid particle type.");
            }
        return typid;
        }

    Scalar getMoveProbability()
        {
        return this->m_move_probability;
        }

    void setMoveProbability(Scalar move_probability)
        {
        this->m_move_probability = fmin(move_probability, 1.0);
        }

    virtual Scalar computeLogBoltmann(uint64_t timestep,
                                      const unsigned int& N,
                                      const unsigned int type_id,
                                      const typename Shape::param_type& shape_new,
                                      const Scalar& inew,
                                      const typename Shape::param_type& shape_old,
                                      const Scalar& iold)
        {
        return (Scalar(N) / Scalar(2.0)) * log(inew / iold);
        }

    virtual Scalar computeEnergy(uint64_t timestep,
                                 const unsigned int& N,
                                 const unsigned int type_id,
                                 const typename Shape::param_type& shape,
                                 const Scalar& inertia)
        {
        return 0.0;
        }

    protected:
    std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc;
    std::shared_ptr<SystemDefinition> m_sysdef;
    unsigned m_ntypes;
    std::vector<Scalar> m_volume;
    std::vector<Scalar> m_step_size;
    Scalar m_move_probability;
    std::vector<vec3<Scalar>> m_centroids;

    }; // end class ShapeMoveBase

template<typename Shape> class PythonShapeMove : public ShapeMoveBase<Shape>
    {
    public:
    PythonShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<IntegratorHPMCMono<Shape>> mc)
        : ShapeMoveBase<Shape>(sysdef, mc)
        {
        m_params.resize(this->m_ntypes);
        m_params_backup.resize(this->m_ntypes);
        }

    void prepare(uint64_t timestep)
        {
        m_params_backup = m_params;
        }

    void update_shape(uint64_t timestep,
                      const unsigned int& type_id,
                      typename Shape::param_type& shape,
                      hoomd::RandomGenerator& rng,
                      bool managed)
        {
        for (unsigned int i = 0; i < m_params[type_id].size(); i++)
            {
            Scalar stepsize = this->m_step_size[type_id];
            hoomd::UniformDistribution<Scalar> uniform(-stepsize, stepsize);
            Scalar r = hoomd::detail::generate_canonical<double>(rng);
            Scalar x = (r < this->m_move_probability) ? uniform(rng) : 0.0;
            // Reflect trial moves about boundaries
            if (m_params[type_id][i] + x > 1)
                {
                m_params[type_id][i] = 2 - (m_params[type_id][i] + x);
                }
            else if (m_params[type_id][i] + x < 0)
                {
                m_params[type_id][i] = -(m_params[type_id][i] + x);
                }
            else
                {
                m_params[type_id][i] += x;
                }
            }
        pybind11::object d = m_python_callback(type_id, m_params[type_id]);
        pybind11::dict shape_dict = pybind11::cast<pybind11::dict>(d);
        shape = typename Shape::param_type(shape_dict, managed);
        }

    void retreat(uint64_t timestep, unsigned int type)
        {
        // move has been rejected.
        m_params[type] = m_params_backup[type];
        }

    pybind11::list getParams(std::string typ)
        {
        unsigned int type_id = this->getValidateType(typ);
        pybind11::list ret;
        for (unsigned int i = 0; i < m_params[type_id].size(); i++)
            {
            ret.append(m_params[type_id][i]);
            }
        return ret;
        }

    void setParams(std::string typ, pybind11::list params)
        {
        unsigned int type_id = this->getValidateType(typ);
        auto N = pybind11::len(params);
        m_params[type_id].resize(N);
        m_params_backup[type_id].resize(N);
        for (unsigned int i = 0; i < N; i++)
            {
            m_params[type_id][i] = params[i].cast<Scalar>();
            m_params_backup[type_id][i] = params[i].cast<Scalar>();
            }
        }

    pybind11::object getCallback()
        {
        return m_python_callback;
        }

    void setCallback(pybind11::object python_callback)
        {
        m_python_callback = python_callback;
        }

    private:
    std::vector<std::vector<Scalar>>
        m_params_backup;                       // tunable shape parameters to perform trial moves on
    std::vector<std::vector<Scalar>> m_params; // tunable shape parameters to perform trial moves on
    // callback that takes m_params as an argument and returns a Python dictionary of shape params.
    pybind11::object m_python_callback;
    };

class ConvexPolyhedronVertexShapeMove : public ShapeMoveBase<ShapeConvexPolyhedron>
    {
    public:
    typedef typename ShapeConvexPolyhedron::param_type param_type;

    ConvexPolyhedronVertexShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                                    std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>> mc)
        : ShapeMoveBase<ShapeConvexPolyhedron>(sysdef, mc)
        {
        this->m_centroids.resize(this->m_ntypes, vec3<Scalar>(0, 0, 0));
        initializeMassProperties();
        }

    void initializeMassProperties()
        {
        auto& mc_params = this->m_mc->getParams();
        for (unsigned int i = 0; i < this->m_ntypes; i++)
            {
            detail::MassProperties<ShapeConvexPolyhedron> mp(mc_params[i]);
            this->m_centroids[i] = mp.getCenterOfMass();
            }
        }

    void setVolume(std::string typ, Scalar volume)
        {
        unsigned int typid = getValidateType(typ);
        this->m_volume[typid] = volume;
        auto& mc_params = this->m_mc->getParams();
        param_type shape = mc_params[typid];
        detail::MassProperties<ShapeConvexPolyhedron> mp(shape);
        Scalar current_volume = mp.getVolume();
        vec3<Scalar> dr = this->m_centroids[typid] - mp.getCenterOfMass();
        ShortReal scale = (ShortReal)fast::pow(volume / current_volume, 1.0 / 3.0);
        this->scaleParticleVolume(shape, dr, scale);
        this->m_mc->setParam(typid, shape);
        }

    void scaleParticleVolume(param_type& shape, vec3<Scalar> dr, ShortReal scale)
        {
        Scalar rsq = 0.0;
        for (unsigned int i = 0; i < shape.N; i++)
            {
            shape.x[i] += static_cast<ShortReal>(dr.x);
            shape.x[i] *= scale;
            shape.y[i] += static_cast<ShortReal>(dr.y);
            shape.y[i] *= scale;
            shape.z[i] += static_cast<ShortReal>(dr.z);
            shape.z[i] *= scale;
            vec3<Scalar> vert(shape.x[i], shape.y[i], shape.z[i]);
            rsq = fmax(rsq, dot(vert, vert));
            }
        shape.diameter = ShortReal(2.0 * fast::sqrt(rsq));
        }

    void prepare(uint64_t timestep)
        {
        m_step_size_backup = this->m_step_size;
        }

    void update_shape(uint64_t timestep,
                      const unsigned int& type_id,
                      param_type& shape,
                      hoomd::RandomGenerator& rng,
                      bool managed)
        {
        // perturb the shape.
        for (unsigned int i = 0; i < shape.N; i++)
            {
            if (hoomd::detail::generate_canonical<double>(rng) < this->m_move_probability)
                {
                vec3<Scalar> vert(shape.x[i], shape.y[i], shape.z[i]);
                move_translate(vert, rng, this->m_step_size[type_id], 3);
                shape.x[i] = static_cast<ShortReal>(vert.x);
                shape.y[i] = static_cast<ShortReal>(vert.y);
                shape.z[i] = static_cast<ShortReal>(vert.z);
                }
            }
        detail::MassProperties<ShapeConvexPolyhedron> mp(shape);
        Scalar volume = mp.getVolume();
        vec3<Scalar> dr = this->m_centroids[type_id] - mp.getCenterOfMass();
        ShortReal scale
            = static_cast<ShortReal>(fast::pow(this->m_volume[type_id] / volume, 1.0 / 3.0));
        scaleParticleVolume(shape, dr, scale);
        this->m_step_size[type_id] *= scale;
        }

    void retreat(uint64_t timestep, unsigned int type)
        {
        this->m_step_size[type] = m_step_size_backup[type];
        }

    private:
    std::vector<Scalar> m_step_size_backup;
    }; // end class ConvexPolyhedronVertexShapeMove

template<class Shape> class ElasticShapeMoveBase : public ShapeMoveBase<Shape>
    {
    public:
    ElasticShapeMoveBase(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<IntegratorHPMCMono<Shape>> mc)
        : ShapeMoveBase<Shape>(sysdef, mc)
        {
        }

    typedef typename Shape::param_type param_type;

    void setStiffness(std::shared_ptr<Variant> stiff)
        {
        m_k = stiff;
        }

    std::shared_ptr<Variant> getStiffness() const
        {
        return m_k;
        }

    pybind11::dict getReferenceShape(std::string typ)
        {
        unsigned int typid = this->getValidateType(typ);
        return m_reference_shapes[typid].asDict();
        }

    virtual void setReferenceShape(std::string typ, pybind11::dict v) = 0;

    protected:
    std::shared_ptr<Variant> m_k; // shape move stiffness
    // shape to reference shape move against
    std::vector<param_type, hoomd::detail::managed_allocator<param_type>> m_reference_shapes;
    };

template<class Shape> class ElasticShapeMove : public ElasticShapeMoveBase<Shape>
    {
    };

template<>
class ElasticShapeMove<ShapeConvexPolyhedron> : public ElasticShapeMoveBase<ShapeConvexPolyhedron>
    {
    public:
    typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixSD;
    typedef typename Eigen::Matrix<Scalar, 3, 3> Matrix3S;

    ElasticShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>> mc)
        : ElasticShapeMoveBase<ShapeConvexPolyhedron>(sysdef, mc)
        {
        m_reference_shapes.resize(this->m_ntypes);
        m_F.resize(this->m_ntypes, Matrix3S::Identity());
        m_F_last.resize(this->m_ntypes, Matrix3S::Identity());
        }

    void prepare(uint64_t timestep)
        {
        m_F_last = m_F;
        }

    //! construct is called at the beginning of every update()
    void update_shape(uint64_t timestep,
                      const unsigned int& type_id,
                      param_type& param,
                      hoomd::RandomGenerator& rng,
                      bool managed)
        {
        Matrix3S F_curr;
        // perform a scaling move
        if (hoomd::detail::generate_canonical<double>(rng) < this->m_move_probability)
            {
            generateExtentional(F_curr, rng, this->m_step_size[type_id] + 1.0);
            }
        else // perform a rotation-scale-rotation move
            {
            quat<Scalar> q(1.0, vec3<Scalar>(0.0, 0.0, 0.0));
            move_rotate<3>(q, rng, 0.5);
            Matrix3S rot, rot_inv, scale;
            Eigen::Quaternion<double> eq(q.s, q.v.x, q.v.y, q.v.z);
            rot = eq.toRotationMatrix();
            rot_inv = rot.transpose();
            generateExtentional(scale, rng, this->m_step_size[type_id] + 1.0);
            F_curr = rot * scale * rot_inv;
            }
        m_F[type_id] = F_curr * m_F[type_id];
        auto transform = F_curr.cast<ShortReal>();
        Scalar dsq = 0.0;
        for (unsigned int i = 0; i < param.N; i++)
            {
            vec3<ShortReal> vert(param.x[i], param.y[i], param.z[i]);
            param.x[i]
                = transform(0, 0) * vert.x + transform(0, 1) * vert.y + transform(0, 2) * vert.z;
            param.y[i]
                = transform(1, 0) * vert.x + transform(1, 1) * vert.y + transform(1, 2) * vert.z;
            param.z[i]
                = transform(2, 0) * vert.x + transform(2, 1) * vert.y + transform(2, 2) * vert.z;
            vert = vec3<Scalar>(param.x[i], param.y[i], param.z[i]);
            dsq = fmax(dsq, dot(vert, vert));
            }
        param.diameter = ShortReal(2.0 * fast::sqrt(dsq));
        }

    Matrix3S getEps(unsigned int type_id)
        {
        return 0.5 * (m_F[type_id].transpose() + m_F[type_id]) - Matrix3S::Identity();
        }

    Matrix3S getEpsLast(unsigned int type_id)
        {
        return 0.5 * (m_F_last[type_id].transpose() + m_F_last[type_id]) - Matrix3S::Identity();
        }

    //! retreat whenever the proposed move is rejected.
    void retreat(uint64_t timestep, unsigned int type)
        {
        // we can swap because m_F_last will be reset on the next prepare
        m_F[type] = m_F_last[type];
        }

    void setReferenceShape(std::string typ, pybind11::dict v)
        {
        unsigned int typid = this->getValidateType(typ);
        param_type shape = param_type(v, false);
        auto current_shape = this->m_mc->getParams()[typid];
        if (current_shape.N != shape.N)
            {
            throw std::runtime_error(
                "Reference and integrator shapes must have the name number of vertices.");
            }

        m_reference_shapes[typid] = shape;

        // put vertices into matrix form and leverage Eigein's linear algebra
        // tools to solve Vprime = F * Vref for F where:
        //   Vref: vertices of the reference (undeformed) shape (3, N)
        //   Vprime: vertices of the current (deformed) shape (3, N)
        //   F: deformation gradient tensor (3,3)
        MatrixSD Vref(3, shape.N), Vprime(3, shape.N);
        for (unsigned int i = 0; i < shape.N; i++)
            {
            Vref(0, i) = m_reference_shapes[typid].x[i];
            Vref(1, i) = m_reference_shapes[typid].y[i];
            Vref(2, i) = m_reference_shapes[typid].z[i];
            Vprime(0, i) = current_shape.x[i];
            Vprime(1, i) = current_shape.y[i];
            Vprime(2, i) = current_shape.z[i];
            }

        // solve system
        Matrix3S ret = Vref.transpose()
                           .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                           .solve(Vprime.transpose());
        m_F[typid] = ret.transpose();

        // compute and store volume of reference shape
        detail::MassProperties<ShapeConvexPolyhedron> mp(m_reference_shapes[typid]);
        this->m_volume[typid] = mp.getVolume();
        }

    Scalar computeLogBoltmann(uint64_t timestep,
                              const unsigned int& N,
                              const unsigned int type_id,
                              const typename ShapeConvexPolyhedron::param_type& shape_new,
                              const Scalar& inew,
                              const typename ShapeConvexPolyhedron::param_type& shape_old,
                              const Scalar& iold)
        {
        Scalar inertia_term = ShapeMoveBase<ShapeConvexPolyhedron>::computeLogBoltmann(timestep,
                                                                                       N,
                                                                                       type_id,
                                                                                       shape_new,
                                                                                       inew,
                                                                                       shape_old,
                                                                                       iold);
        Scalar stiff = (*m_k)(timestep);
        Matrix3S eps = this->getEps(type_id);
        Matrix3S eps_last = this->getEpsLast(type_id);
        Scalar e_ddot_e = (eps * eps).trace();
        Scalar e_ddot_e_last = (eps_last * eps_last).trace();
        return N * stiff * (e_ddot_e_last - e_ddot_e) * this->m_volume[type_id] + inertia_term;
        }

    Scalar computeEnergy(uint64_t timestep,
                         const unsigned int& N,
                         const unsigned int type_id,
                         const param_type& shape,
                         const Scalar& inertia)
        {
        Scalar stiff = (*m_k)(timestep);
        Matrix3S eps = this->getEps(type_id);
        Scalar e_ddot_e = (eps * eps).trace();
        return static_cast<Scalar>(N) * stiff * e_ddot_e * this->m_volume[type_id];
        }

    protected:
    std::vector<Matrix3S> m_F_last; // matrix representing shape deformation at the last step
    std::vector<Matrix3S> m_F;      // matrix representing shape deformation at the current step

    private:
    // These are ElasticShapeMove specific helper functions to randomly
    // sample point on the XYZ=1 surface from a uniform distribution

    //! Check if a point (x,y) lies in the projection of xyz=1 surface
    //! on the xy plane
    inline bool inInSurfaceProjection(Scalar x, Scalar y, Scalar alpha)
        {
        if (x < Scalar(1.0) && y > Scalar(1.0) / (alpha * x))
            return true;
        else if (x >= Scalar(1.0) && y < alpha / x)
            return true;
        else
            return false;
        }

    //! Sample points on the projection of xyz=1
    inline void
    sampleOnSurfaceProjection(Scalar& x, Scalar& y, hoomd::RandomGenerator& rng, Scalar alpha)
        {
        hoomd::UniformDistribution<Scalar> uniform(Scalar(1) / alpha, alpha);
        do
            {
            x = uniform(rng);
            y = uniform(rng);
            } while (!inInSurfaceProjection(x, y, alpha));
        }

    //! Sample points on the projection of xyz=1 surface
    inline void sampleOnSurface(Scalar& x, Scalar& y, hoomd::RandomGenerator& rng, Scalar alpha)
        {
        Scalar sigma_max = 0.0, sigma = 0.0, U = 0.0;
        Scalar alpha2 = alpha * alpha;
        Scalar alpha4 = alpha2 * alpha2;
        sigma_max = fast::sqrt(alpha4 + alpha2 + 1);
        do
            {
            sampleOnSurfaceProjection(x, y, rng, alpha);
            sigma
                = fast::sqrt((1.0 / (x * x * x * x * y * y)) + (1.0 / (x * x * y * y * y * y)) + 1);
            U = hoomd::detail::generate_canonical<Scalar>(rng);
            } while (U > sigma / sigma_max);
        }

    //! Generate an volume conserving extentional deformation matrix
    inline void generateExtentional(Matrix3S& S, hoomd::RandomGenerator& rng, Scalar alpha)
        {
        Scalar x = 0.0, y = 0.0, z = 0.0;
        sampleOnSurface(x, y, rng, alpha);
        z = Scalar(1.0) / x / y;
        S << x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, z;
        }
    };

template<> class ElasticShapeMove<ShapeEllipsoid> : public ElasticShapeMoveBase<ShapeEllipsoid>
    {
    public:
    typedef typename ShapeEllipsoid::param_type param_type;

    ElasticShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<IntegratorHPMCMono<ShapeEllipsoid>> mc)
        : ElasticShapeMoveBase<ShapeEllipsoid>(sysdef, mc)
        {
        m_reference_shapes.resize(this->m_ntypes);
        }

    void setReferenceShape(std::string typ, pybind11::dict v)
        {
        unsigned int typid = this->getValidateType(typ);
        m_reference_shapes[typid] = param_type(v, false);
        // compute and store volume of reference shape
        detail::MassProperties<ShapeEllipsoid> mp(m_reference_shapes[typid]);
        this->m_volume[typid] = mp.getVolume();
        }

    void update_shape(uint64_t timestep,
                      const unsigned int& type_id,
                      param_type& param,
                      hoomd::RandomGenerator& rng,
                      bool managed)
        {
        Scalar lnx = log(param.x / param.y);
        Scalar stepsize = this->m_step_size[type_id];
        Scalar dlnx = hoomd::UniformDistribution<Scalar>(-stepsize, stepsize)(rng);
        Scalar x = fast::exp(lnx + dlnx);
        param.x = static_cast<ShortReal>(x) * param.y;
        param.y = param.y;
        param.z = param.z;
        detail::MassProperties<ShapeEllipsoid> mp(param);
        Scalar volume = mp.getVolume();
        ShortReal scale
            = static_cast<ShortReal>(fast::pow(this->m_volume[type_id] / volume, 1.0 / 3.0));
        param.x *= scale;
        param.y *= scale;
        param.z *= scale;
        }

    void prepare(uint64_t timestep) { }

    void retreat(uint64_t timestep, unsigned int type) { }

    Scalar computeLogBoltmann(uint64_t timestep,
                              const unsigned int& N,
                              const unsigned int type_id,
                              const param_type& shape_new,
                              const Scalar& inew,
                              const param_type& shape_old,
                              const Scalar& iold)
        {
        Scalar inertia_term = ShapeMoveBase<ShapeEllipsoid>::computeLogBoltmann(timestep,
                                                                                N,
                                                                                type_id,
                                                                                shape_new,
                                                                                inew,
                                                                                shape_old,
                                                                                iold);
        Scalar old_energy = computeEnergy(timestep, N, type_id, shape_old, iold);
        Scalar new_energy = computeEnergy(timestep, N, type_id, shape_new, inew);
        return old_energy - new_energy + inertia_term;
        }

    Scalar computeEnergy(uint64_t timestep,
                         const unsigned int& N,
                         const unsigned int type_id,
                         const param_type& shape,
                         const Scalar& inertia)
        {
        Scalar stiff = (*m_k)(timestep);
        Scalar logx = log(shape.x / shape.y);
        return static_cast<Scalar>(N) * stiff * logx * logx;
        }

    private:
    };

namespace detail
    {

template<class Shape> void export_ShapeMoveBase(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ShapeMoveBase<Shape>, std::shared_ptr<ShapeMoveBase<Shape>>>(m, name.c_str())
        .def("getVolume", &ShapeMoveBase<Shape>::getVolume)
        .def("setVolume", &ShapeMoveBase<Shape>::setVolume)
        .def("getStepSize", &ShapeMoveBase<Shape>::getStepSize)
        .def("setStepSize", &ShapeMoveBase<Shape>::setStepSize);
    }

template<class Shape> void export_PythonShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PythonShapeMove<Shape>,
                     ShapeMoveBase<Shape>,
                     std::shared_ptr<PythonShapeMove<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>>())
        .def_property("param_move_probability",
                      &PythonShapeMove<Shape>::getMoveProbability,
                      &PythonShapeMove<Shape>::setMoveProbability)
        .def("getParams", &PythonShapeMove<Shape>::getParams)
        .def("setParams", &PythonShapeMove<Shape>::setParams)
        .def_property("callback",
                      &PythonShapeMove<Shape>::getCallback,
                      &PythonShapeMove<Shape>::setCallback);
    }

inline void export_ConvexPolyhedronVertexShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ConvexPolyhedronVertexShapeMove,
                     ShapeMoveBase<ShapeConvexPolyhedron>,
                     std::shared_ptr<ConvexPolyhedronVertexShapeMove>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>>>())
        .def_property("vertex_move_probability",
                      &ConvexPolyhedronVertexShapeMove::getMoveProbability,
                      &ConvexPolyhedronVertexShapeMove::setMoveProbability)
        .def("getVolume", &ConvexPolyhedronVertexShapeMove::getVolume)
        .def("setVolume", &ConvexPolyhedronVertexShapeMove::setVolume);
    }

template<class Shape>
inline void export_ElasticShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ElasticShapeMove<Shape>,
                     ShapeMoveBase<Shape>,
                     std::shared_ptr<ElasticShapeMove<Shape>>>(m, name.c_str())
        .def_property("normal_shear_ratio",
                      &ElasticShapeMove<Shape>::getMoveProbability,
                      &ElasticShapeMove<Shape>::setMoveProbability)
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>>())
        .def_property("stiffness",
                      &ElasticShapeMove<Shape>::getStiffness,
                      &ElasticShapeMove<Shape>::setStiffness)
        .def("setReferenceShape", &ElasticShapeMove<Shape>::setReferenceShape)
        .def("getReferenceShape", &ElasticShapeMove<Shape>::getReferenceShape);
    }

    } // namespace detail
    } // namespace hpmc
    } // namespace hoomd

#endif
