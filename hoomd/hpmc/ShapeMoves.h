// Copyright (c) 2009-2022 The Regents of the University of Michigan.
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
        : m_mc(mc), m_det_inertia_tensor(0), m_sysdef(sysdef)
        {
        m_ntypes = this->m_sysdef->getParticleData()->getNTypes();
        m_volume.resize(m_ntypes, 0);
        }

    virtual ~ShapeMoveBase() {};

    //! prepare is called at the beginning of every update()
    virtual void prepare(uint64_t timestep)
        {
        throw std::runtime_error("Shape move function not implemented.");
        }

    //! construct is called for each particle type that will be changed in update()
    virtual void update_shape(uint64_t,
                              Scalar& stepsize,
                              const unsigned int&,
                              typename Shape::param_type&,
                              hoomd::RandomGenerator&)
        {
        throw std::runtime_error("Shape move function not implemented.");
        }

    //! retreat whenever the proposed move is rejected.
    virtual void retreat(uint64_t timestep)
        {
        throw std::runtime_error("Shape move function not implemented.");
        }

    Scalar getDetInertiaTensor() const
        {
        return m_det_inertia_tensor;
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

    void scaleParticleVolume(typename Shape::param_type& shape, vec3<Scalar> dr, OverlapReal scale)
        {
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

    // Get the isoperimetric quotient of the shape
    Scalar getIsoperimetricQuotient() const
        {
        return m_isoperimetric_quotient;
        }

    virtual Scalar operator()(uint64_t timestep,
                              const unsigned int& N,
                              const unsigned int type_id,
                              const typename Shape::param_type& shape_new,
                              const Scalar& inew,
                              const typename Shape::param_type& shape_old,
                              const Scalar& iold)
        {
        Scalar newdivold = inew / iold;
        if (newdivold < 0.0)
            {
            newdivold = -1.0 * newdivold;
            } // MOI may be negative depending on order of vertices
        return (Scalar(N) / Scalar(2.0)) * log(newdivold);
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
    Scalar m_det_inertia_tensor;     // determinant of the moment of inertia tensor of the shape
    Scalar m_isoperimetric_quotient; // isoperimetric quotient of the shape
    std::shared_ptr<SystemDefinition> m_sysdef;
    unsigned m_ntypes;
    std::vector<Scalar> m_volume;
    Scalar m_move_probability;
    std::vector<vec3<Scalar>> m_centroids;

    }; // end class ShapeMoveBase

template<typename Shape> class PythonShapeMove : public ShapeMoveBase<Shape>
    {
    public:
    PythonShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<IntegratorHPMCMono<Shape>> mc)
        : ShapeMoveBase<Shape>(sysdef, mc), m_num_params(0)
        {
        m_params.resize(this->m_ntypes);
        m_params_backup.resize(this->m_ntypes);
        this->m_det_inertia_tensor = 1.0;
        }

    void prepare(uint64_t timestep)
        {
        m_params_backup = m_params;
        }

    void update_shape(uint64_t timestep,
                      Scalar& stepsize,
                      const unsigned int& type_id,
                      typename Shape::param_type& shape,
                      hoomd::RandomGenerator& rng)
        {
        for (unsigned int i = 0; i < m_params[type_id].size(); i++)
            {
            Scalar a = fmax(-stepsize, -(m_params[type_id][i]));
            Scalar b = fmin(stepsize, (1.0 - m_params[type_id][i]));
            hoomd::UniformDistribution<Scalar> uniform(a, b);
            Scalar r = hoomd::detail::generate_canonical<double>(rng);
            Scalar x = (r < this->m_move_probability) ? uniform(rng) : 0.0;
            m_params[type_id][i] += x;
            }
        pybind11::object d = m_python_callback(type_id, m_params[type_id]);
        pybind11::dict shape_dict = pybind11::cast<pybind11::dict>(d);
        shape = typename Shape::param_type(shape_dict);
        detail::MassProperties<Shape> mp(shape);
        this->m_det_inertia_tensor = mp.getDetInertiaTensor();
        }

    void retreat(uint64_t timestep)
        {
        // move has been rejected.
        std::swap(m_params, m_params_backup);
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

    pybind11::dict getTypeParams()
        {
        pybind11::dict ret;
        for (unsigned int type_id = 0; type_id < this->m_ntypes; type_id++)
            {
            std::string type_name = this->m_sysdef->getParticleData()->getNameByType(type_id);
            pybind11::list l;
            for (unsigned int i = 0; i < m_params[type_id].size(); i++)
                {
                l.append(m_params[type_id][i]);
                }
            ret[type_name.c_str()] = l;
            }
        return ret;
        }

    Scalar getParamMoveProbability()
        {
        return this->m_move_probability;
        }

    void setParamMoveProbability(Scalar select_ratio)
        {
        this->m_move_probability = fmin(select_ratio, 1.0);
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
    unsigned int m_num_params;                        // cache the number of parameters.
    std::vector<std::vector<Scalar>> m_params_backup; // all params are from 0,1
    std::vector<std::vector<Scalar>> m_params;        // all params are from 0,1
    pybind11::object m_python_callback; // callback that takes m_params as an argiment and returns
    };

template<typename Shape> class ConstantShapeMove : public ShapeMoveBase<Shape>
    {
    public:
    ConstantShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                      pybind11::dict shape_params)
        : ShapeMoveBase<Shape>(sysdef, mc), m_shape_moves({})
        {
        std::vector<pybind11::dict> shape_params_vector(this->m_ntypes);
        for (auto name_and_params : shape_params)
            {
            std::string type_name = pybind11::cast<std::string>(name_and_params.first);
            pybind11::dict type_params = pybind11::cast<pybind11::dict>(name_and_params.second);
            unsigned int type_i = this->m_sysdef->getParticleData()->getTypeByName(type_name);
            shape_params_vector[type_i] = type_params;
            }
        m_shape_params = shape_params_vector;
        for (unsigned int i = 0; i < this->m_ntypes; i++)
            {
            typename Shape::param_type pt(m_shape_params[i]);
            m_shape_moves.push_back(pt);
            }
        if (this->m_ntypes != m_shape_moves.size())
            throw std::runtime_error("Must supply a shape move for each type");
        for (unsigned int i = 0; i < m_shape_moves.size(); i++)
            {
            detail::MassProperties<Shape> mp(m_shape_moves[i]);
            m_determinants.push_back(mp.getDetInertiaTensor());
            }
        }

    void prepare(uint64_t timestep) { }

    void update_shape(uint64_t timestep,
                      Scalar& stepsize,
                      const unsigned int& type_id,
                      typename Shape::param_type& shape,
                      hoomd::RandomGenerator& rng)
        {
        shape = m_shape_moves[type_id];
        this->m_det_inertia_tensor = m_determinants[type_id];
        }

    void retreat(uint64_t timestep)
        {
        // move has been rejected.
        }

    pybind11::dict getShapeParams()
        {
        pybind11::dict shape_params;
        for (unsigned int i = 0; i < m_shape_params.size(); i++)
            {
            pybind11::str type_name = this->m_sysdef->getParticleData()->getNameByType(i);
            shape_params[type_name] = m_shape_params[i];
            }
        return shape_params;
        }

    void setShapeParams(pybind11::dict shape_params)
        {
        std::vector<pybind11::dict> shape_params_vector(m_shape_params.size());
        for (auto name_and_params : shape_params)
            {
            std::string type_name = pybind11::cast<std::string>(name_and_params.first);
            pybind11::dict type_params = pybind11::cast<pybind11::dict>(name_and_params.second);
            unsigned int type_i = this->m_sysdef->getParticleData()->getTypeByName(type_name);
            shape_params_vector[type_i] = type_params;
            typename Shape::param_type pt(type_params);
            m_shape_moves[type_i] = pt;
            }
        m_shape_params = shape_params_vector;
        }

    private:
    std::vector<typename Shape::param_type> m_shape_moves;
    std::vector<Scalar> m_determinants;
    std::vector<pybind11::dict> m_shape_params;
    };

class ConvexPolyhedronVertexShapeMove : public ShapeMoveBase<ShapeConvexPolyhedron>
    {
    public:
    typedef typename ShapeConvexPolyhedron::param_type param_type;

    ConvexPolyhedronVertexShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                                    std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>> mc)
        : ShapeMoveBase<ShapeConvexPolyhedron>(sysdef, mc)
        {
        this->m_det_inertia_tensor = 1.0;
        m_calculated.resize(this->m_ntypes, false);
        this->m_centroids.resize(this->m_ntypes, vec3<Scalar>(0, 0, 0));
        initializeMassProperties();
        }

    void initializeMassProperties()
        {
        auto& mc_params = this->m_mc->getParams();
        for (unsigned int i; i < this->m_ntypes; i++)
            {
            detail::MassProperties<ShapeConvexPolyhedron> mp(mc_params[i]);
            this->m_centroids[i] = mp.getCenterOfMass();
            // possibly add det I here
            m_calculated[i] = true;
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
        OverlapReal scale = (OverlapReal)fast::pow(volume / current_volume, 1.0 / 3.0);
        this->scaleParticleVolume(shape, dr, scale);
        this->m_mc->setParam(typid, shape);
        }

    void scaleParticleVolume(param_type& shape, vec3<Scalar> dr, OverlapReal scale)
        {
        Scalar rsq = 0.0;
        for (unsigned int i = 0; i < shape.N; i++)
            {
            shape.x[i] += (OverlapReal)dr.x;
            shape.x[i] *= scale;
            shape.y[i] += (OverlapReal)dr.y;
            shape.y[i] *= scale;
            shape.z[i] += (OverlapReal)dr.z;
            shape.z[i] *= scale;
            vec3<Scalar> vert(shape.x[i], shape.y[i], shape.z[i]);
            rsq = fmax(rsq, dot(vert, vert));
            }
        shape.diameter = OverlapReal(2.0 * fast::sqrt(rsq));
        }

    Scalar getVertexMoveProbability()
        {
        return this->m_move_probability;
        }

    void setVertexMoveProbability(Scalar vertex_move_prob)
        {
        this->m_move_probability = fmin(vertex_move_prob, 1.0);
        }

    void prepare(uint64_t timestep) { }

    void update_shape(uint64_t timestep,
                      Scalar& stepsize,
                      const unsigned int& type_id,
                      param_type& shape,
                      hoomd::RandomGenerator& rng)
        {
        if (!m_calculated[type_id])
            {
            detail::MassProperties<ShapeConvexPolyhedron> mp(shape);
            this->m_centroids[type_id] = mp.getCenterOfMass();
            m_calculated[type_id] = true;
            }
        // mix the shape.
        for (unsigned int i = 0; i < shape.N; i++)
            {
            if (hoomd::detail::generate_canonical<double>(rng) < this->m_move_probability)
                {
                vec3<Scalar> vert(shape.x[i], shape.y[i], shape.z[i]);
                move_translate(vert, rng, stepsize, 3);
                shape.x[i] = (OverlapReal)vert.x;
                shape.y[i] = (OverlapReal)vert.y;
                shape.z[i] = (OverlapReal)vert.z;
                }
            }
        detail::MassProperties<ShapeConvexPolyhedron> mp(shape);
        Scalar volume = mp.getVolume();
        vec3<Scalar> dr = this->m_centroids[type_id] - mp.getCenterOfMass();
        OverlapReal scale = (OverlapReal)fast::pow(this->m_volume[type_id] / volume, 1.0 / 3.0);
        scaleParticleVolume(shape, dr, scale);
        stepsize *= scale;
        }

    void retreat(uint64_t timestep)
        {
        // move has been rejected.
        }

    private:
    std::vector<bool> m_calculated; // whether or not mass properties has been calculated
    };                              // end class ConvexPolyhedronVertexShapeMove

template<class Shape> class ElasticShapeMove : public ShapeMoveBase<Shape>
    {
    public:
    typedef typename Shape::param_type param_type;
    typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixSD;
    typedef typename Eigen::Matrix<Scalar, 3, 3> Matrix3S;

    ElasticShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<IntegratorHPMCMono<Shape>> mc)
        : ShapeMoveBase<Shape>(sysdef, mc)
        {
        m_mass_props.resize(this->m_ntypes);
        m_reference_shapes.resize(this->m_ntypes);
        m_F.resize(this->m_ntypes, Matrix3S::Identity());
        m_F_last.resize(this->m_ntypes, Matrix3S::Identity());
        this->m_det_inertia_tensor = 1.0;
        }

    void prepare(uint64_t timestep)
        {
        m_F_last = m_F;
        }

    //! construct is called at the beginning of every update()
    void update_shape(uint64_t timestep,
                      Scalar& stepsize,
                      const unsigned int& type_id,
                      param_type& param,
                      hoomd::RandomGenerator& rng)
        {
        Matrix3S transform;
        // perform a scaling move
        if (hoomd::detail::generate_canonical<double>(rng) < this->m_move_probability)
            {
            generateExtentional(transform, rng, stepsize + 1.0);
            }
        else // perform a rotation-scale-rotation move
            {
            quat<Scalar> q(1.0, vec3<Scalar>(0.0, 0.0, 0.0));
            move_rotate<3>(q, rng, 0.5);
            Matrix3S rot, rot_inv, scale;
            Eigen::Quaternion<double> eq(q.s, q.v.x, q.v.y, q.v.z);
            rot = eq.toRotationMatrix();
            rot_inv = rot.transpose();
            generateExtentional(scale, rng, stepsize + 1.0);
            transform = rot * scale * rot_inv;
            }

        m_F[type_id] = transform * m_F[type_id];
        Scalar dsq = 0.0;
        for (unsigned int i = 0; i < param.N; i++)
            {
            vec3<OverlapReal> vert(param.x[i], param.y[i], param.z[i]);
            param.x[i]
                = transform(0, 0) * vert.x + transform(0, 1) * vert.y + transform(0, 2) * vert.z;
            param.y[i]
                = transform(1, 0) * vert.x + transform(1, 1) * vert.y + transform(1, 2) * vert.z;
            param.z[i]
                = transform(2, 0) * vert.x + transform(2, 1) * vert.y + transform(2, 2) * vert.z;
            vert = vec3<Scalar>(param.x[i], param.y[i], param.z[i]);
            dsq = fmax(dsq, dot(vert, vert));
            }
        param.diameter = OverlapReal(2.0 * fast::sqrt(dsq));
        // update allows caching since for some shapes a full compute is not necessary.
        m_mass_props[type_id].updateParam(param, false);
        // update det(I)
        this->m_det_inertia_tensor = m_mass_props[type_id].getDetInertiaTensor();
#ifdef DEBUG
        detail::MassProperties<Shape> mp(param);
        this->m_det_inertia_tensor = mp.getDetInertiaTensor();
        assert(fabs(this->m_det_inertia_tensor - mp.getDetInertiaTensor()) < 1e-5);
#endif
        }

    Matrix3S getEps(unsigned int type_id)
        {
        return 0.5 * ((m_F[type_id].transpose() * m_F[type_id]) - Matrix3S::Identity());
        }

    Matrix3S getEpsLast(unsigned int type_id)
        {
        return 0.5 * ((m_F_last[type_id].transpose() * m_F_last[type_id]) - Matrix3S::Identity());
        }

    //! retreat whenever the proposed move is rejected.
    void retreat(uint64_t timestep)
        {
        // we can swap because m_F_last will be reset on the next prepare
        m_F.swap(m_F_last);
        }

    Scalar getShearScaleRatio()
        {
        return this->m_move_probability;
        }

    void setShearScaleRatio(Scalar scale_shear_ratio)
        {
        this->m_move_probability = fmin(scale_shear_ratio, 1.0);
        }

    void setStiffness(std::shared_ptr<Variant> stiff)
        {
        m_k = stiff;
        }

    std::shared_ptr<Variant> getStiffness() const
        {
        return m_k;
        }

    void setReferenceShape(std::string typ, pybind11::dict v)
        {
        unsigned int typid = this->m_sysdef->getParticleData()->getTypeByName(typ);
        if (typid >= this->m_sysdef->getParticleData()->getNTypes())
            {
            throw std::runtime_error("Invalid particle type.");
            }

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
        detail::MassProperties<Shape> mp(m_reference_shapes[typid]);
        this->m_volume[typid] = mp.getVolume();
        }

    pybind11::dict getReferenceShape(std::string typ)
        {
        unsigned int id = this->m_sysdef->getParticleData()->getTypeByName(typ);
        return m_reference_shapes[id].asDict();
        }

    Scalar operator()(uint64_t timestep,
                      const unsigned int& N,
                      const unsigned int type_id,
                      const typename Shape::param_type& shape_new,
                      const Scalar& inew,
                      const typename Shape::param_type& shape_old,
                      const Scalar& iold)
        {
        Scalar newdivold = inew / iold;
        Scalar inertia_term = (Scalar(N) / Scalar(2.0)) * log(newdivold);
        Scalar stiff = (*m_k)(timestep);
        Matrix3S eps = this->getEps(type_id);
        Matrix3S eps_last = this->getEpsLast(type_id);
        Scalar e_ddot_e = (eps * eps.transpose()).trace();
        Scalar e_ddot_e_last = (eps_last * eps_last.transpose()).trace();
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
        Scalar e_ddot_e = (eps * eps.transpose()).trace();
        return N * stiff * e_ddot_e * this->m_volume[type_id];
        }

    protected:
    std::vector<detail::MassProperties<Shape>> m_mass_props; // mass properties of the shape
    std::vector<Matrix3S> m_F_last; // matrix representing shape deformation at the last step
    std::vector<Matrix3S> m_F;      // matrix representing shape deformation at the current step
    std::vector<param_type, hoomd::detail::managed_allocator<param_type>>
        m_reference_shapes;       // shape to reference shape move against
    std::shared_ptr<Variant> m_k; // shape move stiffness

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

template<> class ElasticShapeMove<ShapeEllipsoid> : public ShapeMoveBase<ShapeEllipsoid>
    {
    public:
    typedef typename ShapeEllipsoid::param_type param_type;

    ElasticShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<IntegratorHPMCMono<ShapeEllipsoid>> mc)
        : ShapeMoveBase<ShapeEllipsoid>(sysdef, mc)
        {
        m_mass_props.resize(this->m_ntypes);
        }

    Scalar getShearScaleRatio()
        {
        return this->m_move_probability;
        }

    void setShearScaleRatio(Scalar scale_shear_ratio)
        {
        this->m_move_probability = fmin(scale_shear_ratio, 1.0);
        }

    void setStiffness(std::shared_ptr<Variant> stiff)
        {
        m_k = stiff;
        }

    std::shared_ptr<Variant> getStiffness() const
        {
        return m_k;
        }

    void setReferenceShape(std::string typ, pybind11::dict v)
        {
        unsigned int typid = this->m_sysdef->getParticleData()->getTypeByName(typ);
        if (typid >= this->m_sysdef->getParticleData()->getNTypes())
            {
            throw std::runtime_error("Invalid particle type.");
            }

        m_reference_shapes[typid] = param_type(v, false);
        ;

        // compute and store volume of reference shape
        detail::MassProperties<ShapeEllipsoid> mp(m_reference_shapes[typid]);
        this->m_volume[typid] = mp.getVolume();
        }

    pybind11::dict getReferenceShape(std::string typ)
        {
        unsigned int id = this->m_sysdef->getParticleData()->getTypeByName(typ);
        return m_reference_shapes[id].asDict();
        }

    void update_shape(uint64_t timestep,
                      Scalar& stepsize,
                      const unsigned int& type_id,
                      param_type& param,
                      hoomd::RandomGenerator& rng)
        {
        Scalar lnx = log(param.x / param.y);
        Scalar dlnx = hoomd::UniformDistribution<Scalar>(-stepsize, stepsize)(rng);
        Scalar x = fast::exp(lnx + dlnx);
        detail::MassProperties<ShapeEllipsoid> mp(param);
        Scalar volume = mp.getVolume();
        Scalar b = fast::pow(this->m_volume[type_id] / volume, 1.0 / 3.0);
        x *= b;
        param.x = (OverlapReal)x;
        param.y = (OverlapReal)b;
        param.z = (OverlapReal)b;
        }

    void prepare(uint64_t timestep) { }

    void retreat(uint64_t timestep) { }

    virtual Scalar operator()(uint64_t timestep,
                              const unsigned int& N,
                              const unsigned int type_id,
                              const param_type& shape_new,
                              const Scalar& inew,
                              const param_type& shape_old,
                              const Scalar& iold)
        {
        Scalar stiff = (*m_k)(timestep);
        Scalar x_new = shape_new.x / shape_new.y;
        Scalar x_old = shape_old.x / shape_old.y;
        return stiff * (log(x_old) * log(x_old) - log(x_new) * log(x_new));
        }

    virtual Scalar computeEnergy(uint64_t timestep,
                                 const unsigned int& N,
                                 const unsigned int type_id,
                                 const param_type& shape,
                                 const Scalar& inertia)
        {
        Scalar stiff = (*m_k)(timestep);
        Scalar logx = log(shape.x / shape.y);
        return N * stiff * logx * logx;
        }

    private:
    std::vector<detail::MassProperties<ShapeEllipsoid>>
        m_mass_props; // mass properties of the shape
    std::vector<param_type, hoomd::detail::managed_allocator<param_type>>
        m_reference_shapes;       // shape to reference shape move against
    std::shared_ptr<Variant> m_k; // shape move stiffness
    };

namespace detail
    {

template<class Shape> void export_ShapeMoveBase(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ShapeMoveBase<Shape>, std::shared_ptr<ShapeMoveBase<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>>())
        .def_property("move_probability",
                      &ShapeMoveBase<Shape>::getMoveProbability,
                      &ShapeMoveBase<Shape>::setMoveProbability)
        .def("getVolume", &ShapeMoveBase<Shape>::getVolume)
        .def("setVolume", &ShapeMoveBase<Shape>::setVolume);
    }

template<class Shape> void export_PythonShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PythonShapeMove<Shape>,
                     ShapeMoveBase<Shape>,
                     std::shared_ptr<PythonShapeMove<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>>())
        .def("getParams", &PythonShapeMove<Shape>::getParams)
        .def("setParams", &PythonShapeMove<Shape>::setParams)
        .def_property("callback",
                      &PythonShapeMove<Shape>::getCallback,
                      &PythonShapeMove<Shape>::setCallback)
        .def("getTypeParams", &PythonShapeMove<Shape>::getTypeParams);
    }

inline void export_ConvexPolyhedronVertexShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ConvexPolyhedronVertexShapeMove,
                     ShapeMoveBase<ShapeConvexPolyhedron>,
                     std::shared_ptr<ConvexPolyhedronVertexShapeMove>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<ShapeConvexPolyhedron>>>())
        .def("getVolume", &ConvexPolyhedronVertexShapeMove::getVolume)
        .def("setVolume", &ConvexPolyhedronVertexShapeMove::setVolume);
    }

template<class Shape>
inline void export_ConstantShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ConstantShapeMove<Shape>,
                     ShapeMoveBase<Shape>,
                     std::shared_ptr<ConstantShapeMove<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            pybind11::dict>())
        .def_property("shape_params",
                      &ConstantShapeMove<Shape>::getShapeParams,
                      &ConstantShapeMove<Shape>::setShapeParams);
    }

template<class Shape>
inline void export_ElasticShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ElasticShapeMove<Shape>,
                     ShapeMoveBase<Shape>,
                     std::shared_ptr<ElasticShapeMove<Shape>>>(m, name.c_str())
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
