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
    ShapeMoveBase(std::shared_ptr<SystemDefinition> sysdef)
        : m_det_inertia_tensor(0), m_sysdef(sysdef)
        {
        this->m_ntypes = this->m_sysdef->getParticleData()->getNTypes();
        }

    ShapeMoveBase(const ShapeMoveBase& src)
        : m_det_inertia_tensor(src.getDeterminantInertiaTensor())
        {
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
    Scalar m_det_inertia_tensor;     // determinant of the moment of inertia tensor of the shape
    Scalar m_isoperimetric_quotient; // isoperimetric quotient of the shape
    std::shared_ptr<SystemDefinition> m_sysdef;
    unsigned m_ntypes;
    }; // end class ShapeMoveBase

template<typename Shape> class PythonShapeMove : public ShapeMoveBase<Shape>
    {
    public:
    PythonShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                    pybind11::object python_function,
                    pybind11::dict params,
                    Scalar mixratio)
        : ShapeMoveBase<Shape>(sysdef), m_num_params(0), m_python_callback(python_function)
        {
        m_param_move_probability = fmin(mixratio, 1.0);
        this->m_det_inertia_tensor = 1.0;
        std::vector<std::vector<Scalar>> params_vector(this->m_ntypes);
        for (auto name_and_params : params)
            {
            std::string type_name = pybind11::cast<std::string>(name_and_params.first);
            std::vector<Scalar> type_params
                = pybind11::cast<std::vector<Scalar>>(name_and_params.second);
            unsigned int type_i = this->m_sysdef->getParticleData()->getTypeByName(type_name);
            params_vector[type_i] = type_params;
            }
        m_params = params_vector;
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
            hoomd::UniformDistribution<Scalar> uniform(
                fmax(-stepsize, -(m_params[type_id][i])),
                fmin(stepsize, (1.0 - m_params[type_id][i])));
            Scalar x = (hoomd::detail::generate_canonical<double>(rng) < m_param_move_probability)
                           ? uniform(rng)
                           : 0.0;
            m_params[type_id][i] += x;
            }
        pybind11::object shape_data = m_python_callback(m_params[type_id]);
        shape = pybind11::cast<typename Shape::param_type>(shape_data);
        detail::MassProperties<Shape> mp(shape);
        this->m_det_inertia_tensor = mp.getDetInertiaTensor();
        }

    void retreat(uint64_t timestep)
        {
        // move has been rejected.
        std::swap(m_params, m_params_backup);
        }

    Scalar getParam(unsigned int k)
        {
        unsigned int n = 0;
        for (unsigned int i = 0; i < m_params.size(); i++)
            {
            unsigned int next = n + m_params[i].size();
            if (k < next)
                return m_params[i][k - n];
            n = next;
            }
        throw std::out_of_range("Error: Could not get parameter, index out of range.\n");
        return Scalar(0.0);
        }

    unsigned int getNumParam()
        {
        if (m_num_params > 0)
            return m_num_params;
        m_num_params = 0;
        for (unsigned int i = 0; i < m_params.size(); i++)
            m_num_params += m_params[i].size();
        return m_num_params;
        }

    static std::string getParamName(unsigned int i)
        {
        std::stringstream ss;
        std::string snum;
        ss << i;
        ss >> snum;
        return "shape_param-" + snum;
        }

    pybind11::dict getParams()
        {
        pybind11::dict params;
        for (unsigned int i = 0; i < m_params.size(); i++)
            {
            pybind11::str type_name = this->m_sysdef->getParticleData()->getNameByType(i);
            params[type_name] = m_params[i];
            }
        return params;
        }

    void setParams(pybind11::dict params)
        {
        std::vector<std::vector<Scalar>> params_vector(m_params.size());
        for (auto name_and_params : params)
            {
            std::string type_name = pybind11::cast<std::string>(name_and_params.first);
            std::vector<Scalar> type_params
                = pybind11::cast<std::vector<Scalar>>(name_and_params.second);
            unsigned int type_i = this->m_sysdef->getParticleData()->getTypeByName(type_name);
            params_vector[type_i] = type_params;
            }
        m_params = params_vector;
        }

    Scalar getParamMoveProbability()
        {
        return m_param_move_probability;
        }

    void setParamMoveProbability(Scalar select_ratio)
        {
        m_param_move_probability = fmin(select_ratio, 1.0);
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
    Scalar m_param_move_probability;
    unsigned int m_num_params; // cache the number of parameters.
    Scalar m_scale; // the scale needed to keep the particle at constant volume. internal use
    std::vector<std::vector<Scalar>> m_params_backup; // all params are from 0,1
    std::vector<std::vector<Scalar>> m_params;        // all params are from 0,1
    pybind11::object m_python_callback; // callback that takes m_params as an argiment and returns
                                        // (shape, det(I))
    };

template<typename Shape> class ConstantShapeMove : public ShapeMoveBase<Shape>
    {
    public:
    ConstantShapeMove(std::shared_ptr<SystemDefinition> sysdef, pybind11::dict shape_params)
        : ShapeMoveBase<Shape>(sysdef), m_shape_moves({})
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
                                    Scalar vertex_move_prob,
                                    Scalar volume)
        : ShapeMoveBase<ShapeConvexPolyhedron>(sysdef), m_volume(volume)
        {
        this->m_det_inertia_tensor = 1.0;
        m_scale = 1.0;
        m_calculated.resize(this->m_ntypes, false);
        m_centroids.resize(this->m_ntypes, vec3<Scalar>(0, 0, 0));
        m_vertex_move_probability = fmin(vertex_move_prob, 1.0);
        }

    Scalar getVertexMoveProbability()
        {
        return m_vertex_move_probability;
        }

    void setVertexMoveProbability(Scalar vertex_move_prob)
        {
        m_vertex_move_probability = fmin(vertex_move_prob, 1.0);
        }

    Scalar getVolume()
        {
        return m_volume;
        }

    void setVolume(Scalar volume)
        {
        m_volume = volume;
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
            m_centroids[type_id] = mp.getCenterOfMass();
            m_calculated[type_id] = true;
            }
        // mix the shape.
        for (unsigned int i = 0; i < shape.N; i++)
            {
            if (hoomd::detail::generate_canonical<double>(rng) < m_vertex_move_probability)
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
        vec3<Scalar> dr = m_centroids[type_id] - mp.getCenterOfMass();
        m_scale = (OverlapReal)fast::pow(m_volume / volume, 1.0 / 3.0);
        Scalar rsq = 0.0;
        std::vector<vec3<Scalar>> points(shape.N);
        for (unsigned int i = 0; i < shape.N; i++)
            {
            shape.x[i] += (OverlapReal)dr.x;
            shape.x[i] *= m_scale;
            shape.y[i] += (OverlapReal)dr.y;
            shape.y[i] *= m_scale;
            shape.z[i] += (OverlapReal)dr.z;
            shape.z[i] *= m_scale;
            vec3<Scalar> vert(shape.x[i], shape.y[i], shape.z[i]);
            rsq = fmax(rsq, dot(vert, vert));
            points[i] = vert;
            }
        mp.updateParam(shape, true);
        this->m_det_inertia_tensor = mp.getDetInertiaTensor();
        m_isoperimetric_quotient = mp.getIsoperimetricQuotient();
        shape.diameter = OverlapReal(2.0 * fast::sqrt(rsq));
        stepsize *= m_scale; // only need to scale if the parameters are not normalized
        }

    void retreat(uint64_t timestep)
        {
        // move has been rejected.
        }

    private:
    Scalar m_vertex_move_probability; // probability of a vertex being selected for a move
    OverlapReal m_scale; // factor to scale the shape by to achieve desired constant volume
    Scalar m_volume;     // desired constant volume of each shape
    std::vector<vec3<Scalar>> m_centroids; // centroid of each type of shape
    std::vector<bool> m_calculated;        // whether or not mass properties has been calculated
    };                                     // end class ConvexPolyhedronVertexShapeMove

template<class Shape> class ElasticShapeMove : public ShapeMoveBase<Shape>
    {
    public:
    typedef typename Shape::param_type param_type;
    typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixSD;
    typedef typename Eigen::Matrix<Scalar, 3, 3> Matrix3S;

    ElasticShapeMove(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<IntegratorHPMCMono<Shape>> mc,
                     Scalar shear_scale_ratio,
                     std::shared_ptr<Variant> k)
        : ShapeMoveBase<Shape>(sysdef), m_mc(mc), m_k(k)
        {
        /* TODO: Since the deformation tensor, F, is no longer stored in the GSD,
                 it has to be computed here. F takes the reference shape and
                 transforms it into a deformed shape by V' = F * Vref. The conponents
                 of F are sampled in such a way that particle volume is always conserved.
        */
        m_mass_props.resize(this->m_ntypes);
        m_reference_shapes.resize(this->m_ntypes);
        m_shear_scale_ratio = fmin(shear_scale_ratio, 1.0);
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
        if (hoomd::detail::generate_canonical<double>(rng) < m_shear_scale_ratio)
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
        return m_shear_scale_ratio;
        }

    void setShearScaleRatio(Scalar scale_shear_ratio)
        {
        m_shear_scale_ratio = fmin(scale_shear_ratio, 1.0);
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
        auto current_shape = m_mc->getParams()[typid];
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
        //   F: deformation gradient tendor (3,3)
        // Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Vref(3, shape.N), Vprime(3,
        // shape.N);
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
        // TODO: one of the following:
        //       1) warn or error out if volume of provided shape doesnt match That
        //          of the integrator definition
        //       2) update m_volume
        //       3) rescale the provided shape with m_volume
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
        if (newdivold < 0.0)
            {
            newdivold = -1.0 * newdivold;
            } // MOI may be negative depending on order of vertices
        Scalar inertia_term = (Scalar(N) / Scalar(2.0)) * log(newdivold);
        Scalar stiff = (*m_k)(timestep);
        Matrix3S eps = this->getEps(type_id);
        Matrix3S eps_last = this->getEpsLast(type_id);
        Scalar e_ddot_e = (eps * eps.transpose()).trace();
        Scalar e_ddot_e_last = (eps_last * eps_last.transpose()).trace();
        return N * stiff * (e_ddot_e_last - e_ddot_e) * this->m_volume + inertia_term;
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
        return N * stiff * e_ddot_e * this->m_volume;
        }

    protected:
    Scalar m_shear_scale_ratio;
    ; // probability of performing a scaling move vs a
      // rotation-scale-rotation move
    std::vector<detail::MassProperties<Shape>> m_mass_props; // mass properties of the shape
    std::vector<Matrix3S> m_F_last; // matrix representing shape deformation at the last step
    std::vector<Matrix3S> m_F;      // matrix representing shape deformation at the current step
    Scalar m_volume;                // volume of shape
    std::vector<param_type, hoomd::detail::managed_allocator<param_type>>
        m_reference_shapes; // shape to reference shape move against
    std::shared_ptr<IntegratorHPMCMono<Shape>> m_mc;
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
                     std::shared_ptr<IntegratorHPMCMono<ShapeEllipsoid>> mc,
                     Scalar move_ratio,
                     std::shared_ptr<Variant> k)
        : ShapeMoveBase<ShapeEllipsoid>(sysdef), m_mc(mc), m_k(k)
        {
        m_mass_props.resize(this->m_ntypes);
        // // typename ShapeEllipsoid::param_type shape(shape_params);
        // m_reference_shapes = shape;
        // detail::MassProperties<ShapeEllipsoid> mp(m_reference_shapes);
        // m_volume = mp.getVolume();
        }

    Scalar getShearScaleRatio()
        {
        return m_shear_scale_ratio;
        }

    void setShearScaleRatio(Scalar scale_shear_ratio)
        {
        m_shear_scale_ratio = fmin(scale_shear_ratio, 1.0);
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
        unsigned int id = this->m_sysdef->getParticleData()->getTypeByName(typ);
        param_type shape = param_type(v, false);
        m_reference_shapes[id] = shape;
        // TODO: one of the following:
        //       1) warn or error out if volume of provided shape doesnt match That
        //          of the integrator definition
        //       2) update m_volume
        //       3) rescale the provided shape with m_volume
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
        m_mass_props[type_id].updateParam(param);
        Scalar volume = m_mass_props[type_id].getVolume();
        Scalar vol_factor = detail::MassProperties<ShapeEllipsoid>::m_vol_factor;
        Scalar b = fast::pow(volume / vol_factor / x, 1.0 / 3.0);
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
    Scalar m_shear_scale_ratio;
    std::vector<detail::MassProperties<ShapeEllipsoid>>
        m_mass_props; // mass properties of the shape
    Scalar m_volume;  // volume of shape
    std::vector<param_type, hoomd::detail::managed_allocator<param_type>>
        m_reference_shapes; // shape to reference shape move against
    std::shared_ptr<IntegratorHPMCMono<ShapeEllipsoid>> m_mc;
    std::shared_ptr<Variant> m_k; // shape move stiffness
    };

namespace detail
    {

template<class Shape> void export_ShapeMoveBase(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ShapeMoveBase<Shape>, std::shared_ptr<ShapeMoveBase<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

template<class Shape> void export_PythonShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PythonShapeMove<Shape>,
                     ShapeMoveBase<Shape>,
                     std::shared_ptr<PythonShapeMove<Shape>>>(m, name.c_str())
        .def(
            pybind11::
                init<std::shared_ptr<SystemDefinition>, pybind11::object, pybind11::dict, Scalar>())
        .def_property("params",
                      &PythonShapeMove<Shape>::getParams,
                      &PythonShapeMove<Shape>::setParams)
        .def_property("param_move_probability",
                      &PythonShapeMove<Shape>::getParamMoveProbability,
                      &PythonShapeMove<Shape>::setParamMoveProbability)
        .def_property("callback",
                      &PythonShapeMove<Shape>::getCallback,
                      &PythonShapeMove<Shape>::setCallback);
    }

// template<class Shape>
inline void export_ConvexPolyhedronVertexShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ConvexPolyhedronVertexShapeMove,
                     ShapeMoveBase<ShapeConvexPolyhedron>,
                     std::shared_ptr<ConvexPolyhedronVertexShapeMove>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar, Scalar>())
        .def_property("volume",
                      &ConvexPolyhedronVertexShapeMove::getVolume,
                      &ConvexPolyhedronVertexShapeMove::setVolume)
        .def_property("vertex_move_probability",
                      &ConvexPolyhedronVertexShapeMove::getVertexMoveProbability,
                      &ConvexPolyhedronVertexShapeMove::setVertexMoveProbability);
    }

template<class Shape>
inline void export_ConstantShapeMove(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ConstantShapeMove<Shape>,
                     ShapeMoveBase<Shape>,
                     std::shared_ptr<ConstantShapeMove<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, pybind11::dict>())
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
                            std::shared_ptr<IntegratorHPMCMono<Shape>>,
                            Scalar,
                            std::shared_ptr<Variant>>())
        .def_property("shear_scale_ratio",
                      &ElasticShapeMove<Shape>::getShearScaleRatio,
                      &ElasticShapeMove<Shape>::setShearScaleRatio)
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
