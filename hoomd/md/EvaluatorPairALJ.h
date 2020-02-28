// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: vramasub

#ifndef __EVALUATOR_PAIR_ALJ_H__
#define __EVALUATOR_PAIR_ALJ_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/VectorMath.h"
#include "hoomd/ManagedArray.h"
#include "GJK_SV.h"

#ifndef NVCC
#include "hoomd/ExecutionConfiguration.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

/*! \file EvaluatorPairALJ.h
    \brief Defines a an evaluator class for the anisotropic LJ table potential.
*/

// need to declare these class methods with __host__ __device__ qualifiers when building in nvcc
//! HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE

#if defined (__SSE__)
#include <immintrin.h>
#endif
#endif

//! Shape parameters for the ALJ potential.
struct alj_shape_params
    {
    HOSTDEVICE alj_shape_params()
        {}

    #ifndef NVCC

    //! Shape constructor
    alj_shape_params(pybind11::list vertices, pybind11::list rr, bool use_device) : has_rounding(false)
        {
        //! Construct table for particle i
        unsigned int N = len(vertices);
        verts = ManagedArray<vec3<Scalar> >(N, use_device);
        for (unsigned int i = 0; i < N; ++i)
            {
            pybind11::list vertices_tmp = pybind11::cast<pybind11::list>(vertices[i]);
            verts[i] = vec3<Scalar>(pybind11::cast<Scalar>(vertices_tmp[0]), pybind11::cast<Scalar>(vertices_tmp[1]), pybind11::cast<Scalar>(vertices_tmp[2]));

            }

        rounding_radii.x = pybind11::cast<Scalar>(rr[0]);
        rounding_radii.y = pybind11::cast<Scalar>(rr[1]);
        rounding_radii.z = pybind11::cast<Scalar>(rr[2]);
        if (rounding_radii.x > 0 || rounding_radii.y > 0 || rounding_radii.z > 0)
            {
            has_rounding = true;
            }
        }

    #endif

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const
        {
        verts.load_shared(ptr, available_bytes);
        }

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const
        {
        verts.attach_to_stream(stream);
        }
    #endif

    //! Shape parameters
    ManagedArray<vec3<Scalar> > verts;       //! Shape vertices.
    vec3<Scalar> rounding_radii;  //! The rounding ellipse.
    // TODO: Get rid of this and just depend on checking whether rounding_radii.x > 0
    bool has_rounding;    //! Whether or not the shape has rounding radii.
    };

//! Potential parameters for the ALJ potential.
struct pair_alj_params
    {
    DEVICE pair_alj_params()
        : epsilon(0.0), sigma_i(0.0), sigma_j(0.0), alpha(0)
        {}

    #ifndef NVCC
    //! Shape constructor
    pair_alj_params(Scalar _epsilon, Scalar _sigma_i, Scalar _sigma_j, unsigned int _alpha, bool use_device)
        : epsilon(_epsilon), sigma_i(_sigma_i), sigma_j(_sigma_j), alpha(_alpha) {}

    #endif

    //! Load dynamic data members into shared memory and increase pointer
    /*! \param ptr Pointer to load data to (will be incremented)
        \param available_bytes Size of remaining shared memory allocation
     */
    HOSTDEVICE void load_shared(char *& ptr, unsigned int &available_bytes) const {}

    #ifdef ENABLE_CUDA
    //! Attach managed memory to CUDA stream
    void attach_to_stream(cudaStream_t stream) const {}
    #endif

    //! Potential parameters
    Scalar epsilon;                      //! interaction parameter.
    Scalar sigma_i;                      //! size of i^th particle.
    Scalar sigma_j;                      //! size of j^th particle.
    unsigned int alpha;                  //! toggle switch of attractive branch of potential.
    };


/*!
 * Anisotropic LJ potential.
 */
template <unsigned int ndim>
class EvaluatorPairALJ
    {
    public:
        typedef pair_alj_params param_type;

        typedef alj_shape_params shape_param_type;

        //! Constructs the pair potential evaluator.
        /*! \param _dr Displacement vector between particle centers of mass.
            \param _rcutsq Squared distance at which the potential goes to 0.
            \param _q_i Quaternion of i^th particle.
            \param _q_j Quaternion of j^th particle.
            \param _params Per type pair parameters of this potential.
        */
        HOSTDEVICE EvaluatorPairALJ(Scalar3& _dr,
                                Scalar4& _qi,
                                Scalar4& _qj,
                                Scalar _rcutsq,
                                const param_type& _params)
            : dr(_dr),rcutsq(_rcutsq),qi(_qi),qj(_qj), _params(_params)
            {
            }

        //! uses diameter
        HOSTDEVICE static bool needsDiameter()
            {
            return true;
            }

        //! Whether the pair potential uses shape.
        HOSTDEVICE static bool needsShape()
            {
            return true;
            }

        //! Whether the pair potential needs particle tags.
        HOSTDEVICE static bool needsTags()
            {
            return true;
            }

        //! whether pair potential requires charges
        HOSTDEVICE static bool needsCharge()
            {
            return false;
            }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        HOSTDEVICE void setDiameter(Scalar di, Scalar dj)
            {
            dia_i = di;
            dia_j = dj;
            }

        //! Accept the optional shape values
        /*! \param shape_i Shape of particle i
            \param shape_j Shape of particle j
        */
        HOSTDEVICE void setShape(const shape_param_type *shapei, const shape_param_type *shapej)
            {
            shape_i = shapei;
            shape_j = shapej;
            }

        //! Accept the optional tags
        /*! \param tag_i Tag of particle i
            \param tag_j Tag of particle j
        */
        HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj)
        {
            tag_i = tagi;
            tag_j = tagj;
        }

        //! Accept the optional charge values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        HOSTDEVICE void setCharge(Scalar qi, Scalar qj){}

        //! Evaluate the force and energy.
        /*! \param force Output parameter to write the computed force.
            \param pair_eng Output parameter to write the computed pair energy.
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff.
            \param torque_i The torque exterted on the i^th particle.
            \param torque_j The torque exterted on the j^th particle.
            \return True if they are evaluated or false if they are not because we are beyond the cutoff.
        */
        HOSTDEVICE  bool
        evaluate(Scalar3& force, Scalar& pair_eng, bool energy_shift, Scalar3& torque_i, Scalar3& torque_j)
            {
            // Define relevant distance parameters (rsqr, r, directional vector)
            Scalar rsq = dot(dr, dr);
            Scalar r = sqrt(rsq);

            if (rsq < rcutsq)
                {
                // Call GJK. In order to ensure that Newton's third law is
                // obeyed, we must avoid any imbalance caused by numerical
                // errors leading to GJK(i, j) returning different results from
                // GJK(j, i). To prevent any such problems, we choose i and j
                // such that tag_i < tag_j.
                vec3<Scalar> v = vec3<Scalar>(), a = vec3<Scalar>(), b = vec3<Scalar>();
                    {
                    // Create local scope to avoid polluting the local scope
                    // with many unnecessary variables after GJK is done.
                    bool flip = (tag_i >= tag_j);
                    const ManagedArray<vec3<Scalar> > &verts1(flip ? shape_j->verts : shape_i->verts);
                    const ManagedArray<vec3<Scalar> > &verts2(flip ? shape_i->verts : shape_j->verts);
                    const quat<Scalar> &q1(flip ? qj : qi), &q2(flip ? qi : qj);
                    vec3<Scalar> dr_use(flip ? -dr : dr);

                    bool success, overlap;
                    // Note the signs of each of the vectors:
                    //    - v points from the contact point on verts2 to the contact points on verts1.
                    //    - a points from the centroid of verts1 to the contact points on verts1.
                    //    - b points from the centroid of verts2 to the contact points on verts2.
                    gjk<ndim>(verts1, verts2, v, a, b, success, overlap, q1, q2, dr_use, shape_i->rounding_radii, shape_j->rounding_radii, shape_i->has_rounding, shape_j->has_rounding);
                    assert(success && !overlap);

                    if (flip)
                        {
                        vec3<Scalar> a_tmp = a;
                        a = b - dr;
                        b = a_tmp - dr;
                        }
                    else
                        {
                        // We want v to be from verts1->verts2, but the value
                        // returned from GJK is from verts2->verts1.
                        v *= Scalar(-1.0);
                        }
                    }
                if (ndim == 2)
                    {
                    v.z = 0;
                    a.z = 0;
                    b.z = 0;
                    }

                Scalar sigma12 = (_params.sigma_i + _params.sigma_j)*Scalar(0.5);

                Scalar contact_sphere_radius_multiplier = 0.15;
                Scalar contact_sphere_radius = contact_sphere_radius_multiplier * sigma12;
                const Scalar two_p_16 = 1.12246204831;  // 2^(1/6)
                const Scalar shift_rho_diff = -0.25; // (1/(2^(1/6)))**12 - (1/(2^(1/6)))**6

                // The energy for the central potential must be rescaled by the
                // orientations (encoded in the a and b vectors).
                Scalar k1 = sqrt(dot(a, a));
                Scalar k2 = sqrt(dot(b, b));
                Scalar rho = sigma12 / (r - Scalar(0.5)*(k1/_params.sigma_i - 1.0) - Scalar(0.5)*(k2/_params.sigma_j - 1.0));
                Scalar invr_rsq = rho*rho;
                Scalar invr_6 = invr_rsq*invr_rsq*invr_rsq;
                Scalar numer = (invr_6*invr_6 - invr_6);

                invr_rsq = sigma12*sigma12/rsq;
                invr_6 = invr_rsq*invr_rsq*invr_rsq;
                Scalar invr_12 = invr_6*invr_6;
                Scalar denom = invr_12 - invr_6;

                Scalar four_epsilon = Scalar(4.0) * _params.epsilon;
                Scalar four_scaled_epsilon = four_epsilon * (numer/denom);

                // Define relevant vectors
                Scalar invnorm_v = fast::rsqrt(dot(v, v));
                Scalar f_scalar = 0;
                Scalar f_scalar_contact = 0;

                pair_eng = 0;

                // We must compute the central LJ force if we are including the
                // attractive component. For pure repulsive (WCA), we only need
                // to compute it if we are within the limited cutoff.
                if ((_params.alpha % 2 != 0) || (r < two_p_16*sigma12))
                    {
                    // Compute force and energy from LJ formula.
                    pair_eng += four_scaled_epsilon * (invr_12 - invr_6);
                    f_scalar = four_scaled_epsilon * ( Scalar(12.0)*invr_12 - Scalar(6.0)*invr_6 ) / (r);

                    // For the WCA case
                    if (_params.alpha % 2 == 0)
                        {
                        pair_eng -= four_scaled_epsilon * shift_rho_diff;
                        }
                    }

                // Similarly, we must compute the contact LJ force if we are
                // including the attractive component. For pure repulsive
                // (WCA), we only need to compute it if we are within the
                // limited cutoff associated with the contact point.
                if ((_params.alpha / 2 != 0) || (1 < two_p_16*contact_sphere_radius*invnorm_v))
                    {
                    // Contact force and energy
                    rho = contact_sphere_radius * invnorm_v;
                    invr_rsq = rho*rho;
                    invr_6 = invr_rsq*invr_rsq*invr_rsq;
                    invr_12 = invr_6*invr_6;
                    pair_eng += four_epsilon * (invr_12 - invr_6);
                    f_scalar_contact = four_epsilon * (Scalar(12.0)*invr_12 - Scalar(6.0)*invr_6) * invnorm_v;

                    // For the WCA case
                    if (_params.alpha / 2 == 0)
                        {
                        pair_eng -= four_epsilon * shift_rho_diff;
                        }
                    }

                // Net force
                vec3<Scalar> f_contact = -f_scalar_contact * v*invnorm_v;
                vec3<Scalar> f = f_scalar * (dr / r) + f_contact;
                if (ndim == 2)
                    {
                    f.z = 0;
                    }
                force = vec_to_scalar3(f);

                // Torque
                torque_i = vec_to_scalar3(cross(a, f_contact));
                torque_j = vec_to_scalar3(cross(dr + b, Scalar(-1.0)*f_contact));

                return true;
                }
              else
                {
                return false;
                }
            }

        #ifndef NVCC
        //! Get the name of the potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return "alj_table";
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        vec3<Scalar> dr;   //!< Stored dr from the constructor
        Scalar rcutsq;     //!< Stored rcutsq from the constructor
        quat<Scalar> qi;   //!< Orientation quaternion for particle i
        quat<Scalar> qj;   //!< Orientation quaternion for particle j
        Scalar dia_i;      //!< Diameter of particle i.
        Scalar dia_j;      //!< Diameter of particle j.
        unsigned int tag_i;      //!< Tag of particle i.
        unsigned int tag_j;      //!< Tag of particle j.
        const shape_param_type *shape_i;      //!< Shape parameters of particle i.
        const shape_param_type *shape_j;      //!< Shape parameters of particle j.
        const param_type& _params;      //!< Potential parameters for the pair of interest.
    };


#ifndef NVCC

// Note: This method assumes that shape_i == shape_j. This should be valid for
// all cases, and this logic will be moved up to the AnisoPotentialPair in
// HOOMD 3.0.
template <>
std::string EvaluatorPairALJ<2>::getShapeSpec() const
    {
    std::ostringstream shapedef;
    const ManagedArray<vec3<Scalar> > &verts(shape_i->verts);       //! Shape vertices.
    const unsigned int N = verts.size();
    shapedef << "{\"type\": \"Polygon\", \"rounding_radius\": 0, \"vertices\": [";
    for (unsigned int i = 0; i < N-1; i++)
        {
        shapedef << "[" << verts[i].x << ", " << verts[i].y << "], ";
        }
    shapedef << "[" << verts[N-1].x << ", " << verts[N-1].y << "]]}";
    return shapedef.str();
    }

// Note: This method assumes that shape_i == shape_j. This should be valid for
// all cases, and this logic will be moved up to the AnisoPotentialPair in
// HOOMD 3.0.
template <>
std::string EvaluatorPairALJ<3>::getShapeSpec() const
    {
    std::ostringstream shapedef;
    const ManagedArray<vec3<Scalar> > &verts(shape_i->verts);
    const unsigned int N = verts.size();
    if (N == 1)
        {
        shapedef << "{\"type\": \"Ellipsoid\", \"a\": " << shape_i->rounding_radii.x <<
                    ", \"b\": " << shape_i->rounding_radii.y <<
                    ", \"c\": " << shape_i->rounding_radii.z <<
                    "}";
        }
    else
        {
        if (shape_i->rounding_radii.x != shape_i->rounding_radii.y ||
                shape_i->rounding_radii.x != shape_i->rounding_radii.z)
            {
                throw std::runtime_error("Shape definition not supported for spheropolyhedra with distinct rounding radii.");
            }
        shapedef << "{\"type\": \"ConvexPolyhedron\", \"rounding_radius\": " << shape_i->rounding_radii.x << ", \"vertices\": [";
        for (unsigned int i = 0; i < N-1; i++)
            {
            shapedef << "[" << verts[i].x << ", " << verts[i].y << ", " << verts[i].z << "], ";
            }
        shapedef << "[" << verts[N-1].x << ", " << verts[N-1].y << ", " << verts[N-1].z << "]]}";
        }

    return shapedef.str();
    }
#endif



#ifndef NVCC
alj_shape_params make_alj_shape_params(pybind11::list shape, pybind11::list rounding_radii, std::shared_ptr<const ExecutionConfiguration> exec_conf)
    {
    alj_shape_params result(shape, rounding_radii, exec_conf->isCUDAEnabled());
    return result;
    }

pair_alj_params make_pair_alj_params(Scalar epsilon, Scalar sigma_i, Scalar sigma_j, unsigned int alpha, std::shared_ptr<const ExecutionConfiguration> exec_conf)
    {
    pair_alj_params result(epsilon, sigma_i, sigma_j, alpha, exec_conf->isCUDAEnabled());
    return result;
    }

//! Function to export the ALJ parameter type to python
void export_shape_params(pybind11::module& m)
{
    pybind11::class_<pair_alj_params>(m, "pair_alj_params")
        .def(pybind11::init<>())
        .def_readwrite("alpha", &pair_alj_params::alpha)
        .def_readwrite("epsilon", &pair_alj_params::epsilon)
        .def_readwrite("sigma_i", &pair_alj_params::sigma_i)
        .def_readwrite("sigma_j", &pair_alj_params::sigma_j);

    m.def("make_pair_alj_params", &make_pair_alj_params);
}

void export_alj_shape_params(pybind11::module& m)
    {
    pybind11::class_<alj_shape_params>(m, "alj_shape_params")
        .def(pybind11::init<>());

    m.def("make_alj_shape_params", &make_alj_shape_params);
    }
#endif

#endif // __EVALUATOR_PAIR_ALJ_H__
