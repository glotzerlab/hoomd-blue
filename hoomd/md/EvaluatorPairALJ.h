// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: vramasub

#ifndef __EVALUATOR_PAIR_ALJ_H__
#define __EVALUATOR_PAIR_ALJ_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/VectorMath.h"
#include "hoomd/ManagedArray.h"
#include "ALJData.h"
#include "GJK.h"
#include <iostream>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

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

/*!
 * Anisotropic LJ potential (assuming analytical kernel and (temporarily) sigma = 1.0)
 */


struct single_shape_table
    {
    HOSTDEVICE single_shape_table()
        : k_maxsq(0.0)
        {}

    #ifndef NVCC

    //! Shape constructor
    single_shape_table(pybind11::list shape, bool use_device)
        : k_maxsq(0.0)
        {
        Scalar kmax = 0;

        //! Construct table for particle i
        unsigned int N = len(shape);
        verts = ManagedArray<vec3<Scalar> >(N, use_device);
        for (unsigned int i = 0; i < N; ++i)
            {
            pybind11::list shape_tmp = pybind11::cast<pybind11::list>(shape[i]);
            verts[i] = vec3<Scalar>(pybind11::cast<Scalar>(shape_tmp[0]), pybind11::cast<Scalar>(shape_tmp[1]), pybind11::cast<Scalar>(shape_tmp[2]));

            Scalar ktest = dot(verts[i], verts[i]);
            if (ktest > kmax)
                {
                kmax = ktest;
                }
            }
        k_maxsq = kmax;
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
    Scalar k_maxsq;                          //! Largest kernel value.
    };

#ifndef NVCC
single_shape_table make_single_shape_table(pybind11::list shape, std::shared_ptr<const ExecutionConfiguration> exec_conf)
    {
    single_shape_table result(shape, exec_conf->isCUDAEnabled());
    return result;
    }
#endif

template <unsigned int ndim>
class EvaluatorPairALJ
    {
    public:
        typedef shape_table param_type;

        typedef single_shape_table shape_param_type;

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

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        HOSTDEVICE void setDiameter(Scalar di, Scalar dj)
            {
            dia_i = di;
            dia_j = dj;
            }

        //! whether pair potential requires charges
        HOSTDEVICE static bool needsCharge()
            {
            return false;
            }

        //! Whether the pair potential uses shape.
        HOSTDEVICE static bool needsShape()
            {
            return true;
            }

        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        HOSTDEVICE void setCharge(Scalar qi, Scalar qj){}

        //! Accept the optional shape values
        /*! \param shape_i Shape of particle i
            \param shape_j Shape of particle j
        */
        HOSTDEVICE void setShape(const shape_param_type *shapei, const shape_param_type *shapej)
            {
            shape_i = shapei;
            shape_j = shapej;
            }

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
            Scalar rsq = dot(dr,dr);
            Scalar r = sqrt(rsq);
            vec3<Scalar> unitr = dr/r;

            // Interaction cutoff is scaled by the max kernel value scaled by
            // the insphere radius, which is the max vertex distance
            // k[ij]_maxsq.
            if ( (rsq/_params.ki_maxsq < rcutsq) || (rsq/_params.kj_maxsq < rcutsq) )
                {
                // Call GJK. In order to ensure that Newton's third law is
                // obeyed, we must avoid any imbalance caused by numerical
                // errors leading to GJK(i, j) returning different results from
                // GJK(j, i). To prevent any such problems, we simply always
                // call GJK twice, once in each direction, and choose the
                // result corresponding to the smaller distance.
                vec3<Scalar> v = vec3<Scalar>(), a = vec3<Scalar>(), b = vec3<Scalar>();
                    {
                    vec3<Scalar> v1 = vec3<Scalar>(), a1 = vec3<Scalar>(), b1 = vec3<Scalar>();
                    vec3<Scalar> v2 = vec3<Scalar>(), a2 = vec3<Scalar>(), b2 = vec3<Scalar>();
                    bool success1, overlap1;
                    bool success2, overlap2;

                    gjk<ndim>(shape_i->verts, shape_j->verts, v1, a1, b1, success1, overlap1, qi, qj, dr);
                    gjk<ndim>(shape_j->verts, shape_i->verts, v2, a2, b2, success2, overlap2, qj, qi, -dr);

                    if (dot(v1, v1) < dot(v2, v2))
                        {
                        v = v1;
                        a = a1;
                        b = b1;
                        }
                    else
                        {
                        v = -v2;
                        a = b2 - dr;
                        b = a2 - dr;
                        }
                    }
                if (ndim == 2)
                    {
                    v.z = 0;
                    a.z = 0;
                    b.z = 0;
                    }

                // Get kernel
                Scalar sigma12 = (_params.sigma_i + _params.sigma_j)*Scalar(0.5);

                Scalar sub_sphere = 0.15;
                const Scalar two_p_16 = 1.12246204831;  // 2^(1/6)

                vec3<Scalar> f;
                vec3<Scalar> rvect;

                Scalar k1 = sqrt(dot(a,a));
                Scalar k2 = sqrt(dot(b,b));
                Scalar rho = sigma12 / (r - Scalar(0.5)*(k1/_params.sigma_i - 1.0) - Scalar(0.5)*(k2/_params.sigma_j - 1.0));
                Scalar invr_rsq = rho*rho;
                Scalar invr_6 = invr_rsq*invr_rsq*invr_rsq;
                Scalar numer = (invr_6*invr_6 - invr_6);

                invr_rsq = sigma12*sigma12/rsq;
                invr_6 = invr_rsq*invr_rsq*invr_rsq;
                Scalar invr_12 = invr_6*invr_6;
                Scalar denom = invr_12 - invr_6;

                Scalar epsilon = _params.epsilon;
                Scalar scaled_epsilon = epsilon*(numer/denom);

                // Define relevant vectors
                rvect = Scalar(-1.0)*v;
                Scalar vsq = dot(v,v);
                Scalar rcheck_isq = fast::rsqrt(vsq);
                rvect = rvect*rcheck_isq;
                Scalar f_scalar = 0;
                Scalar f_scalar_contact = 0;

                // Check repulsion vs attraction for center particle
                if (_params.alpha < 1.0)
                    {
                    if (r < two_p_16*sigma12)
                        {
                        // Center force and energy
                        pair_eng = Scalar(4.0) * scaled_epsilon * (invr_12 - invr_6);
                        f_scalar = Scalar(4.0) * scaled_epsilon * ( Scalar(12.0)*invr_12 - Scalar(6.0)*invr_6 ) / (r);

                        // Shift energy
                        rho = 1.0 / two_p_16;
                        invr_rsq = rho*rho;
                        invr_6 = invr_rsq*invr_rsq*invr_rsq;
                        pair_eng -= Scalar(4.0) * scaled_epsilon * (invr_6*invr_6 - invr_6);
                        }
                    else
                        {
                        pair_eng = 0.0;
                        }
                    }
                else
                    {
                    // Center force and energy
                    pair_eng = Scalar(4.0) * scaled_epsilon * (invr_12 - invr_6);
                    f_scalar = Scalar(4.0) * scaled_epsilon * ( Scalar(12.0)*invr_12 - Scalar(6.0)*invr_6 ) / r;
                    }

                // Check repulsion attraction for contact point
                // No overlap
                if (_params.alpha*0.0 < 1.0)
                    {
                    if (1/rcheck_isq  < two_p_16 *sub_sphere*sigma12)
                        {
                        // Contact force and energy
                        rho = sub_sphere * sigma12 * rcheck_isq;
                        invr_rsq = rho*rho;
                        invr_6 = invr_rsq*invr_rsq*invr_rsq;
                        pair_eng += Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                        f_scalar_contact = Scalar(4.0) * (epsilon) *  ( Scalar(12.0)*invr_6*invr_6 - Scalar(6.0)*invr_6 ) * rcheck_isq;

                        // Shift energy
                        rho = 1.0 / two_p_16;
                        invr_rsq = rho*rho;
                        invr_6 = invr_rsq*invr_rsq*invr_rsq;
                        pair_eng -= Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                        }
                    }
                else
                    {
                    // Contact force and energy
                    rho = sub_sphere * sigma12 * rcheck_isq;
                    invr_rsq = rho*rho;
                    invr_6 = invr_rsq*invr_rsq*invr_rsq;
                    pair_eng += Scalar(4.0) * epsilon * (invr_6*invr_6 - Scalar(1.0)*invr_6);
                    f_scalar_contact = Scalar(4.0) * (epsilon) *  ( Scalar(12.0)*invr_6*invr_6 - Scalar(6.0)*invr_6 ) * rcheck_isq;
                    }

                // Net force
                f = f_scalar * unitr - f_scalar_contact * rvect;
                if (ndim == 2)
                    {
                    f.z = 0;
                    }
                force = vec_to_scalar3(f);

                // Torque
                vec3<Scalar> lever = 0.5*sub_sphere*sigma12*rvect;
                torque_i = vec_to_scalar3(cross(a - lever + rvect/rcheck_isq, f));
                torque_j = vec_to_scalar3(cross(dr + a + lever, Scalar(-1.0)*f));

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
        #endif

    protected:
        vec3<Scalar> dr;   //!< Stored dr from the constructor
        Scalar rcutsq;     //!< Stored rcutsq from the constructor
        quat<Scalar> qi;   //!< Orientation quaternion for particle i
        quat<Scalar> qj;   //!< Orientation quaternion for particle j
        Scalar dia_i;
        Scalar dia_j;
        const shape_param_type *shape_i;
        const shape_param_type *shape_j;
        const param_type& _params;
    };


#ifndef NVCC
void export_single_shape_table(pybind11::module& m)
    {
    pybind11::class_<single_shape_table>(m, "single_shape_table")
        .def(pybind11::init<>())
        .def_readwrite("k_maxsq", &single_shape_table::k_maxsq);

    m.def("make_single_shape_table", &make_single_shape_table);
    }
#endif

#endif // __EVALUATOR_PAIR_ALJ_H__
