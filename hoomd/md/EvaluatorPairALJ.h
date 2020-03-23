// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: vramasub

#ifndef __EVALUATOR_PAIR_ALJ_H__
#define __EVALUATOR_PAIR_ALJ_H__

#ifndef NVCC
#include <string>
#include <sstream>
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

// Note: delta is from the edge to the point.
HOSTDEVICE inline void
pointSegmentDistance(const vec3<Scalar> &point, const vec3<Scalar> &e1, const vec3<Scalar> &e2, vec3<Scalar> &delta, vec3<Scalar> &projection, Scalar &dist)
    {
    vec3<Scalar> edge = e1 - e2;
    Scalar edge_length_sq = dot(edge, edge);
    Scalar t = fmax(0.0, fmin(dot(point - e1, e2 - e1)/edge_length_sq, 1.0));
    projection = e1 - t * edge;
    delta = point - projection;

    // We MUST use sqrt, not rsqrt here. The difference in
    // precision is enough to cause noticeable violations in energy
    // conservation on the GPU (see the double-precision tables here:
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions).
    dist = sqrt(dot(delta, delta));
    }


// Note: delta is from the edge to the point.
HOSTDEVICE inline void
pointFaceDistancev1(const vec3<Scalar> &point, const vec3<Scalar> &f1, const vec3<Scalar> &f2, const vec3<Scalar> &f3, vec3<Scalar> &delta, vec3<Scalar> &projection, Scalar &dist)
    {
    // Use the method of Eberly
    // https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    const vec3<Scalar> &B = f1;
    const vec3<Scalar> E0 = f2 - B;
    const vec3<Scalar> E1 = f3 - B;
    const vec3<Scalar> &P = point;

    vec3<Scalar> D = f1 - P;
    Scalar a = dot(E0, E0);
    Scalar b = dot(E0, E1);
    Scalar c = dot(E1, E1);
    Scalar d = dot(E0, D);
    Scalar e = dot(E1, D);

    Scalar s = b*e - c*d;
    Scalar t = b*d - a*e;

    Scalar det = a*c - b*b;

    if (s+t <= det)
        {
        if (s < 0)
            {
            if (t < 0)
                {
                if (d < 0)
                    {
                    t = 0;
                    if (-d >= a)
                        {
                        s = 1;
                        }
                    else
                        {
                        s = -d/a;
                        }
                    }
                else
                    {
                    s = 0;
                    if (e >= 0)
                        {
                        t = 0;
                        }
                    else if (-e >= c)
                        {
                        t = 1;
                        }
                    else
                        {
                        t = -e/c;
                        }
                    }
                }
            else
                {
                s = 0;
                if (e >= 0)
                    {
                    t = 0;
                    }
                else if (-e >= c)
                    {
                    t = 1;
                    }
                else
                    {
                    t = -e/c;
                    }
                }
            }
        else if (t < 0)
            {
            t = 0;
            if (d >= 0)
                {
                s = 0;
                }
            else if (-d >= a)
                {
                s = 1;
                }
            else
                {
                s = -d/a;
                }
            }
        else
            {
            s /= det;
            t /= det;
            }
        }
    else
        {
        if (s < 0)
            {
            Scalar tmp0 = b + d;
            Scalar tmp1 = c + e;
            if (tmp1 > tmp0)
                {
                Scalar numer = tmp1 - tmp0;
                Scalar denom = a - 2*b + c;
                if (numer >= denom)
                    {
                    s = 1;
                    }
                else
                    {
                    s = numer/denom;
                    }
                t = 1-s;
                }
            else
                {
                s = 0;
                if (tmp1 <= 0)
                    {
                    t = 1;
                    }
                else if (e >= 0)
                    {
                    t = 0;
                    }
                else
                    {
                    t = -e/c;
                    }
                }
            }
        else if (t < 0)
            {
            Scalar tmp0 = b + e;
            Scalar tmp1 = a + d;
            if (tmp1 > tmp0)
                {
                Scalar numer = tmp1 - tmp0;
                Scalar denom = a - 2*b + c;
                if (numer >= denom)
                    {
                    t = 1;
                    }
                else
                    {
                    t = numer / denom;
                    }
                s = 1-t;
                }
            else
                {
                t = 0;
                if (tmp1 <= 0)
                    {
                    s = 1;
                    }
                else if (d >= 0)
                    {
                    s = 0;
                    }
                else
                    {
                    s = -d/a;
                    }
                }
            }
        else
            {
            Scalar numer = (c+e) - (b+d);
            if (numer <= 0)
                {
                s = 0;
                }
            else
                {
                Scalar denom = a - 2*b + c;
                if (numer > denom)
                    {
                    s = 1;
                    }
                else
                    {
                    s = numer/denom;
                    }
                }
            t = 1 - s;
            }
        }

    projection = B + s*E0 + t*E1;
    delta = point - projection;
    dist = sqrt(dot(delta, delta));
    }

HOSTDEVICE inline Scalar det3(const vec3<Scalar> &v1, const vec3<Scalar> &v2, const vec3<Scalar> &v3)
{
    // Compute the determinant of a matrix with columns v1, v2, and v3.
    return (v1.x*(v2.y*v3.z - v3.y*v2.z) +
            v2.x*(v3.y*v1.z - v1.y*v3.z) +
            v3.x*(v1.y*v2.z - v2.y*v1.z));
}

HOSTDEVICE inline Scalar clamp(const Scalar &x)
{
    return (x >= 0)*(x + (x > 1)*(1-x));
}

// Note: delta is from the edge to the point.
HOSTDEVICE inline void
pointFaceDistancev2(const vec3<Scalar> &point, const vec3<Scalar> &f1, const vec3<Scalar> &f2, const vec3<Scalar> &f3, vec3<Scalar> &delta, vec3<Scalar> &projection, Scalar &dist)
    {
    // This method performs a more brute force calculation than the Eberly
    // method that requires more computation but with far less branching. It
    // computes the point-triangle distance analytically by solving a system of
    // linear equations; however, that system will give incorrect results when
    // the projection of the point onto the plane of the triangle lies outside
    // the triangle. To resolve this problem, this method also computes all
    // point-edge distances. Since the answer we want is the minimizer of all
    // distances, if any of these is less than the point-face calculation we
    // know that it is the right answer.
    vec3<Scalar> v1 = f2 - f1;
    vec3<Scalar> v2 = f3 - f1;
    vec3<Scalar> normal = cross(v1, v2);

    vec3<Scalar> solution_vector = point - f1;

    Scalar denom = det3(v1, v2, normal);
    Scalar sub1 = det3(solution_vector, v2, normal);
    Scalar sub2 = det3(v1, solution_vector, normal);

    Scalar alpha = clamp(sub1/denom);
    Scalar beta = clamp(sub2/denom);

    Scalar total = alpha + beta;

    if (total > 1)
    {
        alpha /= total;
        beta /= total;
    }

    projection = f1 + alpha*v1 + beta*v2;
    delta = point - projection;
    dist = sqrt(dot(delta, delta));

    vec3<Scalar> guess_delta, guess_projection;
    Scalar guess_dist;

    // Now test all three edges.
    pointSegmentDistance(point, f1, f2, guess_delta, guess_projection, guess_dist);
    if (guess_dist < dist)
        {
        projection = guess_projection;
        delta = guess_delta;
        dist = guess_dist;
        }
    pointSegmentDistance(point, f1, f3, guess_delta, guess_projection, guess_dist);
    if (guess_dist < dist)
        {
        projection = guess_projection;
        delta = guess_delta;
        dist = guess_dist;
        }
    pointSegmentDistance(point, f2, f3, guess_delta, guess_projection, guess_dist);
    if (guess_dist < dist)
        {
        projection = guess_projection;
        delta = guess_delta;
        dist = guess_dist;
        }
    }


// This function is copied from DEM.
HOSTDEVICE inline Scalar detp(const vec3<Scalar> &m, const vec3<Scalar> &n, const vec3<Scalar> o, const vec3<Scalar> p)
    {
    return dot(m - n, o - p);
    }


// This function is copied from DEM.
HOSTDEVICE inline void
edgeEdgeDistance(const vec3<Scalar> &e00, const vec3<Scalar> &e01, const vec3<Scalar> &e10, const vec3<Scalar> &e11, vec3<Scalar> &closestI, vec3<Scalar> &closestJ, Scalar &closestDistsq)
    {
    // in the style of http://paulbourke.net/geometry/pointlineplane/
    Scalar denominator(detp(e01, e00, e01, e00)*detp(e11, e10, e11, e10) -
        detp(e11, e10, e01, e00)*detp(e11, e10, e01, e00));
    Scalar lambda0((detp(e00, e10, e11, e10)*detp(e11, e10, e01, e00) -
            detp(e00, e10, e01, e00)*detp(e11, e10, e11, e10))/denominator);
    Scalar lambda1((detp(e00, e10, e11, e10) +
            lambda0*detp(e11, e10, e01, e00))/detp(e11, e10, e11, e10));

    lambda0 = clamp(lambda0);
    lambda1 = clamp(lambda1);

    const vec3<Scalar> r0(e01 - e00);
    const Scalar r0sq(dot(r0, r0));
    const vec3<Scalar> r1(e11 - e10);
    const Scalar r1sq(dot(r1, r1));

    closestI = e00 + lambda0*r0;
    closestJ = e10 + lambda1*r1;
    vec3<Scalar> rContact(closestJ - closestI);
    closestDistsq = dot(rContact, rContact);

    Scalar lambda(clamp(dot(e10 - e00, r0)/r0sq));
    vec3<Scalar> candidateI(e00 + lambda*r0);
    vec3<Scalar> candidateJ(e10);
    rContact = candidateJ - candidateI;
    Scalar distsq(dot(rContact, rContact));
    if(distsq < closestDistsq)
        {
        closestI = candidateI;
        closestJ = candidateJ;
        closestDistsq = distsq;
        }

    lambda = clamp(dot(e11 - e00, r0)/r0sq);
    candidateI = e00 + lambda*r0;
    candidateJ = e11;
    rContact = candidateJ - candidateI;
    distsq = dot(rContact, rContact);
    if(distsq < closestDistsq)
        {
        closestI = candidateI;
        closestJ = candidateJ;
        closestDistsq = distsq;
        }

    lambda = clamp(dot(e00 - e10, r1)/r1sq);
    candidateI = e00;
    candidateJ = e10 + lambda*r1;
    rContact = candidateJ - candidateI;
    distsq = dot(rContact, rContact);
    if(distsq < closestDistsq)
        {
        closestI = candidateI;
        closestJ = candidateJ;
        closestDistsq = distsq;
        }

    lambda = clamp(dot(e01 - e10, r1)/r1sq);
    candidateI = e01;
    candidateJ = e10 + lambda*r1;
    rContact = candidateJ - candidateI;
    distsq = dot(rContact, rContact);
    if(distsq < closestDistsq)
        {
        closestI = candidateI;
        closestJ = candidateJ;
        closestDistsq = distsq;
        }

    if(fabs(1 - dot(r0, r1)*dot(r0, r1)/r0sq/r1sq) < 1e-6)
        {
        const Scalar lambda00(clamp(dot(e10 - e00, r0)/r0sq));
        const Scalar lambda01(clamp(dot(e11 - e00, r0)/r0sq));
        const Scalar lambda10(clamp(dot(e00 - e10, r1)/r1sq));
        const Scalar lambda11(clamp(dot(e01 - e10, r1)/r1sq));

        lambda0 = Scalar(.5)*(lambda00 + lambda01);
        lambda1 = Scalar(.5)*(lambda10 + lambda11);

        closestI = e00 + lambda0*r0;
        closestJ = e10 + lambda1*r1;
        }
    }


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
                vec3<Scalar> support_vectors1[ndim], support_vectors2[ndim];
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
                    //    - b points from the centroid of verts1 to the contact points on verts2.
                    gjk<ndim>(verts1, verts2, v, a, b, success, overlap, q1, q2, dr_use, shape_i->rounding_radii, shape_j->rounding_radii, shape_i->has_rounding, shape_j->has_rounding, support_vectors1, support_vectors2);
                    assert(success && !overlap);

                    if (flip)
                        {
                        vec3<Scalar> a_tmp = a;
                        a = b - dr;
                        b = a_tmp - dr;

                        // If parallel faces are a possibility, use the full facet vectors.
                        if (shape_i->verts.size() > ndim && shape_j->verts.size() > ndim)
                            {
                            for (unsigned int i = 0; i < ndim; ++i)
                                {
                                vec3<Scalar> tmp = support_vectors1[i];
                                support_vectors1[i] = support_vectors2[i] - dr;
                                support_vectors2[i] = tmp - dr;
                                }
                            }
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
                    if (shape_i->verts.size() > ndim && shape_j->verts.size() > ndim)
                        {
                        for (unsigned int i = 0; i < ndim; ++i)
                            {
                            support_vectors1[i].z = 0;
                            support_vectors2[i].z = 0;
                            }
                        }
                    }

                Scalar sigma12 = Scalar(0.5) * (_params.sigma_i + _params.sigma_j);

                Scalar contact_sphere_radius_multiplier = 0.15;
                Scalar contact_sphere_diameter = contact_sphere_radius_multiplier * sigma12;

                // The energy for the central potential must be rescaled by the
                // orientations (encoded in the a and b vectors).
                Scalar k1 = sqrt(dot(a, a));
                Scalar k2 = sqrt(dot(dr+b, dr+b));
                Scalar rho = sigma12 / (r - (k1 - Scalar(0.5)*_params.sigma_i) - (k2 - Scalar(0.5)*_params.sigma_j));
                Scalar invr_rsq = rho*rho;
                Scalar invr_6 = invr_rsq*invr_rsq*invr_rsq;
                Scalar numer = (invr_6*invr_6 - invr_6) - SHIFT_RHO_DIFF;

                invr_rsq = sigma12*sigma12/rsq;
                invr_6 = invr_rsq*invr_rsq*invr_rsq;
                Scalar invr_12 = invr_6*invr_6;
                Scalar denom = invr_12 - invr_6 - SHIFT_RHO_DIFF;
                Scalar scale_factor = denom != 0 ? (numer/denom) : 1;

                Scalar four_epsilon = Scalar(4.0) * _params.epsilon;
                Scalar four_scaled_epsilon = four_epsilon * scale_factor;

                // We must compute the central LJ force if we are including the
                // attractive component. For pure repulsive (WCA), we only need
                // to compute it if we are within the limited cutoff.
                Scalar f_scalar = 0;
                pair_eng = 0;
                if ((_params.alpha % 2 != 0) || (r < TWO_P_16*sigma12))
                    {
                    // Compute force and energy from LJ formula.
                    pair_eng += four_scaled_epsilon * (invr_12 - invr_6);
                    f_scalar = four_scaled_epsilon * (Scalar(12.0)*invr_12 - Scalar(6.0)*invr_6) / (r);

                    // For the WCA case
                    if (_params.alpha % 2 == 0)
                        {
                        pair_eng -= four_scaled_epsilon * SHIFT_RHO_DIFF;
                        }
                    }

                vec3<Scalar> f = f_scalar * (dr / r);
                // TODO: If we anticipate running heavily mixed systems of
                // ellipsoids and (sphero)poly[gons|hedra], we could get
                // significant performance benefits from templating here and
                // having multiple evaluators. However, that requires
                // significant reworkings of HOOMD internals, so it's probably
                // not worthwhile. We should consider if we ever want to enable
                // this during the HOOMD3.0 rewrite, though.
                if (shape_i->verts.size() > ndim && shape_j->verts.size() > ndim)
                    {
                    computeContactEnergy(
                            support_vectors1, support_vectors2, contact_sphere_diameter, four_epsilon, f, pair_eng, torque_i, torque_j);
                    }
                else
                    {
                    computeContactEnergy(
                            v, a, b, dr, contact_sphere_diameter, four_epsilon, f, pair_eng, torque_i, torque_j);
                    }
                force = vec_to_scalar3(f);

                if (ndim == 2)
                    {
                    force.z = 0;
                    torque_i.x = 0;
                    torque_i.y = 0;
                    torque_j.x = 0;
                    torque_j.y = 0;
                    }


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
            return "alj";
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:

        // Version of contact energy using a single contact point (when at
        // least one of the two shapes has no flat faces).
        HOSTDEVICE void computeContactEnergy(
                const vec3<Scalar> &v, const vec3<Scalar> &a, const vec3<Scalar> &b, const vec3<Scalar> &dr,
                const Scalar contact_sphere_diameter, const Scalar &four_epsilon,
                vec3<Scalar> &force, Scalar &pair_eng, Scalar3 &torque_i, Scalar3 &torque_j)
            {
            Scalar norm_v = sqrt(dot(v, v));
            vec3<Scalar> torquei, torquej;
            computeContactForceAndTorque(contact_sphere_diameter, v, norm_v, four_epsilon, a, dr+b, force, pair_eng, torquei, torquej);
            torque_i = vec_to_scalar3(torquei);
            torque_j = vec_to_scalar3(torquej);
            }

        // Version of contact energy using multiple contact points (when both
        // faces have some flat faces that could be arbitrarily close to parallel).
        // Must be implemented for a specific dimensionality, the default function
        // exists (but does nothing) to avoid undefined symbols.
        HOSTDEVICE void computeContactEnergy(
                const vec3<Scalar> support_vectors1[ndim], const vec3<Scalar> support_vectors2[ndim],
                const Scalar contact_sphere_diameter, const Scalar &four_epsilon,
                vec3<Scalar> &force, Scalar &pair_eng, Scalar3 &torque_i, Scalar3 &torque_j) {}

        // Core routine for calculating interaction between two points.
        // The contact points must be with respect to each particle's center of
        // mass (contact_point_i is relative to the origin, which is the center
        // of verts1, whereas contact_point_j is relative to the center of
        // verts2, which is -dr). The contact_vector must point from verts1 to verts2
        HOSTDEVICE inline void computeContactForceAndTorque(const Scalar contact_sphere_diameter, const vec3<Scalar> &contact_vector, const Scalar contact_distance, const Scalar four_epsilon, const vec3<Scalar> &contact_point_i, const vec3<Scalar> &contact_point_j,
                vec3<Scalar> &force, Scalar &pair_eng, vec3<Scalar> &torque_i, vec3<Scalar> &torque_j)
            {
            if ((_params.alpha / 2 != 0) || (1 < TWO_P_16*contact_sphere_diameter / contact_distance))
                {
                Scalar rho = contact_sphere_diameter / contact_distance;
                Scalar invr_rsq = rho*rho;
                Scalar invr_6 = invr_rsq*invr_rsq*invr_rsq;
                Scalar invr_12 = invr_6*invr_6;
                Scalar energy_contact = four_epsilon * (invr_12 - invr_6);
                Scalar scalar_force_contact = four_epsilon * (Scalar(12.0)*invr_12 - Scalar(6.0)*invr_6) / contact_distance;

                // For the WCA case
                if (_params.alpha / 2 == 0)
                    {
                    energy_contact -= four_epsilon * SHIFT_RHO_DIFF;
                    }

                pair_eng += energy_contact;
                vec3<Scalar> force_contact = -scalar_force_contact * contact_vector / contact_distance;
                force += force_contact;

                torque_i += cross(contact_point_i, force_contact);
                torque_j += cross(contact_point_j, Scalar(-1.0) * force_contact);
                }
            }

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

        constexpr static Scalar TWO_P_16 = 1.12246204831;  // 2^(1/6)
        constexpr static Scalar SHIFT_RHO_DIFF = -0.25; // (1/(2^(1/6)))**12 - (1/(2^(1/6)))**6
    };


template <>
HOSTDEVICE inline void EvaluatorPairALJ<2>::computeContactEnergy(
        const vec3<Scalar> support_vectors1[2], const vec3<Scalar> support_vectors2[2],
        const Scalar contact_sphere_diameter, const Scalar &four_epsilon,
        vec3<Scalar> &force, Scalar &pair_eng, Scalar3 &torque_i, Scalar3 &torque_j)
    {
    vec3<Scalar> torquei, torquej;
    vec3<Scalar> projection, vec;
    Scalar dist;
    // First compute the interaction of the verts on 1 to the edge on 2.
    for (unsigned int i = 0; i < 2; ++i)
        {
        pointSegmentDistance(support_vectors1[i], support_vectors2[0], support_vectors2[1], vec, projection, dist);
        computeContactForceAndTorque(contact_sphere_diameter, -vec, dist, four_epsilon, support_vectors1[i], dr + support_vectors1[i] - vec, force, pair_eng, torquei, torquej);
        }

    // Now compute the interaction of the verts on 2 to the edge on 1.
    for (unsigned int i = 0; i < 2; ++i)
        {
        pointSegmentDistance(support_vectors2[i], support_vectors1[0], support_vectors1[1], vec, projection, dist);

        computeContactForceAndTorque(contact_sphere_diameter, vec, dist, four_epsilon, projection, dr + support_vectors2[i], force, pair_eng, torquei, torquej);
        }
    torque_i = vec_to_scalar3(torquei);
    torque_j = vec_to_scalar3(torquej);
    }

template <>
HOSTDEVICE inline void EvaluatorPairALJ<3>::computeContactEnergy(
        const vec3<Scalar> support_vectors1[3], const vec3<Scalar> support_vectors2[3],
        const Scalar contact_sphere_diameter, const Scalar &four_epsilon,
        vec3<Scalar> &force, Scalar &pair_eng, Scalar3 &torque_i, Scalar3 &torque_j)
    {
    vec3<Scalar> torquei, torquej;
    vec3<Scalar> projection, vec;
    Scalar dist;
    // Compute the interaction of the verts on 1 to the face on 2.
    for (unsigned int i = 0; i < 3; ++i)
        {
        pointFaceDistancev1(support_vectors1[i], support_vectors2[0], support_vectors2[1], support_vectors2[2], vec, projection, dist);
        computeContactForceAndTorque(contact_sphere_diameter, -vec, dist, four_epsilon, support_vectors1[i], dr + support_vectors1[i] - vec, force, pair_eng, torquei, torquej);
        }

    // Compute the interaction of the verts on 2 to the edge on 1.
    for (unsigned int i = 0; i < 3; ++i)
        {
        pointFaceDistancev1(support_vectors2[i], support_vectors1[0], support_vectors1[1], support_vectors1[2], vec, projection, dist);

        computeContactForceAndTorque(contact_sphere_diameter, vec, dist, four_epsilon, projection, dr + support_vectors2[i], force, pair_eng, torquei, torquej);
        }

    // Compute interaction between pairs of edges.
    for (unsigned int i = 0; i < 3; ++i)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            vec3<Scalar> closestI, closestJ;
            Scalar distsq;
            edgeEdgeDistance(support_vectors1[i], support_vectors1[(i+1)%3], support_vectors2[j], support_vectors2[(j+1)%3], closestI, closestJ, distsq);
            vec3<Scalar> vec = closestJ-closestI;
        computeContactForceAndTorque(contact_sphere_diameter, vec, sqrt(distsq), four_epsilon, closestI, closestJ, force, pair_eng, torquei, torquej);
        }
    }
    torque_i = vec_to_scalar3(torquei);
    torque_j = vec_to_scalar3(torquej);
    }



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
