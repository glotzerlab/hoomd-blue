// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __EVALUATOR_PAIR_ALJ_H__
#define __EVALUATOR_PAIR_ALJ_H__

#ifndef __HIPCC__
#include <sstream>
#include <string>
#endif

#include "GJK_SV.h"
#include "hoomd/ManagedArray.h"
#include "hoomd/VectorMath.h"

#ifndef __HIPCC__
#include "hoomd/ExecutionConfiguration.h"
#include <pybind11/pybind11.h>
#endif

/*! \file EvaluatorPairALJ.h
    \brief Defines an evaluator class for the anisotropic LJ potential.
*/

// need to declare these class methods with __host__ __device__ qualifiers when building in nvcc
//! HOSTDEVICE is __host__ __device__ when included in nvcc and blank when included into the host
//! compiler
#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE

#if !defined(__HIPCC__) && defined(__SSE__)
#include <immintrin.h>
#endif
#endif

namespace hoomd
    {
namespace md
    {

using namespace detail;

/*! Clip a value between 0 and 1 */
HOSTDEVICE inline Scalar clip(const Scalar& x)
    {
    return (x >= 0) * (x + (x > 1) * (1 - x));
    }

// Note: delta is from the edge to the point.
//! Compute the distance from a point to a line segment in 2D.
/*! All the points (point, e1, and e2) must be provided in the same coordinate
 * system i.e. relative to the same origin. The delta and projection arguments
 * are overwritten by reference. The projection is provided in the same
 * coordinate system as the input points, and the delta vector points from the
 * projection to the input point (i.e. from the segment to the point).
 *
 * \param point The point to which to calculate distance.
 * \param e1 The first endpoint of the segment from which to calculate distance.
 * \param e2 The first endpoint of the segment from which to calculate distance.
 * \param delta The vector pointing from the closest point on the input segment to the input point
 * (updated by reference).
 * \param projection The closest point on the segment defined by e2-e1 to the input point (updated
 * by reference).
 */
HOSTDEVICE inline void pointSegmentDistance(const vec3<Scalar>& point,
                                            const vec3<Scalar>& e1,
                                            const vec3<Scalar>& e2,
                                            vec3<Scalar>& delta,
                                            vec3<Scalar>& projection)
    {
    vec3<Scalar> edge = e1 - e2;
    Scalar edge_length_sq = dot(edge, edge);
    Scalar t = clip(dot(point - e1, e2 - e1) / edge_length_sq);
    projection = e1 - t * edge;
    delta = point - projection;
    }

//! Convert a quaternion to a rotation matrix.
/*! The ALJ potential requires performing a large number of repeated rotations
 * to evaluate the support function in GJK. This problem is exacerbated for
 * polyhedra due to the numerous distance calculations that must be performed
 * to average over all relevant interactions. Since the orientation of the
 * particles is constant throughout these calculations, we can reduce the cost
 * by converting the orientation quaternions to rotation matrices up front and
 * reusing them.
 *
 * This implementation uses 12 multiplications and 12 additions, which is
 * the minimal number of operations according to Wikipedia
 * (https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Comparison_with_other_representations_of_rotations).
 * Effort is also made using extra braces to indicate potential local
 * optimizations, but a more optimal sequence of operations may exist.
 *
 * \param q The orientation quaternion to convert.
 * \param mat The output matrix (overwritten by reference).
 */
HOSTDEVICE inline void quat2mat(const quat<Scalar>& q, Scalar (&mat)[3][3])
    {
    Scalar two_x = Scalar(2.0) * q.v.x;
    Scalar two_y = Scalar(2.0) * q.v.y;
    Scalar two_z = Scalar(2.0) * q.v.z;
    Scalar two_x_sq = q.v.x * two_x;
    Scalar two_y_sq = q.v.y * two_y;
    Scalar two_z_sq = q.v.z * two_z;

    mat[0][0] = Scalar(1.0) - two_y_sq - two_z_sq;
    mat[1][1] = Scalar(1.0) - two_x_sq - two_z_sq;
    mat[2][2] = Scalar(1.0) - two_x_sq - two_y_sq;

        {
        Scalar y_two_z = q.v.y * two_z;
        Scalar s_two_x = q.s * two_x;
        mat[1][2] = y_two_z - s_two_x;
        mat[2][1] = y_two_z + s_two_x;
        }

        {
        Scalar x_two_y = q.v.x * two_y;
        Scalar s_two_z = q.s * two_z;
        mat[0][1] = x_two_y - s_two_z;
        mat[1][0] = x_two_y + s_two_z;
        }

        {
        Scalar x_two_z = q.v.x * two_z;
        Scalar s_two_y = q.s * two_y;
        mat[0][2] = x_two_z + s_two_y;
        mat[2][0] = x_two_z - s_two_y;
        }
    }

//! Anisotropic LJ (ALJ) potential.
/*! The ALJ potential is a generalization of Lennard-Jones potential for convex
 * anisotropic shapes. The potential is defined by a mean-field approximation
 * to the total interaction energy between a pair of anistropic shapes that
 * reduces the total interaction to a simple sum between 1) a contact
 * interaction at between the closest points on each shape to the other, and 2)
 * a center-center interaction analogous to the standard isotropic LJ
 * potential.
 */
template<unsigned int ndim> class EvaluatorPairALJ
    {
    public:
    //! Potential parameters for the ALJ potential.
    struct param_type
        {
        HOSTDEVICE param_type()
            : epsilon(0.0), sigma_i(0.0), sigma_j(0.0), contact_sigma_i(0.0), contact_sigma_j(0.0),
              alpha(0), average_simplices(false)
            {
            }

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
         *  \param available_bytes Size of remaining shared memory allocation
         */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifndef __HIPCC__
        //! Shape constructor
        param_type(Scalar _epsilon,
                   Scalar _sigma_i,
                   Scalar _sigma_j,
                   Scalar _contact_sigma_i,
                   Scalar _contact_sigma_j,
                   unsigned int _alpha,
                   bool _average_simplices,
                   bool managed)
            : epsilon(_epsilon), sigma_i(_sigma_i), sigma_j(_sigma_j),
              contact_sigma_i(_contact_sigma_i), contact_sigma_j(_contact_sigma_j), alpha(_alpha),
              average_simplices(_average_simplices)
            {
            }

        param_type(pybind11::dict params, bool managed)
            : epsilon(params["epsilon"].cast<Scalar>()), sigma_i(params["sigma_i"].cast<Scalar>()),
              sigma_j(params["sigma_j"].cast<Scalar>()),
              contact_sigma_i(params["sigma_i"].cast<Scalar>()
                              * params["contact_ratio_i"].cast<Scalar>()),
              contact_sigma_j(params["sigma_j"].cast<Scalar>()
                              * params["contact_ratio_j"].cast<Scalar>()),
              alpha(params["alpha"].cast<unsigned int>()),
              average_simplices(params["average_simplices"].cast<bool>())
            {
            }

        pybind11::object toPython()
            {
            using namespace pybind11::literals;
            return pybind11::dict("epsilon"_a = epsilon,
                                  "sigma_i"_a = sigma_i,
                                  "sigma_j"_a = sigma_j,
                                  "contact_ratio_i"_a = contact_sigma_i / sigma_i,
                                  "contact_ratio_j"_a = contact_sigma_j / sigma_j,
                                  "alpha"_a = alpha,
                                  "average_simplices"_a = average_simplices);
            }

#endif

#ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void set_memory_hint() const { }
#endif

        //! Potential parameters
        Scalar epsilon;         //! Interaction energy scale.
        Scalar sigma_i;         //! Size of particle i.
        Scalar sigma_j;         //! Size of particle j.
        Scalar contact_sigma_i; //! Size of contact sphere on particle i.
        Scalar contact_sigma_j; //! Size of contact sphere on particle j.
        unsigned int alpha;     //! Toggle switch of attractive branch of potential (0 for all
                            //! repulsive, 3 for all attractive, 1 for center-center attraction, 2
                            //! for contact-contact attraction).
        bool average_simplices; //! Whether or not to average interactions over simplices.
        };

    //! Shape parameters for the ALJ potential.
    /*! The ALJ potential in its current form can be used to handle a generalized
     * class of shapes composed of the Minkowski sum of an ellipsoid and a convex
     * polyhedron. The shape parameters here are sufficiently general as to handle
     * all possible cases in both 2D and 3D. The shape is stored as a set of
     * (optional) vertices that trace out the underlying polytope (polygon in 2D,
     * polyhedron in 3D) and a set of 3 rounding radii stored as a vec3. For
     * simulations of polytopes, the contact point formalism is inherently unstable
     * due to the oscillations that may occur when facets are nearly parallel. To
     * resolve this, the ALJ potential implements a form of face averaging that is
     * essentially a local form of the discrete element method (DEM). To do this,
     * it sums over all relevant point-edge interactions in 2D, and in 3D it sums
     * both point-face interactions and edge-edge interactions. To support this
     * calculation over an entire polytope, the shape params also encode the faces.
     * The faces are stored in two separate arrays. The faces array is a linear
     * container that indexes into the vertices to indicate which vertices compose
     * each face. Since this array is linear, the face_offsets array is used to
     * indicate the index in teh faces array corresponding to the start of each
     * face of the polytope.
     */
    struct shape_type
        {
        HOSTDEVICE shape_type() : has_rounding(false) { }

#ifndef __HIPCC__

        //! Shape constructor
        /*! \param vertices Nested pybind list that translates to an Nx3 set of vertices
            \param faces_ Nested pybind list that contains indices into vertices corresponding to
           each face.
           \param rr The semimajor axes of the rounding ellipse.
           \param managed
           Whether or not the shape params are managed on the host, forwarded through to underlying
           arrays for migration to the GPU as needed.

           TODO: This constructor does not validate that 3D structures have faces set, since
           shape_type does not know the dimension of the composing class.
         */
        shape_type(pybind11::dict shape, bool managed) : has_rounding(false)
            {
            // Store the rounding radii.
            auto rr = shape["rounding_radii"].cast<pybind11::tuple>();
            rounding_radii.x = pybind11::cast<Scalar>(rr[0]);
            rounding_radii.y = pybind11::cast<Scalar>(rr[1]);
            rounding_radii.z = pybind11::cast<Scalar>(rr[2]);
            if (rounding_radii.x > 0 || rounding_radii.y > 0 || rounding_radii.z > 0)
                {
                has_rounding = true;
                }

            // Check to see if vertices and faces are empty. If so then, the shape is an ellipsoid,
            // and we don't need to iterate over polytope based attributes.
            const auto vertices = shape["vertices"].cast<pybind11::list>();
            const auto N_vertices = static_cast<unsigned int>(len(vertices));

            const auto faces_ = shape["faces"].cast<pybind11::list>();
            const auto N_faces = static_cast<unsigned int>(len(faces_));

            // Simulating an ellipsoid just return
            if (N_vertices == 0 && N_faces == 0)
                {
                // The GJK implementation assumes that all shapes have at least one
                // vertex. For ellipsoids, we must use the origin as that vertex since it
                // guarantees that the polyhedral component of the support function will
                // always be 0 and not contribute to the total support function.
                verts = ManagedArray<vec3<Scalar>>(1, managed);
                verts[0] = vec3<Scalar>(0, 0, 0);
                faces = ManagedArray<unsigned int>(1, managed);
                faces[0] = 0;
                face_offsets = ManagedArray<unsigned int>(1, managed);
                face_offsets[0] = 0;
                return;
                }

            if (N_vertices == 0)
                {
                throw std::runtime_error("Cannot have empty vertices list.");
                }

            verts = ManagedArray<vec3<Scalar>>(N_vertices, managed);
            face_offsets = ManagedArray<unsigned int>(N_faces, managed);

            // Unpack the list[list] of vertices into a ManagedArray.
            for (unsigned int i = 0; i < N_vertices; ++i)
                {
                pybind11::list vertices_tmp = pybind11::cast<pybind11::list>(vertices[i]);
                verts[i] = vec3<Scalar>(pybind11::cast<Scalar>(vertices_tmp[0]),
                                        pybind11::cast<Scalar>(vertices_tmp[1]),
                                        pybind11::cast<Scalar>(vertices_tmp[2]));
                }

            // If no faces exist i.e. 2D then we make an empty array and return.
            if (N_faces == 0)
                {
                faces = ManagedArray<unsigned int>(0, managed);
                return;
                }

            // Since we don't know the total number of face indices a priori (we only have access to
            // the top-level list), we first construct the face offsets array.  Then, we allocate
            // the faces array and loop a second time to store those indices linearly.
            face_offsets[0] = 0;

            for (unsigned int i = 0; i < (N_faces - 1); ++i)
                {
                pybind11::list faces_tmp = pybind11::cast<pybind11::list>(faces_[i]);
                face_offsets[i + 1] = face_offsets[i] + static_cast<unsigned int>(len(faces_tmp));
                }
            pybind11::list faces_tmp = pybind11::cast<pybind11::list>(faces_[N_faces - 1]);
            const unsigned int total_face_indices
                = face_offsets[N_faces - 1] + static_cast<unsigned int>(len(faces_tmp));

            faces = ManagedArray<unsigned int>(total_face_indices, managed);
            unsigned int counter = 0;
            for (unsigned int i = 0; i < N_faces; ++i)
                {
                pybind11::list face_tmp = pybind11::cast<pybind11::list>(faces_[i]);
                for (unsigned int j = 0; j < len(face_tmp); ++j)
                    {
                    faces[counter] = pybind11::cast<unsigned int>(face_tmp[j]);
                    ++counter;
                    }
                }
            }

        pybind11::object toPython()
            {
            pybind11::list vertices;

            // ellipsoids are stored as having 1 vertex, in this case, do not add
            // vertices to the list for python
            if (verts.size() >= 2)
                {
                for (unsigned int i = 0; i < verts.size(); ++i)
                    {
                    auto& position = verts[i];
                    vertices.append(pybind11::make_tuple(position.x, position.y, position.z));
                    }
                }

            pybind11::list py_faces;

            // ellipsoids are stored as having a single face offset of 0, in
            // this case, do not add faces to the list for python
            if (face_offsets.size() >= 2)
                {
                unsigned int current_face_index = 0;
                for (unsigned int i = 1; i < face_offsets.size(); ++i)
                    {
                    pybind11::list face_vertices;
                    for (; current_face_index < face_offsets[i]; ++current_face_index)
                        {
                        face_vertices.append(faces[current_face_index]);
                        }
                    py_faces.append(face_vertices);
                    }

                // Copy final face
                pybind11::list face_vertices;
                for (; current_face_index < faces.size(); ++current_face_index)
                    {
                    face_vertices.append(faces[current_face_index]);
                    }
                py_faces.append(face_vertices);
                }

                {
                using namespace pybind11::literals;
                return pybind11::dict(
                    "vertices"_a = vertices,
                    "faces"_a = py_faces,
                    "rounding_radii"_a
                    = pybind11::make_tuple(rounding_radii.x, rounding_radii.y, rounding_radii.z));
                }
            }

#endif

        //! Load dynamic data members into shared memory and increase pointer
        /*! \param ptr Pointer to load data to (will be incremented)
            \param available_bytes Size of remaining shared memory allocation
         */
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes)
            {
            verts.load_shared(ptr, available_bytes);
            faces.load_shared(ptr, available_bytes);
            face_offsets.load_shared(ptr, available_bytes);
            }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const
            {
            verts.allocate_shared(ptr, available_bytes);
            faces.allocate_shared(ptr, available_bytes);
            face_offsets.allocate_shared(ptr, available_bytes);
            }

#ifdef ENABLE_HIP
        //! Attach managed memory to CUDA stream
        void set_memory_hint() const
            {
            verts.set_memory_hint();
            faces.set_memory_hint();
            face_offsets.set_memory_hint();
            }
#endif

        //! Shape parameters
        ManagedArray<vec3<Scalar>> verts;        //! Shape vertices.
        ManagedArray<unsigned int> faces;        //! Shape faces.
        ManagedArray<unsigned int> face_offsets; //! Index where each faces starts.
        vec3<Scalar> rounding_radii;             //! The semimajor axes of the rounding ellipse.
        bool has_rounding;                       //! Whether or not the shape has rounding radii.
        };

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
        : dr(_dr), rcutsq(_rcutsq), qi(_qi), qj(_qj), _params(_params)
        {
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

    //! Whether pair potential requires charges
    HOSTDEVICE static bool needsCharge()
        {
        return false;
        }

    /// Whether the potential implements the energy_shift parameter
    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
        return false;
        }

    //! Accept the optional shape values
    /*! \param shape_i Shape of particle i
        \param shape_j Shape of particle j
    */
    HOSTDEVICE void setShape(const shape_type* shapei, const shape_type* shapej)
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
    HOSTDEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate the force and energy.
    /*! \param force Output parameter to write the computed force.
        \param pair_eng Output parameter to write the computed pair energy.
        \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
       cutoff.
       \param torque_i The torque exterted on the i^th particle.
       \param torque_j The torque
       exterted on the j^th particle. \return True if they are evaluated or false if they are not
       because we are beyond the cutoff.
    */
    HOSTDEVICE bool evaluate(Scalar3& force,
                             Scalar& pair_eng,
                             bool energy_shift,
                             Scalar3& torque_i,
                             Scalar3& torque_j)
        {
        Scalar rsq = dot(dr, dr);

        if (rsq < rcutsq)
            {
            Scalar r = sqrt(rsq);

            // Since we will be performing a rotation many times, it's worthwhile to
            // convert the quaternions to rotation matrices and use those for repeated
            // rotations. The conversions below use the fewest operations I could come
            // up with (12 multiplications and 12 additions). Optimizing this
            // conversion is only really important for low vertex shapes where the
            // additional cost of the conversion could offset the added speed of the
            // rotations. We create local scope for all the intermediate products to
            // avoid namespace pollution with unnecessary variables..
            Scalar mati[3][3], matj[3][3];
            quat2mat(qi, mati);
            quat2mat(qj, matj);

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
                const ManagedArray<vec3<Scalar>>& verts1(flip ? shape_j->verts : shape_i->verts);
                const ManagedArray<vec3<Scalar>>& verts2(flip ? shape_i->verts : shape_j->verts);
                const Scalar(&mat1)[3][3](flip ? matj : mati), (&mat2)[3][3](flip ? mati : matj);
                const quat<Scalar>&q1(flip ? qj : qi), &q2(flip ? qi : qj);
                vec3<Scalar> dr_use(flip ? -dr : dr);

                bool success, overlap;
                // Note the signs of each of the vectors:
                //    - v points from the contact point on verts2 to the contact points on verts1.
                //    - a points from the centroid of verts1 to the contact points on verts1.
                //    - b points from the centroid of verts1 to the contact points on verts2.
                gjk<ndim>(verts1,
                          verts2,
                          v,
                          a,
                          b,
                          success,
                          overlap,
                          mat1,
                          mat2,
                          q1,
                          q2,
                          dr_use,
                          shape_i->rounding_radii,
                          shape_j->rounding_radii,
                          shape_i->has_rounding,
                          shape_j->has_rounding);
                // Unphysical ALJ simulation results may be the result of
                // invalid collision detection from GJK, which will normally
                // occur silently. This assertion helps debug such errors by
                // failing fast in debug mode if GJK failed to converge.
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

            Scalar sigma12 = Scalar(0.5) * (_params.sigma_i + _params.sigma_j);
            Scalar sigma12_sq = sigma12 * sigma12;

            Scalar contact_sphere_diameter
                = Scalar(0.5) * (_params.contact_sigma_i + _params.contact_sigma_j);
            Scalar contact_sphere_diameter_sq = contact_sphere_diameter * contact_sphere_diameter;

            // The energy for the central potential must be rescaled by the
            // orientations (encoded in the a and b vectors).
            Scalar k1 = sqrt(dot(a, a));
            Scalar k2 = sqrt(dot(dr + b, dr + b));
            Scalar rho = sigma12
                         / (r - (k1 - Scalar(0.5) * _params.sigma_i)
                            - (k2 - Scalar(0.5) * _params.sigma_j));
            Scalar invr_rsq = rho * rho;
            Scalar invr_6 = invr_rsq * invr_rsq * invr_rsq;
            Scalar numer
                = (invr_6 * invr_6 - invr_6) - (_params.alpha % 2 == 0 ? SHIFT_RHO_DIFF : 0);

            invr_rsq = sigma12_sq / rsq;
            invr_6 = invr_rsq * invr_rsq * invr_rsq;
            Scalar invr_12 = invr_6 * invr_6;
            Scalar denom = invr_12 - invr_6 - (_params.alpha % 2 == 0 ? SHIFT_RHO_DIFF : 0);
            Scalar scale_factor = denom != 0 ? (numer / denom) : 1;

            Scalar four_epsilon = Scalar(4.0) * _params.epsilon;
            Scalar four_scaled_epsilon = four_epsilon;
            if (_params.alpha % 2 == 0)
                {
                four_scaled_epsilon = four_epsilon * scale_factor;
                }

            // We must compute the central LJ force if we are including the
            // attractive component. For pure repulsive (WCA), we only need
            // to compute it if we are within the limited cutoff.
            Scalar f_scalar = 0;
            pair_eng = 0;
            if ((_params.alpha % 2 != 0) || (rsq < TWO_P_13 * sigma12_sq))
                {
                // Compute force and energy from LJ formula.
                pair_eng += four_scaled_epsilon * (invr_12 - invr_6);
                f_scalar
                    = four_scaled_epsilon * (Scalar(12.0) * invr_12 - Scalar(6.0) * invr_6) / r;

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
            // having multiple evaluators to handle the specific pairs of
            // interactions. However, that requires significant reworkings
            // of HOOMD internals, so it's probably not worthwhile. We
            // should consider if we ever want to enable this during the
            // HOOMD3.0 rewrite, though.

            // TODO: Can prefilter whether we need to call the function
            // based on whether the minimum distance is within rcut,
            // otherwise we know that none of the interactions will count.
            if (_params.average_simplices && shape_i->verts.size() > ndim
                && shape_j->verts.size() > ndim)
                {
                computeSimplexInteractions(a,
                                           b,
                                           dr,
                                           mati,
                                           matj,
                                           contact_sphere_diameter_sq,
                                           four_epsilon,
                                           f,
                                           pair_eng,
                                           torque_i,
                                           torque_j);
                }
            else
                {
                vec3<Scalar> torquei, torquej;
                computeContactLJ(contact_sphere_diameter_sq,
                                 v,
                                 four_epsilon,
                                 a,
                                 dr + b,
                                 f,
                                 pair_eng,
                                 torquei,
                                 torquej);
                torque_i = vec_to_scalar3(torquei);
                torque_j = vec_to_scalar3(torquej);
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

#ifndef __HIPCC__
    //! Get the name of the potential
    /*! \returns The potential name. Must be short and all lowercase, as this is the name energies
       will be logged as via analyze.log.
    */
    static std::string getName()
        {
        return "alj";
        }
    static std::string getShapeParamName()
        {
        return "Shape";
        }
    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    //! Calculate contact interaction between two simplices.
    /*! This method takes two faces of polytopes and computes all requisite
     * pairwise interactions between them. It must be specialized for each
     * necessary dimensionality because the minimal set of interactions
     * that must be computed depend on the set of lower-dimensional
     * simplices that make up an n-dimensional face. The function is given
     * an empty definition by default.
     *
     *  \param support_vectors1 Set of vectors composing the simplex on particle i.
     *  \param support_vectors2 Set of vectors composing the simplex on particle j.
     *  \param contact_sphere_diameter_sq The length scale of the LJ interaction (i.e. sigma^2).
     *  \param four_epsilon 4*epsilon, the energy scale for the LJ interaction.
     *  \param contact_point_i The contact point on particle i in the frame of particle i.
     *  \param contact_point_j The contact point on particle j in the frame of particle j (IMPORTANT
     * NOTE, this is different from many places in this code where vectors relating to particle j
     * are in the coordinate system centered on particle i).
     * \param force The current force vector to which the force computed in this method is added.
     * \param pair_eng The current pair interaction energy to which the energy computed in this
     * method is added.
     * \param torque_i The current torque on particle i to which the torque computed in this method
     * is added.
     * \param torque_j The current torque on particle j to which the torque computed in this method
     * is added.
     */
    HOSTDEVICE inline void computeSimplexInteractions(const vec3<Scalar>& a,
                                                      const vec3<Scalar>& b,
                                                      const vec3<Scalar>& dr,
                                                      const Scalar (&mati)[3][3],
                                                      const Scalar (&matj)[3][3],
                                                      const Scalar contact_sphere_diameter_sq,
                                                      const Scalar& four_epsilon,
                                                      vec3<Scalar>& force,
                                                      Scalar& pair_eng,
                                                      Scalar3& torque_i,
                                                      Scalar3& torque_j)
        {
        }

    //! Core routine for calculating contact interaction between two points.
    /*! This method is defined separately because it must be called for
     * both the base contact-point formalism and the extended simplex
     * averaging method we use for polytopes with many parallel faces. In
     * the latter case, we may have to compute a large number of
     * interactions, and each interaction will call this method.
     * The contact points must be with respect to each particle's center of
     * mass (contact_point_i is relative to the origin, which is the center
     * of verts1, whereas contact_point_j is relative to the center of
     * verts2, which is -dr). The contact_vector must point from verts1 to verts2
     *
     *  \param contact_sphere_diameter_sq The length scale of the LJ interaction (i.e. sigma^2).
     *  \param contact_vector The vector pointing from the contact point on particle i to the
     * contact point on particle j.
     * \param contact_distance The length of the contact_vector provided separately since it's
     * typically computed as part of determining the contact_vector.
     *  \param four_epsilon 4*epsilon, the energy scale for the LJ interaction.
     *  \param contact_point_i The contact point on particle i in the frame of particle i.
     *  \param contact_point_j The contact point on particle j in the frame of particle j (IMPORTANT
     * NOTE, this is different from many places in this code where vectors relating to particle j
     * are in the coordinate system centered on particle i).
     * \param force The current force vector to which the force computed in this method is added.
     * \param pair_eng The current pair interaction energy to which the energy computed in this
     * method is added.
     * \param torque_i The current torque on particle i to which the torque computed in this method
     * is added.
     * \param torque_j The current torque on particle j to which the torque computed in this method
     * is added.
     */
    HOSTDEVICE inline void computeContactLJ(const Scalar& contact_sphere_diameter_sq,
                                            const vec3<Scalar>& contact_vector,
                                            const Scalar& four_epsilon,
                                            const vec3<Scalar>& contact_point_i,
                                            const vec3<Scalar>& contact_point_j,
                                            vec3<Scalar>& force,
                                            Scalar& pair_eng,
                                            vec3<Scalar>& torque_i,
                                            vec3<Scalar>& torque_j)
        {
        Scalar contact_distance_sq = dot(contact_vector, contact_vector);
        if ((_params.alpha / 2 != 0)
            || (contact_distance_sq < TWO_P_13 * contact_sphere_diameter_sq))
            {
            Scalar contact_distance = sqrt(contact_distance_sq);
            Scalar invr_rsq = contact_sphere_diameter_sq / contact_distance_sq;
            Scalar invr_6 = invr_rsq * invr_rsq * invr_rsq;
            Scalar invr_12 = invr_6 * invr_6;
            Scalar energy_contact = four_epsilon * (invr_12 - invr_6);
            Scalar scalar_force_contact
                = four_epsilon * (Scalar(12.0) * invr_12 - Scalar(6.0) * invr_6) / contact_distance;

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

    vec3<Scalar> dr;           //!< Stored dr from the constructor
    Scalar rcutsq;             //!< Stored rcutsq from the constructor
    quat<Scalar> qi;           //!< Orientation quaternion for particle i
    quat<Scalar> qj;           //!< Orientation quaternion for particle j
    unsigned int tag_i;        //!< Tag of particle i.
    unsigned int tag_j;        //!< Tag of particle j.
    const shape_type* shape_i; //!< Shape parameters of particle i.
    const shape_type* shape_j; //!< Shape parameters of particle j.
    const param_type& _params; //!< Potential parameters for the pair of interest.

    constexpr static Scalar TWO_P_13 = 1.2599210498948732; // 2^(1/3)
    constexpr static Scalar SHIFT_RHO_DIFF = -0.25;        // (1/(2^(1/6)))**12 - (1/(2^(1/6)))**6
    };

//! Compute the distance between a point and the face of a polytope.
/*! This function is adapted from DEMEvaluator::vertexFace. The energy
 * evaluation has been removed from this function, so this function only
 * computes the distance and leaves the energy evaluation to a separate
 * routine. The internals of this method are adapted from DEM to use the data
 * structures of the ALJ potential, namely the EvaluatorPairALJ::shape_param instead of the
 * various arrays stored in the DEM3DForceCompute class.
 *
 * Note that all calculations in this method are done in particle j's reference
 * frame, so r0 is immediately shifted to r0j and used accordingly.
 *
 * \param rij The vector from the particle containing r0 to the particle whose face is being
 * evaluated.
 * \param shape_j The shape parameters of shape j.
 * \param r0 The vertex on the first particle.
 * \param matj The orientation of the second particle.
 * \param face_j The index into shape_j->face_offsets of the current face on the second particle.
 * \param delta The vector pointing from the closest point on the face of particle j to the input
 * point r0 (updated by reference).
 * \param projection The closest point on the face to r0, IN THE FRAME OF PARTICLE j
 * (updated by reference).
 */
HOSTDEVICE inline void pointFaceDistance(const vec3<Scalar>& rij,
                                         const EvaluatorPairALJ<3>::shape_type* shape_j,
                                         const vec3<Scalar>& r0,
                                         const Scalar (&matj)[3][3],
                                         unsigned int face_j,
                                         vec3<Scalar>& delta,
                                         vec3<Scalar>& projection)
    {
    // distsq will be used to hold the square distance from r0 to the
    // face of interest; work relative to particle j's center of mass
    Scalar distsq(0);
    // r0 from particle j's frame of reference
    const vec3<Scalar> r0j(r0 - rij);
    // rPrime is the closest point to r0
    vec3<Scalar> rPrime;

    // vertex0 is the reference point in particle j to "fan out" from
    unsigned int start_idx = shape_j->face_offsets[face_j];
    unsigned int end_idx = (face_j == shape_j->face_offsets.size() - 1)
                               ? shape_j->faces.size()
                               : shape_j->face_offsets[face_j + 1];
    const vec3<Scalar> vertex0(rotate(matj, shape_j->verts[shape_j->faces[start_idx]]));

    // r0r0: vector from vertex0 to r0 relative to particle j
    const vec3<Scalar> r0r0(r0j - vertex0);

    // check distance for first edge of polygon
    const vec3<Scalar> secondVertex(rotate(matj, shape_j->verts[shape_j->faces[start_idx + 1]]));
    const vec3<Scalar> rsec(secondVertex - vertex0);
    Scalar lambda(dot(r0r0, rsec) / dot(rsec, rsec));
    lambda = clip(lambda);
    vec3<Scalar> closest(vertex0 + lambda * rsec);
    vec3<Scalar> closestr0(closest - r0j);
    Scalar closestDistsq(dot(closestr0, closestr0));
    distsq = closestDistsq;
    rPrime = closest;

    // indices of three points in triangle of interest: vertex0Index, i, facesj[i]
    // p01 and p02: two edge vectors of the triangle of interest
    vec3<Scalar> p1, p2(secondVertex), p01, p02(secondVertex - vertex0);

    // iterate through all fan triangles
    for (unsigned int next_idx = start_idx + 2; next_idx < end_idx; ++next_idx)
        {
        Scalar alpha(0), beta(0);

        p1 = p2;
        p2 = rotate(matj, shape_j->verts[shape_j->faces[next_idx]]);
        p01 = p02;
        p02 = p2 - vertex0;

        // pc: vector normal to the triangle of interest
        const vec3<Scalar> pc(cross(p01, p02));

        // distance matrix A is:
        // [ p01.x p02.x pc.x ]
        // [ p01.y p02.y pc.y ]
        // [ p01.z p02.z pc.z ]
        Scalar magA(p01.x * (p02.y * pc.z - pc.y * p02.z) - p02.x * (p01.y * pc.z - pc.y * p01.z)
                    + pc.x * (p01.y * p02.z - p02.y * p01.z));

        alpha = ((p02.y * pc.z - pc.y * p02.z) * r0r0.x + (pc.x * p02.z - p02.x * pc.z) * r0r0.y
                 + (p02.x * pc.y - pc.x * p02.y) * r0r0.z)
                / magA;
        beta = ((pc.y * p01.z - p01.y * pc.z) * r0r0.x + (p01.x * pc.z - pc.x * p01.z) * r0r0.y
                + (pc.x * p01.y - p01.x * pc.y) * r0r0.z)
               / magA;

        alpha = clip(alpha);
        beta = clip(beta);
        const Scalar k(alpha + beta);

        if (k > 1)
            {
            alpha /= k;
            beta /= k;
            }

        // check distance for exterior edge of polygon
        const vec3<Scalar> p12(p2 - p1);
        Scalar lambda(dot(r0j - p1, p12) / dot(p12, p12));
        lambda = clip(lambda);
        vec3<Scalar> closest(p1 + lambda * p12);
        vec3<Scalar> closestr0(closest - r0j);
        Scalar closestDistsq(dot(closestr0, closestr0));
        if (closestDistsq < distsq)
            {
            distsq = closestDistsq;
            rPrime = closest;
            }

        // closest: closest point in triangle (in particle j's reference frame)
        closest = vertex0 + alpha * p01 + beta * p02;
        // closestr0: vector between r0 and closest
        closestr0 = closest - r0j;
        closestDistsq = dot(closestr0, closestr0);
        if (closestDistsq < distsq)
            {
            distsq = closestDistsq;
            rPrime = closest;
            }
        }

    // check distance for last edge of polygon
    const vec3<Scalar> rlast(p2 - vertex0);
    lambda = dot(r0r0, rlast) / dot(rlast, rlast);
    lambda = clip(lambda);
    closest = vertex0 + lambda * rlast;
    closestr0 = closest - r0j;
    closestDistsq = dot(closestr0, closestr0);
    if (closestDistsq < distsq)
        {
        distsq = closestDistsq;
        rPrime = closest;
        }

    projection = rPrime;
    delta = r0j - projection;
    }

//! Compute the distance between two line segments in 3D.
/*! This function is adapted from DEMEvaluator::edgeEdge. Like
 * pointFaceDistance, the energy evaluation has been removed from this
 * function. The only other change is that the first few lines of
 * DEMEvaluator::edgeEdge do an unnecessary number of repeated computations
 * through calling the detp function defined there, but most of these dot
 * products are identical and can be reduced as in the first 10 lines of this
 * function. All inputs must be provided in the same coordinate system.
 *
 * \param e00 The first vertex of the first line segment.
 * \param e01 The second vertex of the first line segment.
 * \param e10 The first vertex of the second line segment.
 * \param e11 The second vertex of the second line segment.
 * \param closestI The closest point to the second segment on the first segment (overwritten by
 * reference).
 * \param closestI The closest point to the first segment on the second segment
 * (overwritten by reference).
 */
HOSTDEVICE inline void edgeEdgeDistance(const vec3<Scalar>& e00,
                                        const vec3<Scalar>& e01,
                                        const vec3<Scalar>& e10,
                                        const vec3<Scalar>& e11,
                                        vec3<Scalar>& closestI,
                                        vec3<Scalar>& closestJ)
    {
    // This math is identical to DEM's but is simplified to reduce the number
    // of dot products and clarify the purpose of the calculations that are
    // present.
    // in the style of http://paulbourke.net/geometry/pointlineplane/
    const vec3<Scalar> r0(e01 - e00);
    const vec3<Scalar> r1(e11 - e10);
    const Scalar r0sq(dot(r0, r0));
    const Scalar r1sq(dot(r1, r1));
    const Scalar r1r0(dot(r1, r0));

    const vec3<Scalar> diff0010(e00 - e10);
    const Scalar detp5(dot(diff0010, r1));
    const Scalar detp7(dot(diff0010, r0));

    Scalar r0sqr1sq = r0sq * r1sq;
    Scalar r1r0r1r0 = r1r0 * r1r0;
    const Scalar denominator(r0sqr1sq - r1r0r1r0);
    Scalar lambda0((detp5 * r1r0 - detp7 * r1sq) / denominator);
    Scalar lambda1((detp5 + lambda0 * r1r0) / r1sq);

    lambda0 = clip(lambda0);
    lambda1 = clip(lambda1);

    closestI = e00 + lambda0 * r0;
    closestJ = e10 + lambda1 * r1;
    vec3<Scalar> rContact(closestJ - closestI);
    Scalar closestDistsq = dot(rContact, rContact);

    Scalar lambda(clip(dot(e10 - e00, r0) / r0sq));
    vec3<Scalar> candidateI(e00 + lambda * r0);
    rContact = e10 - candidateI;
    Scalar distsq(dot(rContact, rContact));
    if (distsq < closestDistsq)
        {
        closestI = candidateI;
        closestJ = e10;
        closestDistsq = distsq;
        }

    lambda = clip(dot(e11 - e00, r0) / r0sq);
    candidateI = e00 + lambda * r0;
    rContact = e11 - candidateI;
    distsq = dot(rContact, rContact);
    if (distsq < closestDistsq)
        {
        closestI = candidateI;
        closestJ = e11;
        closestDistsq = distsq;
        }

    lambda = clip(dot(diff0010, r1) / r1sq);
    vec3<Scalar> candidateJ = e10 + lambda * r1;
    rContact = candidateJ - e00;
    distsq = dot(rContact, rContact);
    if (distsq < closestDistsq)
        {
        closestI = e00;
        closestJ = candidateJ;
        closestDistsq = distsq;
        }

    lambda = clip(dot(e01 - e10, r1) / r1sq);
    candidateJ = e10 + lambda * r1;
    rContact = candidateJ - e01;
    distsq = dot(rContact, rContact);
    if (distsq < closestDistsq)
        {
        closestI = e01;
        closestJ = candidateJ;
        closestDistsq = distsq;
        }

    if (fabs(1 - r1r0r1r0 / r0sqr1sq) < 1e-6)
        {
        const Scalar lambda00(clip(dot(e10 - e00, r0) / r0sq));
        const Scalar lambda01(clip(dot(e11 - e00, r0) / r0sq));
        const Scalar lambda10(clip(dot(e00 - e10, r1) / r1sq));
        const Scalar lambda11(clip(dot(e01 - e10, r1) / r1sq));

        lambda0 = Scalar(.5) * (lambda00 + lambda01);
        lambda1 = Scalar(.5) * (lambda10 + lambda11);

        closestI = e00 + lambda0 * r0;
        closestJ = e10 + lambda1 * r1;
        }
    }

//! Find the simplex defined by verts that is closest to the provided vector.
/*! The closest simplex in this case is defined as the set of ndim points that
 * are a subset of a face of the polytope defined by verts that also form the
 * minimal set of angles with the provided vector. These angles are computed
 * using dot products of the rotated vertices with the vector.
 *
 * \param verts The vertices of the shape.
 * \param vector The point outside the shape, given in the body frame of the verts.
 * \param mat The orientation of the verts.
 * \param unique_vectors Array of vec3 to which the rotated vectors on the simplex are written
 * (overwritten by references).
 */
template<unsigned int ndim>
HOSTDEVICE inline void findSimplex(const vec3<Scalar>& vector,
                                   const Scalar (&mat)[3][3],
                                   vec3<Scalar> (&unique_vectors)[ndim],
                                   const typename EvaluatorPairALJ<ndim>::shape_type* shape)
    {
    if (ndim == 3)
        {
        // Declare arrays and variables.
        Scalar mat_inverse[3][3];
        vec3<Scalar> norm_vector = vector / sqrt(dot(vector, vector));

        vec3<Scalar> rot_vector;
        vec3<Scalar> plane_normal_vector;

        // Scalar norm_of_rot_vector;
        Scalar rot_vec_scale;
        Scalar rot_vec_scale_prev;
        Scalar rot_vec_proj_to_normal_vector;
        Scalar angle;
        Scalar angles[ndim];
        for (unsigned int i = 0; i < ndim; ++i)
            angles[i] = Scalar(M_PI);

        unsigned int near_face_idx;

        // Prepare inverse of rotation matrix, "mat_inverse".
        // Inverse of rotation matrix is same as transpose of the rotation matrix.
        for (unsigned int i = 0; i < 3; ++i)
            {
            for (unsigned int j = 0; j < 3; ++j)
                {
                mat_inverse[i][j] = mat[j][i];
                }
            }

        // Prepare a vector prepared by rotating "vector" with a matrix "mat_inverse" (passive
        // rotation). This vector will be denoted as "rot_vector".
        rot_vector = rotate(mat_inverse, vector);
        // norm_of_rot_vector = sqrt(dot(rot_vector, rot_vector)); // norm of the "rot_vector";
        /// Go through each face and check "rot_vec_scale".
        /// Here, "rot_vec_scale" * "rot_vector corresponds" to the intersection point between a
        /// plane defined by a "face" and a line defined by "rot_vector". We would like to find the
        /// face where "rot_vec_scale" is positive and is minimum.

        unsigned int num_faces = shape->face_offsets.size();

        // First check the face 0.
        near_face_idx = 0;
        plane_normal_vector = cross(shape->verts[shape->faces[shape->face_offsets[0]]]
                                        - shape->verts[shape->faces[shape->face_offsets[0] + 1]],
                                    shape->verts[shape->faces[shape->face_offsets[0]]]
                                        - shape->verts[shape->faces[shape->face_offsets[0] + 2]]);
        rot_vec_proj_to_normal_vector = dot(plane_normal_vector, rot_vector);
        rot_vec_scale = dot(plane_normal_vector, shape->verts[shape->faces[shape->face_offsets[0]]])
                        / rot_vec_proj_to_normal_vector;
        rot_vec_scale_prev = rot_vec_scale;

        // Now go through each face using for loop.

        for (unsigned int i = 1; i < num_faces; ++i)
            {
            plane_normal_vector
                = cross(shape->verts[shape->faces[shape->face_offsets[i]]]
                            - shape->verts[shape->faces[shape->face_offsets[i] + 1]],
                        shape->verts[shape->faces[shape->face_offsets[i]]]
                            - shape->verts[shape->faces[shape->face_offsets[i] + 2]]);
            rot_vec_proj_to_normal_vector = dot(plane_normal_vector, rot_vector);
            rot_vec_scale
                = dot(plane_normal_vector, shape->verts[shape->faces[shape->face_offsets[i]]])
                  / rot_vec_proj_to_normal_vector;

            if (rot_vec_scale > 0.0)
                {
                if (rot_vec_scale_prev > 0.0)
                    {
                    if (rot_vec_scale < rot_vec_scale_prev)
                        {
                        near_face_idx = i;
                        rot_vec_scale_prev = rot_vec_scale;
                        }
                    }
                else
                    {
                    near_face_idx = i;
                    rot_vec_scale_prev = rot_vec_scale;
                    }
                }
            }
        // Now find the simplex.
        unsigned int last_idx = (near_face_idx == num_faces - 1)
                                    ? shape->faces.size()
                                    : shape->face_offsets[near_face_idx + 1];
        unsigned int i = shape->face_offsets[near_face_idx];

        vec3<Scalar> rot_shift_vec = rotate(mat, shape->verts[shape->faces[i]]);
        vec3<Scalar> norm_rot_shift_vector
            = rot_shift_vec / sqrt(dot(rot_shift_vec, rot_shift_vec));
        unique_vectors[0] = rot_shift_vec;
        angles[0] = acos(dot(norm_rot_shift_vector, norm_vector));

        for (unsigned int i = shape->face_offsets[near_face_idx] + 1; i < last_idx; ++i)
            {
            rot_shift_vec = rotate(mat, shape->verts[shape->faces[i]]);
            norm_rot_shift_vector = rot_shift_vec / sqrt(dot(rot_shift_vec, rot_shift_vec));
            angle = acos(dot(norm_rot_shift_vector, norm_vector));

            unsigned int insertion_index = (angle < angles[0] ? 0 : (angle < angles[1] ? 1 : 2));
            if (ndim == 3 && angle >= angles[2])
                {
                insertion_index = 3;
                }

            if (insertion_index < ndim)
                {
                for (unsigned int j = (ndim - 1); j > insertion_index; --j)
                    {
                    unique_vectors[j] = unique_vectors[j - 1];
                    angles[j] = angles[j - 1];
                    }
                unique_vectors[insertion_index] = rot_shift_vec;
                angles[insertion_index] = angle;
                }
            }
        }

    else
        {
        Scalar angles[ndim];
        for (unsigned int i = 0; i < ndim; ++i)
            angles[i] = Scalar(M_PI);

        vec3<Scalar> norm_vector = vector / sqrt(dot(vector, vector));

        vec3<Scalar> rot_shift_vec = rotate(mat, shape->verts[0]);
        vec3<Scalar> norm_rot_shift_vector
            = rot_shift_vec / sqrt(dot(rot_shift_vec, rot_shift_vec));
        unique_vectors[0] = rot_shift_vec;
        angles[0] = acos(dot(norm_rot_shift_vector, norm_vector));

        for (unsigned int i = 1; i < shape->verts.size(); ++i)
            {
            rot_shift_vec = rotate(mat, shape->verts[i]);
            norm_rot_shift_vector = rot_shift_vec / sqrt(dot(rot_shift_vec, rot_shift_vec));
            Scalar angle = acos(dot(norm_rot_shift_vector, norm_vector));

            unsigned int insertion_index = (angle < angles[0] ? 0 : (angle < angles[1] ? 1 : 2));
            if (ndim == 3 && angle >= angles[2])
                {
                insertion_index = 3;
                }

            if (insertion_index < ndim)
                {
                for (unsigned int j = (ndim - 1); j > insertion_index; --j)
                    {
                    unique_vectors[j] = unique_vectors[j - 1];
                    angles[j] = angles[j - 1];
                    }
                unique_vectors[insertion_index] = rot_shift_vec;
                angles[insertion_index] = angle;
                }
            }
        }
    }

//! Find the face defined by verts that is closest to the provided vector.
/*! This function is very similar to findSimplex (see the documentation to that function above),
 * except that this function is designed to work in three dimensions where the
 * closest simplex may not be enough because a single face may be composed of
 * multiple simplices (where a simplex is defined in the sense of a convex
 * hull, i.e. it always contains exactly ndim points). For instance, a face of
 * a cube is composed of two triangular simplices. The first part of this
 * function is identical to findSimplex, except that instead of storing the
 * vectors found only the corresponding indices are stored. Then, a search
 * through all the faces of the shape is performed to identify the face
 * containing those three vertices; that face must be unique, and is the face
 * that will be used.
 *
 * \param vector The point outside the shape, given in the body frame of the verts.
 * \param mat The orientation of the verts.
 * \param face The face on the shape closest to vector (overwritten by reference).
 * \param shape The shape parameters of the particle being tested.
 */
template<unsigned int ndim>
HOSTDEVICE inline void findFace(const vec3<Scalar>& vector,
                                const Scalar (&mat)[3][3],
                                unsigned int& face,
                                const typename EvaluatorPairALJ<ndim>::shape_type* shape)
    {
    if (ndim == 3)
        {
        // Declare arrays and variables.
        Scalar mat_inverse[3][3];
        vec3<Scalar> rot_vector;
        vec3<Scalar> plane_normal_vector;

        // Scalar norm_of_rot_vector;
        Scalar rot_vec_scale;
        Scalar rot_vec_scale_prev;
        Scalar rot_vec_proj_to_normal_vector;
        unsigned int near_face_idx;

        // Prepare inverse of rotation matrix, "mat_inverse".
        // Inverse of rotation matrix is same as transpose of the rotation matrix.
        for (unsigned int i = 0; i < 3; ++i)
            {
            for (unsigned int j = 0; j < 3; ++j)
                {
                mat_inverse[i][j] = mat[j][i];
                }
            }

        // Prepare a vector prepared by rotating "vector" with a matrix "mat_inverse" (passive
        // rotation). This vector will be denoted as "rot_vector".
        rot_vector = rotate(mat_inverse, vector);
        // norm_of_rot_vector = sqrt(dot(rot_vector, rot_vector)); // norm of the "rot_vector";

        /// Go through each face and check "rot_vec_scale".
        /// Here, "rot_vec_scale" * "rot_vector corresponds" to the intersection point between a
        /// plane defined by a "face" and a line defined by "rot_vector". We would like to find the
        /// face where "rot_vec_scale" is positive and is minimum.

        unsigned int num_faces = shape->face_offsets.size();

        // First check the face 0.
        near_face_idx = 0;
        plane_normal_vector = cross(shape->verts[shape->faces[shape->face_offsets[0]]]
                                        - shape->verts[shape->faces[shape->face_offsets[0] + 1]],
                                    shape->verts[shape->faces[shape->face_offsets[0]]]
                                        - shape->verts[shape->faces[shape->face_offsets[0] + 2]]);
        rot_vec_proj_to_normal_vector = dot(plane_normal_vector, rot_vector);
        rot_vec_scale = dot(plane_normal_vector, shape->verts[shape->faces[shape->face_offsets[0]]])
                        / rot_vec_proj_to_normal_vector;
        rot_vec_scale_prev = rot_vec_scale;

        // Now go through each face using for loop.

        for (unsigned int i = 1; i < num_faces; ++i)
            {
            plane_normal_vector
                = cross(shape->verts[shape->faces[shape->face_offsets[i]]]
                            - shape->verts[shape->faces[shape->face_offsets[i] + 1]],
                        shape->verts[shape->faces[shape->face_offsets[i]]]
                            - shape->verts[shape->faces[shape->face_offsets[i] + 2]]);
            rot_vec_proj_to_normal_vector = dot(plane_normal_vector, rot_vector);
            rot_vec_scale
                = dot(plane_normal_vector, shape->verts[shape->faces[shape->face_offsets[i]]])
                  / rot_vec_proj_to_normal_vector;

            if (rot_vec_scale > 0.0)
                {
                if (rot_vec_scale_prev > 0.0)
                    {
                    if (rot_vec_scale < rot_vec_scale_prev)
                        {
                        near_face_idx = i;
                        rot_vec_scale_prev = rot_vec_scale;
                        }
                    }
                else
                    {
                    near_face_idx = i;
                    rot_vec_scale_prev = rot_vec_scale;
                    }
                }
            }
        face = near_face_idx;
        }

    else
        {
        // First identify the closest points
        unsigned int unique_vectors[ndim];
        Scalar angles[ndim];
        for (unsigned int i = 0; i < ndim; ++i)
            angles[i] = Scalar(M_PI);

        vec3<Scalar> norm_vector = vector / sqrt(dot(vector, vector));

        vec3<Scalar> rot_shift_vec = rotate(mat, shape->verts[0]);
        vec3<Scalar> norm_rot_shift_vector
            = rot_shift_vec / sqrt(dot(rot_shift_vec, rot_shift_vec));
        unique_vectors[0] = 0;
        angles[0] = acos(dot(norm_rot_shift_vector, norm_vector));

        for (unsigned int i = 1; i < shape->verts.size(); ++i)
            {
            rot_shift_vec = rotate(mat, shape->verts[i]);
            norm_rot_shift_vector = rot_shift_vec / sqrt(dot(rot_shift_vec, rot_shift_vec));
            Scalar angle = acos(dot(norm_rot_shift_vector, norm_vector));

            unsigned int insertion_index = (angle < angles[0] ? 0 : (angle < angles[1] ? 1 : 2));
            if (ndim == 3 && angle >= angles[2])
                {
                insertion_index = 3;
                }

            if (insertion_index < ndim)
                {
                for (unsigned int j = (ndim - 1); j > insertion_index; --j)
                    {
                    unique_vectors[j] = unique_vectors[j - 1];
                    angles[j] = angles[j - 1];
                    }
                unique_vectors[insertion_index] = i;
                angles[insertion_index] = angle;
                }
            }

        // Now loop over all faces and find the one that has the three closest points.
        unsigned int num_faces = shape->face_offsets.size();
        for (face = 0; face < num_faces; ++face)
            {
            unsigned int last_idx
                = (face == num_faces - 1) ? shape->faces.size() : shape->face_offsets[face + 1];
            bool found0 = false, found1 = false, found2 = false;
            for (unsigned int j = shape->face_offsets[face]; j < last_idx; ++j)
                {
                if (shape->faces[j] == unique_vectors[0])
                    found0 = true;
                if (shape->faces[j] == unique_vectors[1])
                    found1 = true;
                if (shape->faces[j] == unique_vectors[2])
                    found2 = true;
                }
            if (found0 && found1 && found2)
                break;
            }
        }
    }

template<>
HOSTDEVICE inline void
EvaluatorPairALJ<2>::computeSimplexInteractions(const vec3<Scalar>& a,
                                                const vec3<Scalar>& b,
                                                const vec3<Scalar>& dr,
                                                const Scalar (&mati)[3][3],
                                                const Scalar (&matj)[3][3],
                                                const Scalar contact_sphere_diameter_sq,
                                                const Scalar& four_epsilon,
                                                vec3<Scalar>& force,
                                                Scalar& pair_eng,
                                                Scalar3& torque_i,
                                                Scalar3& torque_j)
    {
    vec3<Scalar> support_vectors1[2], support_vectors2[2];
    findSimplex(b, mati, support_vectors1, shape_i);
    findSimplex(dr + a, matj, support_vectors2, shape_j);
    // Since simplex finding is based on angles, we need to compute dot
    // products in the local frame of the second particle and then shift
    // the final result back into the global frame (the body frame of
    // particle 1).
    for (unsigned int i = 0; i < 2; ++i)
        {
        support_vectors2[i] += Scalar(-1.0) * dr;
        }

    vec3<Scalar> torquei, torquej;
    vec3<Scalar> projection, vec;
    // First compute the interaction of the verts on 1 to the edge on 2.
    for (unsigned int i = 0; i < 2; ++i)
        {
        pointSegmentDistance(support_vectors1[i],
                             support_vectors2[0],
                             support_vectors2[1],
                             vec,
                             projection);
        computeContactLJ(contact_sphere_diameter_sq,
                         -vec,
                         four_epsilon,
                         support_vectors1[i],
                         dr + support_vectors1[i] - vec,
                         force,
                         pair_eng,
                         torquei,
                         torquej);
        }

    // Now compute the interaction of the verts on 2 to the edge on 1.
    for (unsigned int i = 0; i < 2; ++i)
        {
        pointSegmentDistance(support_vectors2[i],
                             support_vectors1[0],
                             support_vectors1[1],
                             vec,
                             projection);

        computeContactLJ(contact_sphere_diameter_sq,
                         vec,
                         four_epsilon,
                         projection,
                         dr + support_vectors2[i],
                         force,
                         pair_eng,
                         torquei,
                         torquej);
        }
    torque_i = vec_to_scalar3(torquei);
    torque_j = vec_to_scalar3(torquej);
    }

template<>
HOSTDEVICE inline void
EvaluatorPairALJ<3>::computeSimplexInteractions(const vec3<Scalar>& a,
                                                const vec3<Scalar>& b,
                                                const vec3<Scalar>& dr,
                                                const Scalar (&mati)[3][3],
                                                const Scalar (&matj)[3][3],
                                                const Scalar contact_sphere_diameter_sq,
                                                const Scalar& four_epsilon,
                                                vec3<Scalar>& force,
                                                Scalar& pair_eng,
                                                Scalar3& torque_i,
                                                Scalar3& torque_j)
    {
    unsigned int face_i, face_j;
    findFace<3>(b, mati, face_i, shape_i);
    findFace<3>(dr + a, matj, face_j, shape_j);

    vec3<Scalar> torquei, torquej;
    vec3<Scalar> projection, vec;

    unsigned int start_idx_i = shape_i->face_offsets[face_i];
    unsigned int end_idx_i = (face_i == shape_i->face_offsets.size() - 1)
                                 ? shape_i->faces.size()
                                 : shape_i->face_offsets[face_i + 1];
    const vec3<Scalar> rij(Scalar(-1.0) * dr);
    // Compute the interaction of the verts on 1 to the face on 2.
    for (unsigned int i = start_idx_i; i < end_idx_i; ++i)
        {
        const vec3<Scalar> vertex(rotate(mati, shape_i->verts[shape_i->faces[i]]));
        pointFaceDistance(rij, shape_j, vertex, matj, face_j, vec, projection);

        computeContactLJ(contact_sphere_diameter_sq,
                         -vec,
                         four_epsilon,
                         vertex,
                         dr + vertex - vec,
                         force,
                         pair_eng,
                         torquei,
                         torquej);
        }

    unsigned int start_idx_j = shape_j->face_offsets[face_j];
    unsigned int end_idx_j = (face_j == shape_j->face_offsets.size() - 1)
                                 ? shape_j->faces.size()
                                 : shape_j->face_offsets[face_j + 1];
    // Compute the interaction of the verts on 2 to the edge on 1.
    for (unsigned int i = start_idx_j; i < end_idx_j; ++i)
        {
        const vec3<Scalar> vertex(rotate(matj, shape_j->verts[shape_j->faces[i]]));
        pointFaceDistance(dr, shape_i, vertex, mati, face_i, vec, projection);
        computeContactLJ(contact_sphere_diameter_sq,
                         vec,
                         four_epsilon,
                         projection,
                         dr + vertex,
                         force,
                         pair_eng,
                         torquei,
                         torquej);
        }

    // Compute interaction between pairs of edges.
    for (unsigned int i = start_idx_i; i < end_idx_i; ++i)
        {
        vec3<Scalar> e00(rotate(mati, shape_i->verts[shape_i->faces[i]]));
        unsigned int idx_e01(i == end_idx_i - 1 ? start_idx_i : i + 1);
        vec3<Scalar> e01(rotate(mati, shape_i->verts[shape_i->faces[idx_e01]]));

        for (unsigned int j = start_idx_j; j < end_idx_j; ++j)
            {
            vec3<Scalar> e10(rotate(matj, shape_j->verts[shape_j->faces[j]]) + Scalar(-1.0) * dr);
            unsigned int idx_e11(j == end_idx_j - 1 ? start_idx_j : j + 1);
            vec3<Scalar> e11(rotate(matj, shape_j->verts[shape_j->faces[idx_e11]])
                             + Scalar(-1.0) * dr);

            vec3<Scalar> closestI, closestJ;
            edgeEdgeDistance(e00, e01, e10, e11, closestI, closestJ);
            vec3<Scalar> vec = closestJ - closestI;
            computeContactLJ(contact_sphere_diameter_sq,
                             vec,
                             four_epsilon,
                             closestI,
                             dr + closestJ,
                             force,
                             pair_eng,
                             torquei,
                             torquej);
            }
        }
    torque_i = vec_to_scalar3(torquei);
    torque_j = vec_to_scalar3(torquej);
    }

#ifndef __HIPCC__

// Note: This method assumes that shape_i == shape_j. This should be valid for
// all cases, and this logic will be moved up to the AnisoPotentialPair in
// HOOMD 3.0.
template<> inline std::string EvaluatorPairALJ<2>::getShapeSpec() const
    {
    std::ostringstream shapedef;
    const ManagedArray<vec3<Scalar>>& verts(shape_i->verts); //! Shape vertices.
    const unsigned int N = verts.size();
    if (N == 1)
        {
        shapedef << "{\"type\": \"Ellipsoid\", \"a\": "
                 << shape_i->rounding_radii.x + (_params.contact_sigma_i / 2)
                 << ", \"b\": " << shape_i->rounding_radii.y + (_params.contact_sigma_i / 2) <<
            // Render ellipsoid with a third dimension of 1.
            ", \"c\": " << 1 << "}";
        }
    else
        {
        if (shape_i->rounding_radii.x != shape_i->rounding_radii.y
            || shape_i->rounding_radii.x != shape_i->rounding_radii.z)
            {
            throw std::runtime_error(
                "Shape definition not supported for spheropolygons with distinct rounding radii.");
            }

        shapedef << "{\"type\": \"Polygon\", \"rounding_radius\": "
                 << shape_i->rounding_radii.x + (_params.contact_sigma_i / 2)
                 << ", \"vertices\": [";
        for (unsigned int i = 0; i < N - 1; ++i)
            {
            shapedef << "[" << verts[i].x << ", " << verts[i].y << "], ";
            }
        shapedef << "[" << verts[N - 1].x << ", " << verts[N - 1].y << "]]}";
        }
    return shapedef.str();
    }

// Note: This method assumes that shape_i == shape_j. This should be valid for
// all cases, and this logic will be moved up to the AnisoPotentialPair in
// HOOMD 3.0.
template<> inline std::string EvaluatorPairALJ<3>::getShapeSpec() const
    {
    std::ostringstream shapedef;
    const ManagedArray<vec3<Scalar>>& verts(shape_i->verts);
    const unsigned int N = verts.size();
    if (N == 1)
        {
        shapedef << "{\"type\": \"Ellipsoid\", \"a\": "
                 << shape_i->rounding_radii.x + (_params.contact_sigma_i / 2)
                 << ", \"b\": " << shape_i->rounding_radii.y + (_params.contact_sigma_i / 2)
                 << ", \"c\": " << shape_i->rounding_radii.z + (_params.contact_sigma_i / 2) << "}";
        }
    else
        {
        if (shape_i->rounding_radii.x != shape_i->rounding_radii.y
            || shape_i->rounding_radii.x != shape_i->rounding_radii.z)
            {
            throw std::runtime_error(
                "Shape definition not supported for spheropolyhedra with distinct rounding radii.");
            }
        shapedef << "{\"type\": \"ConvexPolyhedron\", \"rounding_radius\": "
                 << shape_i->rounding_radii.x + (_params.contact_sigma_i / 2)
                 << ", \"vertices\": [";
        for (unsigned int i = 0; i < N - 1; ++i)
            {
            shapedef << "[" << verts[i].x << ", " << verts[i].y << ", " << verts[i].z << "], ";
            }
        shapedef << "[" << verts[N - 1].x << ", " << verts[N - 1].y << ", " << verts[N - 1].z
                 << "]]}";
        }

    return shapedef.str();
    }

#endif

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_PAIR_ALJ_H__
