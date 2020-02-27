#ifndef __GJK_SV_H__
#define __GJK_SV_H__

#include "hoomd/VectorMath.h"
#include "hoomd/ManagedArray.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#include <stdexcept>
#endif


/////////////////////////////////////////////
////////////// BEGIN GJK_VEC3 ///////////////
/////////////////////////////////////////////
// The 2-simplex subroutine of the Signed Volumes subalgorithm requires
// projecting the face of interest onto one of the coordinate axes (choosing
// the one that minimizes precision loss). In practice, this projection is done
// by simply ignoring one of the components of the vectors. Since this is far
// more easily and efficiently accomplished by indexing into the vector (rather
// than creating lots of branch divergence using the .x, .y, and .z attributes
// of normal vec3s), I've defined a new vec3 here. This version is limited to
// only the operations that are actually used in the GJK algorithm.

//! Array-based 3 element vector
/*! \tparam Real Data type of the components

    gjk_vec3 defines a simple 3 element vector. The components are available through the public arr attributed, which can be directly indexed into. The only operations defined on gjk_vec3 objects are the dot product and the cross product. 
*/
template < class Real >
struct gjk_vec3
    {
    //! Construct a gjk_vec3
    /*! \param vec HOOMD vec3
    */
    HOSTDEVICE gjk_vec3(const vec3<Real>& vec)
        {
            arr[0] = vec.x;
            arr[1] = vec.y;
            arr[2] = vec.z;
        }

    //! Construct a gjk_vec3
    /*! \param _x x-component
        \param _y y-component
        \param _z z-component
    */
    HOSTDEVICE gjk_vec3(const Real& _x, const Real& _y, const Real& _z)
        {
            arr[0] = _x;
            arr[1] = _y;
            arr[2] = _z;
        }

    //! Default construct a 0 vector
    HOSTDEVICE gjk_vec3()
        {
            arr[0] = 0;
            arr[1] = 0;
            arr[2] = 0;
        }

    Real arr[3]; //!< Vector components
    };


//! dot product of two gjk_vec3s
/*! \param a First vector
    \param b Second vector

    \returns the dot product a.x*b.x + a.y*b.y + a.z*b.z.
*/
template < class Real >
HOSTDEVICE inline Real dot(const gjk_vec3<Real>& a, const gjk_vec3<Real>& b)
    {
    return (a.arr[0]*b.arr[0] + a.arr[1]*b.arr[1] + a.arr[2]*b.arr[2]);
    }


//! dot product of a gjk_vec3 with a vec3
/*! \param a First vector
    \param b Second vector

    \returns the dot product a.x*b.x + a.y*b.y + a.z*b.z.
*/
template < class Real >
HOSTDEVICE inline Real dot(const gjk_vec3<Real>& a, const vec3<Real>& b)
    {
    return (a.arr[0]*b.x + a.arr[1]*b.y + a.arr[2]*b.z);
    }

template < class Real >
HOSTDEVICE inline Real dot(const vec3<Real>& a, const gjk_vec3<Real>& b)
    {
    return (a.x*b.arr[0] + a.y*b.arr[1] + a.z*b.arr[2]);
    }


//! cross product of two gjk_vec3s
/*! \param a First vector
    \param b Second vector

    \returns the cross product (a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x).
*/
template < class Real >
HOSTDEVICE inline vec3<Real> cross(const gjk_vec3<Real>& a, const gjk_vec3<Real>& b)
    {
    return vec3<Real>(a.arr[1] * b.arr[2] - a.arr[2] * b.arr[1],
                      a.arr[2] * b.arr[0] - a.arr[0] * b.arr[2],
                      a.arr[0] * b.arr[1] - a.arr[1] * b.arr[0]);
    }


//! Subtraction of two gjk_vec3s
/*! \param a First vector
    \param b Second vector

    Subtraction is component wise.
    \returns The vector (a.x-b.x, a.y-b.y, a.z-b.z).
*/
template < class Real >
HOSTDEVICE inline vec3<Real> operator-(const gjk_vec3<Real>& a, const gjk_vec3<Real>& b)
    {
    return vec3<Real>(a.arr[0] - b.arr[0],
                      a.arr[1] - b.arr[1],
                      a.arr[2] - b.arr[2]);
    }


//! Multiplication of a gjk_vec3 by a scalar
/*! \param a vector
    \param b scalar

    Multiplication is component wise.
    \returns The vector (a.x*b, a.y*b, a.z*b).
*/
template < class Real >
HOSTDEVICE inline gjk_vec3<Real> operator*(const Real& b, const gjk_vec3<Real>& a)
    {
    return gjk_vec3<Real>(a.arr[0] * b,
                          a.arr[1] * b,
                          a.arr[2] * b);
    }

//! Assignment-addition of two gjk_vec3s
/*! \param a First vector
    \param b Second vector

    Addition is component wise.
    \returns The vector (a.x += b.x, a.y += b.y, a.z += b.z).
*/
template < class Real >
HOSTDEVICE inline gjk_vec3<Real>& operator +=(gjk_vec3<Real>& a, const gjk_vec3<Real>& b)
    {
    a.arr[0] += b.arr[0];
    a.arr[1] += b.arr[1];
    a.arr[2] += b.arr[2];
    return a;
    }

//! Assignment-addition of a gjk_vec3 to a vec3
/*! \param a First vector
    \param b Second vector

    Addition is component wise.
    \returns The vector (a.x += b.x, a.y += b.y, a.z += b.z).
*/
template < class Real >
HOSTDEVICE inline vec3<Real>& operator +=(vec3<Real>& a, const gjk_vec3<Real>& b)
    {
    a.x += b.arr[0];
    a.y += b.arr[1];
    a.z += b.arr[2];
    return a;
    }

//! Equality test of two gjk_vec3s
/*! \param a First vector
    \param b Second vector
    \returns true if the two vectors are identically equal, false if they are not
*/
template < class Real >
HOSTDEVICE inline bool operator ==(const gjk_vec3<Real>& a, const gjk_vec3<Real>& b)
    {
    return (a.arr[0] == b.arr[0]) && (a.arr[1] == b.arr[1]) && (a.arr[2] == b.arr[2]);
    }

/////////////////////////////////////////////
/////////////// END GJK_VEC3 ///////////////
/////////////////////////////////////////////

HOSTDEVICE inline void support_polyhedron(const ManagedArray<vec3<Scalar> > &verts, const vec3<Scalar> &vector, const quat<Scalar> &q, const vec3<Scalar> shift, unsigned int &idx)
    {
    // Compute the support function of the polyhedron.
    unsigned int index = 0;

    Scalar max_dist_sq = dot((rotate(q, verts[index]) + shift), vector);
    for (unsigned int i = 1; i < verts.size(); ++i)
        {
        Scalar dist_sq = dot((rotate(q, verts[i]) + shift), vector);

        if (dist_sq > max_dist_sq)
            {
            max_dist_sq = dist_sq;
            index = i;
            }
        }
    idx = index;
    }

HOSTDEVICE inline void support_ellipsoid(const vec3<Scalar> &rounding_radii, const vec3<Scalar> &vector, const quat<Scalar> &q, const vec3<Scalar> shift, vec3<Scalar> &ellipsoid_support_vector)
    {
    // Compute the support function of the rounding ellipsoid. Since we only
    // have the principal axes, we have to otate the vector into the principal
    // frame, compute the support, and then rotate back out.
    vec3<Scalar> rotated_vector = rotate(conj(q), vector);
    vec3<Scalar> numerator(
            rounding_radii.x*rounding_radii.x*rotated_vector.x,
            rounding_radii.y*rounding_radii.y*rotated_vector.y,
            rounding_radii.z*rounding_radii.z*rotated_vector.z);
    vec3<Scalar> dvec(
            rounding_radii.x*rotated_vector.x,
            rounding_radii.y*rotated_vector.y,
            rounding_radii.z*rotated_vector.z);
    ellipsoid_support_vector = rotate(q, numerator / fast::sqrt(dot(dvec, dvec)));
    }



//! Returns 1 if a and b have the same sign, and zero otherwise.
/*! The compareSigns function is used to implement the signed hyperplane checks
 *  that are used by the Signed Volumes subalgorithm to discard various Voronoi
 *  regions from consideration as containing the closest point to the origin.
 */
HOSTDEVICE inline unsigned int compareSigns(Scalar a, Scalar b)
{
    // Maybe there's a faster way to deal with this set of operations?
    return static_cast<unsigned int>(!((a > 0) ^ (b > 0)));
}


//! Find the point of minimum norm in a 1-simplex (an edge).
/*! \param W The set of (at most ndim+1) vectors making up the current simplex. The size of W is constant, but not all vectors are active at any given time.
 *  \param W_used A set of bit flags used to indicate which elements of W are actually participating in defining the current simplex (updated in place by the subalgorithm).
 *  \param lambdas The multipliers (that will be overwritten in place) that indicate what linear combination of W represents the point of minimum norm.
 */
template <unsigned int ndim>
HOSTDEVICE inline void s1d(gjk_vec3<Scalar>* W, unsigned int &W_used, Scalar* lambdas)
{
    // Identify the appropriate indices
    constexpr unsigned int max_num_points = ndim + 1;
    bool s1_set = false;
    unsigned int i1 = 0xffffffff, i2 = 0xffffffff;
    for (unsigned int i = 0; i < max_num_points; i++)
    {
        if (W_used & (1 << i))
        {
            if (s1_set)
            {
                i2 = i;
                break;
            }
            else
            {
                i1 = i;
                s1_set = true;
            }
        }
    }

    // Calculate the signed volume of the simplex.
    gjk_vec3<Scalar> t = W[i2] - W[i1];
    unsigned int I = 0;
    Scalar neg_tI = -t.arr[0];
    
    if (abs(t.arr[1]) > abs(neg_tI))
    {
        I = 1;
        neg_tI = -t.arr[1];
    }

    if (abs(t.arr[2]) > abs(neg_tI))
    {
        I = 2;
        neg_tI = -t.arr[2];
    }

    Scalar pI = (dot(W[i2], t)/dot(t, t)) * neg_tI + W[i2].arr[I];
    
    // Identify the signed volume resulting from replacing each point by the origin.
    Scalar C[2] = {-W[i2].arr[I] + pI, W[i1].arr[I] - pI};
    unsigned int sign_comparisons[2] = {compareSigns(neg_tI, C[0]), compareSigns(neg_tI, C[1])};
    
    // If all signed volumes are identical, the origin lies inside the simplex.
    if (sign_comparisons[0] + sign_comparisons[1] == 2)
    {
        lambdas[i1] = C[0] / neg_tI;
        lambdas[i2] = C[1] / neg_tI;
    }
    else
    {
        // The point to retain is the one whose sign matches. In the
        // first case, the origin lies past the first point.
        if (sign_comparisons[0])
        {
            W_used &= ~(1 << i2);
            lambdas[i1] = 1;
        }
        else
        {
            W_used &= ~(1 << i1);
            lambdas[i2] = 1;
        }
    }
}


//! Find the point of minimum norm in a 2-simplex (a face).
/*! \param W The set of (at most ndim+1) vectors making up the current simplex. The size of W is constant, but not all vectors are active at any given time.
 *  \param W_used A set of bit flags used to indicate which elements of W are actually participating in defining the current simplex (updated in place by the subalgorithm).
 *  \param lambdas The multipliers (that will be overwritten in place) that indicate what linear combination of W represents the point of minimum norm.
 */
template <unsigned int ndim>
HOSTDEVICE inline void s2d(gjk_vec3<Scalar>* W, unsigned int &W_used, Scalar* lambdas)
{
    // This function is always called with two points. This constant is defined
    // to avoid magical 3s everywhere in loops.
    constexpr unsigned int max_num_points = ndim + 1;
    constexpr unsigned int num_points = 3;
    unsigned int counter = 0, point0_idx = 0, point1_idx = 0, point2_idx = 0;
    for (unsigned int i = 0; i < max_num_points; ++i)
    {
        if (W_used & (1 << i))
        {
            if (counter == 0)
            {
                point0_idx = i;
            }
            else if (counter == 1)
            {
                point1_idx  = i;
            }
            else
            {
                point2_idx  = i;
            }
            counter += 1;
        }
    }

    gjk_vec3<Scalar> n = cross(W[point1_idx] - W[point0_idx], W[point2_idx] - W[point0_idx]);
    gjk_vec3<Scalar> p0 = (dot(W[point0_idx], n)/dot(n, n))*(n);
    
    // Choose maximum area plane to project onto.
    // Make sure to store the *signed* area of the plane.
    // This loop is unrolled to save a few extra ops (assigning
    // an initial area of zero, an extra abs, etc)
    unsigned int idx_x = 1;
    unsigned int idx_y = 2;
    Scalar mu_max = (
            W[point1_idx].arr[1] * W[point2_idx].arr[2] +
            W[point0_idx].arr[1] * W[point1_idx].arr[2] +
            W[point2_idx].arr[1] * W[point0_idx].arr[2] - 
            W[point1_idx].arr[1] * W[point0_idx].arr[2] -
            W[point2_idx].arr[1] * W[point1_idx].arr[2] -
            W[point0_idx].arr[1] * W[point2_idx].arr[2]);
    
    // This term is multiplied by -1.
    Scalar mu = (
            W[point1_idx].arr[2] * W[point0_idx].arr[0] +
            W[point2_idx].arr[2] * W[point1_idx].arr[0] +
            W[point0_idx].arr[2] * W[point2_idx].arr[0] -
            W[point1_idx].arr[2] * W[point2_idx].arr[0] -
            W[point0_idx].arr[2] * W[point1_idx].arr[0] -
            W[point2_idx].arr[2] * W[point0_idx].arr[0]);
    if (abs(mu) > abs(mu_max))
    {
        mu_max = mu;
        idx_x = 0;
    }

    mu = (
            W[point1_idx].arr[0] * W[point2_idx].arr[1] +
            W[point0_idx].arr[0] * W[point1_idx].arr[1] +
            W[point2_idx].arr[0] * W[point0_idx].arr[1] -
            W[point1_idx].arr[0] * W[point0_idx].arr[1] -
            W[point2_idx].arr[0] * W[point1_idx].arr[1] -
            W[point0_idx].arr[0] * W[point2_idx].arr[1]);
    if (abs(mu) > abs(mu_max))
    {
        mu_max = mu;
        idx_x = 0;
        idx_y = 1;
    }
    
    // Compute the signed areas of each of the simplices formed by replacing an
    // index with a projection of the origin onto the area in this plane
    Scalar C[num_points] = {0};
    bool sign_comparisons[num_points] = {false};
    
    C[0] = (p0.arr[idx_x] * W[point1_idx].arr[idx_y] +
            p0.arr[idx_y] * W[point2_idx].arr[idx_x] +
            W[point1_idx].arr[idx_x] * W[point2_idx].arr[idx_y] -
            p0.arr[idx_x] * W[point2_idx].arr[idx_y] -
            p0.arr[idx_y] * W[point1_idx].arr[idx_x] -
            W[point2_idx].arr[idx_x] * W[point1_idx].arr[idx_y]);
    sign_comparisons[0] = compareSigns(mu_max, C[0]);
    
    C[1] = (p0.arr[idx_x] * W[point2_idx].arr[idx_y] +
            p0.arr[idx_y] * W[point0_idx].arr[idx_x] +
            W[point2_idx].arr[idx_x] * W[point0_idx].arr[idx_y] -
            p0.arr[idx_x] * W[point0_idx].arr[idx_y] -
            p0.arr[idx_y] * W[point2_idx].arr[idx_x] -
            W[point0_idx].arr[idx_x] * W[point2_idx].arr[idx_y]);
    sign_comparisons[1] = compareSigns(mu_max, C[1]);
    
    C[2] = (p0.arr[idx_x] * W[point0_idx].arr[idx_y] +
            p0.arr[idx_y] * W[point1_idx].arr[idx_x] +
            W[point0_idx].arr[idx_x] * W[point1_idx].arr[idx_y] -
            p0.arr[idx_x] * W[point1_idx].arr[idx_y] -
            p0.arr[idx_y] * W[point0_idx].arr[idx_x] -
            W[point1_idx].arr[idx_x] * W[point0_idx].arr[idx_y]);
    sign_comparisons[2] = compareSigns(mu_max, C[2]);

    if (sign_comparisons[0] + sign_comparisons[1] + sign_comparisons[2] == 3)
    {
        lambdas[point0_idx] = C[0] / mu_max;
        lambdas[point1_idx] = C[1] / mu_max;
        lambdas[point2_idx] = C[2] / mu_max;
    }
    else
    {
        Scalar d = 1e9;
        gjk_vec3<Scalar> new_point;
        unsigned int new_W_used = 0;
        for (unsigned int j = 0; j < num_points; ++j)
        {
            if (!sign_comparisons[j])
            {
                unsigned int new_used = W_used;
                // Test removal of the current point.
                if (j == 0)
                {
                    new_used &= ~(1 << point0_idx);
                }
                else if (j == 1)
                {
                    new_used &= ~(1 << point1_idx);
                }
                else
                {
                    new_used &= ~(1 << point2_idx);
                }

                Scalar new_lambdas[max_num_points] = {0};
                
                s1d<ndim>(W, new_used, new_lambdas);
                // Consider resetting in place if possible.
                new_point.arr[0] = 0;
                new_point.arr[1] = 0;
                new_point.arr[2] = 0;
                for (unsigned int i = 0; i < max_num_points; ++i)
                {
                    if (new_used & (1 << i))
                    {
                        new_point += new_lambdas[i] * W[i];
                    }
                }
                Scalar d_star = dot(new_point, new_point);
                if (d_star < d)
                {
                    new_W_used = new_used;
                    d = d_star;
                    for (unsigned int i = 0; i < max_num_points; ++i)
                    {
                        lambdas[i] = new_lambdas[i];
                    }
                }
            }
        }
        W_used = new_W_used;
    }
}


//! Find the point of minimum norm in a 3-simplex (a volume, i.e. a tetrahedron).
/*! \param W The set of (at most ndim+1) vectors making up the current simplex. The size of W is constant, but not all vectors are active at any given time.
 *  \param W_used A set of bit flags used to indicate which elements of W are actually participating in defining the current simplex (updated in place by the subalgorithm).
 *  \param lambdas The multipliers (that will be overwritten in place) that indicate what linear combination of W represents the point of minimum norm.
 */
HOSTDEVICE inline void s3d(gjk_vec3<Scalar>* W, unsigned int &W_used, Scalar* lambdas)
{
    // This function is always called with 4 points, so a constant is defined
    // for clarity.
    constexpr unsigned int num_points = 4;
    // Unlike s1d and s2d, this function can only be called in 3d so it does not use the template 
    constexpr unsigned int ndim = 3;
    constexpr unsigned int max_num_points = ndim + 1;
    Scalar C[num_points] = {0};

    // Compute all minors and the total determinant of the matrix M,
    // which is the transpose of the W matrix with an extra row of
    // ones at the bottom. Since the indexing is nontrivial and the
    // array is small (and we can save on some negation), all the
    // computations are done directly rather than with a loop.
    // C[0] and C[2] are negated due to the (-1)^(i+j+1) prefactor,
    // where i is always 4 because we're expanding about the 4th row.
    C[0] = (W[3].arr[0] * W[2].arr[1] * W[1].arr[2] +
            W[2].arr[0] * W[1].arr[1] * W[3].arr[2] +
            W[1].arr[0] * W[3].arr[1] * W[2].arr[2] -
            W[1].arr[0] * W[2].arr[1] * W[3].arr[2] -
            W[2].arr[0] * W[3].arr[1] * W[1].arr[2] -
            W[3].arr[0] * W[1].arr[1] * W[2].arr[2]);
    C[1] = (W[0].arr[0] * W[2].arr[1] * W[3].arr[2] +
            W[2].arr[0] * W[3].arr[1] * W[0].arr[2] +
            W[3].arr[0] * W[0].arr[1] * W[2].arr[2] -
            W[3].arr[0] * W[2].arr[1] * W[0].arr[2] -
            W[2].arr[0] * W[0].arr[1] * W[3].arr[2] -
            W[0].arr[0] * W[3].arr[1] * W[2].arr[2]);
    C[2] = (W[3].arr[0] * W[1].arr[1] * W[0].arr[2] +
            W[1].arr[0] * W[0].arr[1] * W[3].arr[2] +
            W[0].arr[0] * W[3].arr[1] * W[1].arr[2] -
            W[0].arr[0] * W[1].arr[1] * W[3].arr[2] -
            W[1].arr[0] * W[3].arr[1] * W[0].arr[2] -
            W[3].arr[0] * W[0].arr[1] * W[1].arr[2]);
    C[3] = (W[0].arr[0] * W[1].arr[1] * W[2].arr[2] +
            W[1].arr[0] * W[2].arr[1] * W[0].arr[2] +
            W[2].arr[0] * W[0].arr[1] * W[1].arr[2] -
            W[2].arr[0] * W[1].arr[1] * W[0].arr[2] -
            W[1].arr[0] * W[0].arr[1] * W[2].arr[2] -
            W[0].arr[0] * W[2].arr[1] * W[1].arr[2]);

    Scalar dM = C[0] + C[1] + C[2] + C[3];

    unsigned int sign_comparisons[4] = {0};
    sign_comparisons[0] = compareSigns(dM, C[0]);
    sign_comparisons[1] = compareSigns(dM, C[1]);
    sign_comparisons[2] = compareSigns(dM, C[2]);
    sign_comparisons[3] = compareSigns(dM, C[3]);

    if ((sign_comparisons[0] + sign_comparisons[1] + sign_comparisons[2] +
            sign_comparisons[3]) == num_points)
    {
        for (unsigned int i = 0; i < num_points; ++i)
        {
            lambdas[i] = C[i] / dM;
        }
    }
    else
    {
        Scalar d = 1e9, d_star = 0;
        gjk_vec3<Scalar> new_point;
        unsigned int new_W_used = 0;
        for (unsigned int j = 0; j < num_points; ++j)
        {
            if (!sign_comparisons[j])
            {
                // Test removal of the current point.
                unsigned int new_used = W_used;
                new_used &= ~(1 << j);
                Scalar new_lambdas[max_num_points] = {0};

                s2d<3>(W, new_used, new_lambdas);

                new_point = gjk_vec3<Scalar>();
                for (unsigned int i = 0; i < max_num_points; ++i)
                {
                    if (new_used & (1 << i))
                    {
                        new_point += new_lambdas[i] * W[i];
                    }
                }
                d_star = dot(new_point, new_point);
                if (d_star < d)
                {
                    new_W_used = new_used;
                    d = d_star;
                    for (unsigned int i = 0; i < max_num_points; ++i)
                    {
                        lambdas[i] = new_lambdas[i];
                    }
                }
            }
        }
        W_used = new_W_used;
    }
}


// TODO: Rewrite the subalgorithm function separately for 2d and 3d.
// That will avoid the extra if check for 2d (probably premature optimization,
// but may help clean the code's usage of a template parameter for s3d.
//! The Signed Volumes subalgorithm for finding minimal spanning convex sets.
/*! The Signed Volumes subalgorithm (described in Montanari, M., Petrinic, N.,
 *  & Barbieri, E. (2017). Improving the GJK algorithm for faster and more
 *  reliable distance queries between convex objects. ACM Transactions on
 *  Graphics (TOG), 36(3), 1-17, DOI: 10.1145/3083724) is a robust procedure
 *  for finding the point of minimum norm in a convex set as a linear
 *  combination of the extremal points of that set. The algorithm takes a
 *  convex simplex and solves a linear system of equations for the closest
 *  point to the origin. Since the algorithm is specifically designed for 3
 *  dimensional queries, this function dispatches to three separate subroutines
 *  that handle the different dimensionalities explicitly.
 *
 *  \param W The set of (at most ndim+1) vectors making up the current simplex. The size of W is constant, but not all vectors are active at any given time.
 *  \param W_used A set of bit flags used to indicate which elements of W are actually participating in defining the current simplex (updated in place by the subalgorithm).
 *  \param lambdas The multipliers (that will be overwritten in place) that indicate what linear combination of W represents the point of minimum norm.
 */
template <unsigned int ndim>
HOSTDEVICE inline void sv_subalgorithm(gjk_vec3<Scalar>* W, unsigned int &W_used, Scalar* lambdas)
{
    // The W array is never modified by this function.  The W_used may be
    // modified if necessary, and the lambdas will be updated.  All the other
    // functions (if they need to make deeper calls e.g. s3d->s2d) will have to
    // make copies of W_used to avoid overwriting that data incorrectly.
    unsigned int num_used = 0;
    constexpr unsigned int max_num_points = ndim + 1;
    for (unsigned int i = 0; i < max_num_points; ++i)
    {
        num_used += (W_used >> i) & 1;
    }

    // Start with the most common cases.
    if (num_used == 1)
    {
        for (unsigned int i = 0; i < max_num_points; ++i)
        {
            if (W_used & (1 << i))
            {
                lambdas[i] = 1;
            }
        }
    }
    else if (num_used == 2)
    {
        s1d<ndim>(W, W_used, lambdas);
    }
    else if (num_used == 3)
    {
        s2d<ndim>(W, W_used, lambdas);
    }
    // TODO: This branch will never happen in 3D, but without using C++17 features (if constexpr) I'm not
    // sure how to avoid compiling this for the 2D template, so I get a lot of warnings on compilation.
    else
    {
        // This case only happens in 3D, so no dimensionality is specified.
        s3d(W, W_used, lambdas);
    }
}


//! Apply the GJK algorithm to find the vector between the closest points on two sets of shapes.
/*! This function implements the Gilbert-Johnson-Keerthi distance algorithm for
 *  finding the minimum distance between two convex sets. The purpose of this
 *  function is to be used in concert with the EvaluatorPairALJI class for
 *  computing the force and energy between two anisotropic bodies. This paper
 *  uses information from two papers:
 *      1. Gino Van den Bergen (1999) A Fast and Robust GJK Implementation for
 *         Collision Detection of Convex Objects, Journal of Graphics Tools,
 *         4:2, 7-25, DOI: 10.1080/10867651.1999.10487502.
 *      2. Montanari, M., Petrinic, N., & Barbieri, E. (2017). Improving the
 *         GJK algorithm for faster and more reliable distance queries between
 *         convex objects. ACM Transactions on Graphics (TOG), 36(3), 1-17,
 *         DOI: 10.1145/3083724.
 *
 *  The standard GJK algorithm is a descent algorithm for finding the minimum
 *  distance between two convex sets. The core insight of the algorithm is that
 *  this query can be performed in the configuration space of the Minkowski
 *  difference body of the two sets. The algorithm takes the Minkowski
 *  difference (more precisely, the Minkowski sum A-B) of two convex sets and
 *  performs a search to see if the origin is contained in the difference (this
 *  difference is also known as the translational configuration space obstacle,
 *  or TCSO, in this context).  If the origin is contained in the TCSO, the
 *  same point is encompassed by both shapes, so they must be overlapping. If
 *  not, the point on the TCSO closest to the origin corresponds to the vector
 *  of minimum norm connecting the two sets.
 *
 *  In order to determine whether the origin is contained in the TCSO, the
 *  algorithm iteratively constructs simplices that are composed of points on
 *  the TCSO until one is found that contains the origin. In the case of
 *  nonoverlapping shapes, the algorithm ultimately constructs a simplex of the
 *  TCSO containing the closest point to the origin. The termination condition
 *  in this case relies on maintaining a lower bound of this distance that is
 *  based on the distance from the origin to the closest hyperplane. For
 *  efficiency in overlap checks, it is notable that this lower bound provides
 *  an immediate indication if two shapes are nonoverlapping if at any point a
 *  separating axis is found (i.e. a hyperplane with positive distance). To
 *  actually find the distance between two shapes, the algorithm proceeds until
 *  the vector distance between the two shapes is within some tolerance of the
 *  estimated lower bound.
 *
 *  The GJK algorithm relies on support mappings, which provide a way to find
 *  the furthest point from the center of a given convex set in a specified
 *  direction. Crucially, the support function for the Minkowski sum of two
 *  convex shapes is the sum of the support functions of each shape, so knowing
 *  the support functions for the underlying shapes immediately allows
 *  evaluation of the support function of the difference body. Since the
 *  support function of the difference body is all that is needed to construct
 *  simplices at each GJK iteration, at each iteration the GJK algorithm
 *  requires only minimal geometric information. The use of support mappings
 *  eschews the need for any explicit mathematical descriptions of the two sets
 *  and obviates the need for constructing the full difference body between the
 *  two convex sets. As a result, GJK is more generally applicable than other
 *  related algorithms, which are typically designed specially for polytopes or
 *  other classes of shapes.
 *
 *  A critical component of the algorithm is the determination of a minimal
 *  simplex.  The support function is used to define a new candidate point for
 *  inclusion in the minimal simplex at each iteration, and then the resulting
 *  simplex is reduced to a minimal set such that the closest point to the
 *  origin is contained in the span of that set. Constructing this minimal
 *  simplex essentially involves solving a (possibly overdetermined) system of
 *  linear equations to find the point closest to the origin. The classic
 *  solution to this, the Johnson subalgorithm, uses Cramer's rule to solve
 *  this system.  Although normally inefficient, in this case Cramer's rule is
 *  optimal because at each iteration only one new vertex is added to the
 *  simplex, so most of the minors computed in Cramer's rule are already known.
 *  However, in certain nearly degenerate cases the Johnson subalgorithm can
 *  fail to return the correct result. In this case, the original solution has
 *  always been to follow a backup procedure that amounts to a brute force
 *  search through all possible simplices. 
 *
 *  The backup procedure is quite expensive, so in principle we would like to
 *  use the Johnson algorithm alone. Bergen et al assert that in practice the
 *  difference between the best result of the Johnson algorithm and the output
 *  of the backup procedure are nearly equivalent with the appropriate
 *  termination conditions imposed to avoid instabilities. In practice, we have
 *  found this to be true. However, for our purposes we require more
 *  information than the original GJK algorithm provides. Since our use-case is
 *  to compute forces and torques in EvaluatorPairALJ, we also need information
 *  on the points in the original convex sets that correspond to the minimum
 *  distance vector in the TCSO. Although the GJK algorithm can easily be
 *  augmented to provide this information, we find that for this usage the
 *  Johnson algorithm falls short. While the distances it provides remain close
 *  to optimal, the exact points it finds can deviate substantially from the
 *  correct ones in nearly degenerate cases, such as when faces are nearly
 *  parallel. The result of this is that while forces can still be calculated
 *  nearly correctly, torques can be completely incorrect, and in fact may even
 *  point in the wrong direction.
 *
 *  To ameliorate this problem, we instead use the Signed Volumes subalgorithm
 *  described by Montenari et al. This algorithm resolves the issues with the
 *  Johnson algorithm in all tested cases. Moreover, the Johnson algorithm is
 *  quite memory intensive because it requires caching all support function
 *  evaluations and cofactors required for Cramer's rule. As a result, we find
 *  that the Signed Volumes subalgorithm performs better than the Johnson
 *  algorithm (even without the backup procedure) when run on modern GPUs.
 *
 *  \param verts1 The vertices of the first body.
 *  \param verts2 The vertices of the second body.
 *  \param v Reference to vec3 that will be overwritten with the vector joining the closest intersecting points on the two bodies (CRITICAL NOTE: The direction of the vector is from verts2 to verts1).
 *  \param a Reference to vec3 that will be overwritten with the vector from the origin (in the frame defined by verts1) to the point on the body represented by verts1 that is closest to verts2.
 *  \param b Reference to vec3 that will be overwritten with the vector from the origin (in the frame defined by verts2) to the point on the body represented by verts2 that is closest to verts1.
 *  \param success Reference to bool that will be overwritten with whether or not the algorithm terminated in the maximum number of allowed iterations (verts1.size + verts2.size + 1).
 *  \param overlap Reference to bool that will be overwritten with whether or not an overlap was detected.
 *  \param qi The orientation of the first shape (will be applied to verts1).
 *  \param qj The orientation of the first shape (will be applied to verts2).
 *  \param dr The vector pointing from the position of particle 2 to the position of particle 1 (note the sign; this is reversed throughout most of the calculations below).
 */
template <unsigned int ndim>
HOSTDEVICE inline void gjk(const ManagedArray<vec3<Scalar> > &verts1, const ManagedArray<vec3<Scalar> > &verts2, vec3<Scalar> &v, vec3<Scalar> &a, vec3<Scalar> &b, bool& success, bool& overlap, const quat<Scalar> &qi, const quat<Scalar> &qj, const vec3<Scalar> &dr, const vec3<Scalar> &rounding_radii1, const vec3<Scalar> &rounding_radii2, bool has_rounding1, bool has_rounding2)
    {
    // At any point only a subset of W is in use (identified by W_used), but
    // the total possible is capped at ndim+1 because that is the largest
    // number of affinely independent points in R^n.
    constexpr unsigned int max_num_points = ndim + 1;
    success = true;

    // Start with guess as vector pointing from the centroid of verts1 to the
    // centroid of verts2.
    //vec3<Scalar> mean1, mean2;
    //for(unsigned int i = 0; i < verts1.size(); i++)
        //{
        //mean1 += rotate(qi, verts1[i]);
        //}
    //for(unsigned int i = 0; i < verts2.size(); i++)
        //{
        //mean2 += (rotate(qj, verts2[i]) + Scalar(-1.0)*dr);
        //}
    //mean1 /= Scalar(verts1.size());
    //mean2 /= Scalar(verts2.size());
    //v = mean1 - mean2;
    v = dr;

    // We don't bother to initialize most of these arrays since the W_used
    // array controls which data is valid. 
    gjk_vec3<Scalar> W[max_num_points];
    Scalar lambdas[max_num_points];
    unsigned int W_used = 0;
    unsigned int indices1[max_num_points] = {0};
    unsigned int indices2[max_num_points] = {0};
    vec3<Scalar> ellipsoid_supports1[max_num_points];
    vec3<Scalar> ellipsoid_supports2[max_num_points];

    for (unsigned int i = 0; i < max_num_points; ++i)
        {
        // We initialize W to avoid accidentally termianting if the new w is
        // somehow equal to somthing saved in one of the uninitialized W[i].
        W[i] = gjk_vec3<Scalar>();
        ellipsoid_supports1[i] = vec3<Scalar>();
        ellipsoid_supports2[i] = vec3<Scalar>();
        }

    // The tolerances are compile-time constants.
    constexpr Scalar eps(1e-8), omega(1e-4);

    Scalar u(0); 
    bool close_enough(false);
    unsigned int max_iterations = verts1.size() + verts2.size() + 1;
    unsigned int iteration = 0;
    while (!close_enough)
        {
        iteration += 1;
        if (iteration > max_iterations)
            {
            success = false;
            break;
            }
        // support_{A-B}(-v) = support(A, -v) - support(B, v)
        vec3<Scalar> ellipsoid_support1, ellipsoid_support2;
        unsigned int i1, i2;
        support_polyhedron(verts1, -v, qi, vec3<Scalar>(0, 0, 0), i1);
        support_polyhedron(verts2, v, qj, Scalar(-1.0)*dr, i2);
        if (has_rounding1)
            {
            support_ellipsoid(rounding_radii1, -v, qi, vec3<Scalar>(0, 0, 0), ellipsoid_support1);
            }
        if (has_rounding2)
            {
            support_ellipsoid(rounding_radii2, v, qj, Scalar(-1.0)*dr, ellipsoid_support2);
            }

        gjk_vec3<Scalar> w(rotate(qi, verts1[i1]) + ellipsoid_support1 - (rotate(qj, verts2[i2]) + Scalar(-1.0)*dr + ellipsoid_support2));

        // Check termination conditions for degenerate cases:
        // 1) If we are repeatedly finding the same point but can't get closer
        // and can't terminate within machine precision.
        // 2) If we are cycling between two points.
        // In either case, because of the tracking with W_used, we can
        // guarantee that the new w will be found in one of the W (but possibly
        // in one of the unused slots.
#ifndef NVCC
        bool degenerate(false);
        for (unsigned int i = 0; i < max_num_points; i++)
            {
            if (w == W[i])
                {
                degenerate = true;
                break;
                }
            }
#endif

        Scalar vnorm = sqrt(dot(v, v));
        Scalar d = dot(v, w)/vnorm;
        // If we ever have d > 0, we can immediately that the two shapes never
        // intersect! Actually finding an intersection requires waiting until
        // we actually have an affinely dependent set of points, though.
        u = u > d ? u : d;
#ifdef NVCC
        close_enough = ( ((vnorm - u) <= eps*vnorm) || (vnorm < omega) );
#else
        close_enough = ( degenerate || ((vnorm - u) <= eps*vnorm) || (vnorm < omega) );
#endif
        if (!close_enough)
            {
            unsigned int new_index(0);
            for (; new_index < max_num_points; ++new_index)
                {
                // At least one of these must be empty, otherwise we have an
                // overlap. 
                if (!(W_used & (1 << new_index)))
                    {
                    W[new_index] = w;
                    W_used |= (1 << new_index);
                    indices1[new_index] = i1;
                    indices2[new_index] = i2;
                    ellipsoid_supports1[new_index] = ellipsoid_support1;
                    ellipsoid_supports2[new_index] = ellipsoid_support2;
                    break;
                    }
                }
            sv_subalgorithm<ndim>(W, W_used, lambdas);

            v = vec3<Scalar>();
            for (unsigned int i = 0; i < max_num_points; i++)
                {
                if (W_used & (1 << i))
                    {
                    v += lambdas[i]*W[i];
                    }
                }
            }
        }

    a = vec3<Scalar>();
    b = vec3<Scalar>();
    unsigned int counter = 0;
    for (unsigned int i = 0; i < max_num_points; i++)
        {
        if (W_used & (1 << i))
            {
            a += lambdas[i]*(rotate(qi, verts1[indices1[i]]) + ellipsoid_supports1[i]);
            b += lambdas[i]*(rotate(qj, verts2[indices2[i]]) + Scalar(-1.0)*dr + ellipsoid_supports2[i]);
            counter += 1;
            }
        }

    overlap = (counter == (max_num_points));
    }

#endif // __GJK_SV_H__
