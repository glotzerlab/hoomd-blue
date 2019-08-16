#ifndef __GJK_H__
#define __GJK_H__

#include "hoomd/VectorMath.h"

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#include <stdexcept>
#endif

DEVICE inline unsigned int support(const ManagedArray<vec3<Scalar> > &verts, const vec3<Scalar> &vector, const quat<Scalar> &q, const vec3<Scalar> shift)
    {
    unsigned int index = 0;

    Scalar max_dist = dot((rotate(q, verts[index]) + shift), vector);
    for (unsigned int i = 1; i < verts.size(); ++i)
        {
        Scalar dist = dot((rotate(q, verts[i]) + shift), vector);
        if (dist > max_dist)
            {
            max_dist = dist;
            index = i;
            }
        }
    return index;
    }


// Note: All of the bitwise indexing schemes could fail if ndim is too large.
// However, this shouldn't be a concern for any realistic number of dimensions.
// All bit flags are representing either arrays of size max_num_points, which
// is ndim+1, or of size max_power_set_size, which is 2^(max_num_points).
// Therefore, the largest possible bit indexer (which is currently
// comb_contains, must be of size 2^(2^(ndim+1)). As long as these indexes are
// declared of type unsigned long long int, they are guaranteed at least 64
// bits, which means they support up to 5 dimensions 2^(5+1) = 2^6 = 64). If
// more dimensions were ever needed, we could replace these bit indexes with
// simple boolean arrays, but this slows down code substantially on the GPU.

//! Apply the GJK algorithm to find the vector between the closest points on two sets of shapes.
/*! This function implements the GJK algorithm as described in
 *  Gino Van den Bergen (1999) A Fast and Robust GJK Implementation for
 *  Collision Detection of Convex Objects, Journal of Graphics Tools, 4:2, 7-25,
 *  DOI: 10.1080/10867651.1999.10487502.
 *
 *  The standard GJK algorithm takes the Minkowski difference (more precisely,
 *  the Minkowski sum A-B) of two convex shapes and performs a search to see if
 *  the origin is contained in the difference (this difference is also known as
 *  the translational configuration space obstacle, or TCSO, in this context).
 *  If the origin is contained in the TCSO, the same point is encompassed by
 *  both shapes, so they must be overlapping. In order to determine whether the
 *  origin is contained in the TCSO, the algorithm iteratively constructs
 *  simplices that are composed of points on the TCSO until one is found that
 *  contains the origin. In the case of nonoverlapping shapes, the algorithm
 *  ultimately constructs a simplex of the TCSO containing the closest point to
 *  the origin. The termination condition in this case relies on maintaining a
 *  lower bound of this distance that is based on the distance from the origin
 *  to the closest hyperplane. For efficiency in overlap checks, it is notable
 *  that this lower bound provides an immediate indication if two shapes are
 *  nonoverlapping if at any point a separating axis is found (i.e. a
 *  hyperplane with positive distance). To actually find the distance between
 *  two shapes, the algorithm proceeds until the vector distance between the
 *  two shapes is within some tolerance of the estimated lower bound.
 *
 *  There are two important components of the algorithm worthy of special note.
 *  One critical component of this algorithm is the actual construction of the
 *  simplex at each step in the iteration. This construction is performed using
 *  the Johnson subalgorithm, which essentially uses Cramer's rule to solve a
 *  system of linear equations. Although normally inadvisable, in this case
 *  Cramer's rule is optimal because at each iteration only one new vertex is
 *  added to the simplex, so most of the minors computed in Cramer's rule are
 *  already known. The second crucial component is the use of support mappings,
 *  which provide a way to find the furthest point from the center of a given
 *  convex shape in a specified direction. These support mappings can be
 *  implemented for any general convex shape, and their usage makes this
 *  algorithm both general and efficient since it eschews any need for any
 *  other explicit mathematical description of the shape.
 *
 *  The implementation discussed in the above paper offers multiple
 *  optimizations on the original, including the separating hyperplane check
 *  noted previously. The algorithm is accelerated by caching all support
 *  function evaluations and the cofactors required for Cramer's rule. To
 *  improve robustness, we apply the improved termination conditions suggested
 *  by van den Bergen as well. Additionally, we omit the use of the classical
 *  backup procedure provided in the original paper, which in practice provides
 *  very minimal improvement on the solution at significant computational cost.
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
DEVICE inline void gjk(const ManagedArray<vec3<Scalar> > &verts1, const ManagedArray<vec3<Scalar> > &verts2, vec3<Scalar> &v, vec3<Scalar> &a, vec3<Scalar> &b, bool& success, bool& overlap, const quat<Scalar> &qi, const quat<Scalar> &qj, const vec3<Scalar> &dr)
    {
    // At any point only a subset of W is in use (identified by W_used), but
    // the total possible is capped at ndim+1 because that is the largest
    // number of affinely independent points in R^n.
    constexpr unsigned int max_num_points = ndim + 1;
    success = true;
    overlap = true;

    // Start with guess as vector pointing from the centroid of verts1 to the
    // centroid of verts2.
    vec3<Scalar> mean1, mean2;
    for(unsigned int i = 0; i < verts1.size(); ++i)
        {
        mean1 += rotate(qi, verts1[i]);
        }
    for(unsigned int i = 0; i < verts2.size(); ++i)
        {
        mean2 += (rotate(qj, verts2[i]) + Scalar(-1.0)*dr);
        }
    mean1 /= Scalar(verts1.size());
    mean2 /= Scalar(verts2.size());
    v = mean1 - mean2;

    vec3<Scalar> W[max_num_points];
    Scalar lambdas[max_num_points] = {0};
    unsigned int W_used = 0;  // To be used as a set of bit flags
    unsigned int indices1[max_num_points] = {0};
    unsigned int indices2[max_num_points] = {0};

    for (unsigned int i = 0; i < max_num_points; ++i)
        {
        // We initialize W to avoid accidentally termianting if the new w is
        // somehow equal to somthing saved in one of the uninitialized W[i].
        W[i] = vec3<Scalar>();
        }

    // The first dimension shape of the deltas array is the total
    // number of possible subsets, which is the cardinality
    // of the power set (technically minus the empty set, but
    // for indexing simplicity we just leave the 0th row empty).
    // The second is the maximum number of affinely
    // independent points, which is the dimension + 1
    constexpr unsigned int max_power_set_size = (1 << (max_num_points));
    Scalar deltas[max_power_set_size][max_num_points];
    for (unsigned int i = 0; i < max_power_set_size; ++i)
        {
        for (unsigned int j = 0; j < max_num_points; ++j)
            {
            deltas[i][j] = 0;
            }
        }

    // Base case for recursive algorithm is a set of size 1. While the distance
    // subalgorithm is recursive in computing sets of increasing size, the
    // outer GJK algorithm is not. Therefore, there is no guarantee that all
    // sets of size one will have been set by other calls to the Johnson
    // algorithm before accessing them, so we need to establish the base case
    // explicitly by setting these here.
    for (unsigned int i = 0; i < max_num_points; ++i)
        {
        deltas[1 << i][i] = 1;
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
        unsigned int i1 = support(verts1, -v, qi, vec3<Scalar>(0, 0, 0));
        unsigned int i2 = support(verts2, v, qj, Scalar(-1.0)*dr);
        vec3<Scalar> w = rotate(qi, verts1[i1]) - (rotate(qj, verts2[i2]) + Scalar(-1.0)*dr);

        // Check termination conditions for degenerate cases:
        // 1) If we are repeatedly finding the same point but can't get closer
        // and can't terminate within machine precision.
        // 2) If we are cycling between two points.
        // In either case, because of the tracking with W_used, we can
        // guarantee that the new w will be found in one of the W (but possibly
        // in one of the unused slots.
        //
        // We don't bother with this on the GPU, the resulting warp divergence
        // is worse for performance than just going through the max number of
        // iterations on all threads.
#ifndef NVCC
        bool degenerate(false);
        for (unsigned int i = 0; i < max_num_points; ++i)
            {
            if (w == W[i])
                {
                degenerate = true;
                break;
                }
            }
#endif

        Scalar vnorm = sqrt(dot(v, v));
        Scalar dvw = dot(v, w);
        Scalar d = dvw/vnorm;
        u = u > d ? u : d;
        if (d > 0)
            {
            overlap = false;
            }

#ifdef NVCC
        close_enough = ( ((vnorm - u) <= eps*vnorm) || (vnorm < omega) );
#else
        close_enough = ( degenerate || ((vnorm - u) <= eps*vnorm) || (vnorm < omega) );
#endif
        if (!close_enough)
            {
            unsigned int added_element(0);
            for (; added_element < max_num_points; ++added_element)
                {
                // At least one of these must be empty, otherwise we have an
                // overlap.
                if (!(W_used & (1 << added_element)))
                    {
                    W[added_element] = w;
                    W_used |= (1 << added_element);
                    indices1[added_element] = i1;
                    indices2[added_element] = i2;
                    break;
                    }
                }
            bool use_last(false);

            /////////////////////////////////////////////
            ////////// BEGIN JOHNSON ALGORITHM //////////
            /////////////////////////////////////////////
            unsigned int added_index = 1 << added_element;

            // If there is only one point in use, we can return immediately.
            unsigned int num_used = 0;
            for (unsigned int i = 0; i < max_num_points; ++i)
                {
                num_used += (W_used >> i) & (1);
                }
            if (num_used == 1)
                {
                for (unsigned int i = 0; i < max_num_points; ++i)
                    {
                    if (W_used & (1 << i))
                        {
                        lambdas[i] = 1.0;
                        }
                    }
                }
            else
                {
                // The check_indexes array is simply linearly indexed, and it contains the
                // subsets of W that we need to test. The sets are inserted as they are
                // created over the course of the recursive algorithm. The sets are stored
                // as unsigned integer representations of (ndim+1)-bits. For example, if W
                // has 4 points, the subset consisting of the first 3 points is stored as
                // 0111. The current_subset is used to select the current index to check, and
                // next_subset_slot indicates the next open spot when a new set is found. To
                // efficiently determine whether a particular subset is new or has been
                // seen before, we maintain a separate boolean array comb_contains that is
                // updated as new subsets are found. Note that comb_contains[0] is never
                // used since it corresponds to the empty set, but is left indexed this way
                // to simplify the access pattern using the bit-based indexing
                // scheme. For efficiency, comb_contains is not actually an
                // array, but is instead an unsigned int whose digits are bit
                // flags.
                unsigned int current_subset = 0, next_subset_slot = 0;
                unsigned int check_indexes[max_power_set_size - 1] = {0};
                unsigned long long int comb_contains = 0;

                for (unsigned int i = 0; i < max_num_points; ++i)
                    {
                    unsigned int index_i(1 << i);
                    if (W_used & index_i)
                        {
                        check_indexes[next_subset_slot] = index_i;
                        comb_contains |= (1 << index_i);
                        next_subset_slot += 1;
                        }
                    }

                unsigned int desired_index = max_power_set_size;
                while (current_subset < next_subset_slot)
                    {
                    unsigned int current_index = check_indexes[current_subset];

                    // Make a local copy of the deltas that will be used frequently
                    // to hint to the GPU that these values should be cached.
                    // To give the GPU a hint that we need to cache specific
                    // delta values, we point to just the relevant subset for
                    // the current index. Rather than making another local
                    // array (which would require further memory allocation and
                    // copying), we just access the relevant deltas by pointer,
                    // which seems to be sufficient.
                    Scalar *deltas_current = deltas[current_index];

                    // The vector k must be fixed, but can be chosen
                    // arbitrarily. A simple deterministic choice is the last
                    // used index.
                    unsigned int k = 0;
                    for (unsigned int i = 0; i < max_num_points; ++i)
                        {
                        if ((1 << i) & current_index)
                            {
                            k = i;
                            break;
                            }
                        }

                    // Caching the W_k value for speed on the GPU.
                    const vec3<Scalar> W_k = W[k];

                    bool complete = true;
                    for (unsigned int new_element = 0; new_element < max_num_points; ++new_element)
                        {
                        unsigned int shifted_new_element = 1 << new_element;
                        // Add new elements that are in use and not contained in the current set.
                        if ((W_used & shifted_new_element) && !(current_index & shifted_new_element))
                            {
                            // Generate the corresponding bit-based index for the new set.
                            unsigned int new_index = current_index | shifted_new_element;

                            Scalar delta_new = 0;

                            // The only sets for which we will not have cached data are
                            // sets that contain the element most recently added to W.
                            if ((added_index & current_index) || (added_element == new_element))
                                {
                                // Caching the W_new value for speed on the GPU.
                                const vec3<Scalar> W_diff = W_k - W[new_element];

                                // Use bitwise checks of all possible elements to find set members
                                for (unsigned int possible_element = 0; possible_element < max_num_points; ++possible_element)
                                    {
                                    if ((1 << possible_element) & current_index)
                                        {
                                        delta_new += *(deltas_current + possible_element)*dot(W[possible_element], W_diff);
                                        }
                                    }
                                deltas[new_index][new_element] = delta_new;
                                }
                            else
                                {
                                // Minimize data reads by using delta_current
                                // in the termination conditional below and
                                // only reading from deltas when we are using a
                                // cached value.
                                delta_new = deltas[new_index][new_element];
                                }
                            unsigned int shifted_new_index = 1 << new_index;
                            if (!(comb_contains & shifted_new_index))
                                {
                                comb_contains |= shifted_new_index;
                                check_indexes[next_subset_slot] = new_index;
                                next_subset_slot += 1;
                                }

                            // Part (ii) of termination condition: Delta_j(X U {y_j}) <= 0
                            // for all j not in current_subset
                            // Could add additional check beforehand using added_index
                            if (delta_new > 0)
                                {
                                complete = false;
                                }
                            }
                        }
                    // Part (i) of termination condition: Delta_i(X) > 0 for all i in current_subset
                    // Could add additional check beforehand using added_index
                    if (complete)
                        {
                        for (unsigned int i = 0; i < max_num_points; ++i)
                            {
                            if (((1 << i) & current_index) && (*(deltas_current+i) <= 0))
                                {
                                complete = false;
                                break;
                                }
                            }
                        }
                    if (complete)
                        {
                        desired_index = current_index;
                        break;
                        }
                    current_subset += 1;
                    }

                // If we didn't find a solution that fits the termination
                // criterion, we just use the previous solution, because it's
                // typically close enough.
                if (desired_index == max_power_set_size)
                    {
                    use_last = true;
                    }
                else
                    {
                    // The sum of relevant deltas is used to scale the lambdas.
                    Scalar delta_total(0);
                    for (unsigned int i = 0; i < max_num_points; ++i)
                        {
                        delta_total += deltas[desired_index][i];
                        }

                    for (unsigned int i = 0; i < max_num_points; ++i)
                        {
                        unsigned int shifted_i = 1 << i;
                        if (shifted_i & desired_index)
                            {
                            W_used |= shifted_i;
                            lambdas[i] = deltas[desired_index][i]/delta_total;
                            }
                        else
                            {
                            W_used &= ~shifted_i;
                            }
                        }
                    }
                }
            /////////////////////////////////////////////
            /////////// END JOHNSON ALGORITHM ///////////
            /////////////////////////////////////////////

            if (use_last)
                {
                break;
                }

            v = vec3<Scalar>();
            for (unsigned int i = 0; i < max_num_points; ++i)
                {
                if (W_used & (1 << i))
                    {
                    v += lambdas[i]*W[i];
                    }
                }
            }
        }

    // A compiler bug causes the for loop below to never terminate when
    // using max_num_points as the upper bound. Defining a new (equivalent)
    // variable seems to fix it.
    constexpr unsigned int new_limit = max_num_points;
    a = vec3<Scalar>();
    b = vec3<Scalar>();
    for (unsigned int i = 0; i < new_limit; ++i)
        {
        if (W_used & (1 << i))
            {
            a += lambdas[i]*rotate(qi, verts1[indices1[i]]);
            b += lambdas[i]*(rotate(qj, verts2[indices2[i]]) + Scalar(-1.0)*dr);
            }
        }
    }
#endif // __GJK_H__
