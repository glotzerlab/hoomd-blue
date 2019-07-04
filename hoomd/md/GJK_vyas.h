#ifndef __GJK_H__
#define __GJK_H__

#include "hoomd/VectorMath.h"

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#include <stdexcept>
#endif

DEVICE inline unsigned int support(const ManagedArray<vec3<Scalar> > verts, const unsigned int &num_verts, const vec3<Scalar> &vector, const quat<Scalar> &q, const vec3<Scalar> shift)
    {
    unsigned int index = 0;

    Scalar max_dist = dot((rotate(q, verts[index]) + shift), vector);
    for (unsigned int i = 1; i < num_verts; ++i)
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


/* Version of GJK that, when it finds overlaps, instead returns two vectors
 * that point from the centers of each particle to the point on the other one
 * closest to the center (not quite the same as maximum penetration point, but
 * similar).
 */
/* FOR NOW, LEAVING COMMENTED OUT BECAUSE THERE'S NOT A GOOD WAY TO PASS POS1
 * AND POS2 AS MANAGEDARRAYS IN THE RECURSIVE CASE.
 */
//template <unsigned int ndim>
//DEVICE inline void gjk_with_overlaps(vec3<Scalar> &pos1, vec3<Scalar> &pos2, ManagedArray<vec3<Scalar> > verts1, unsigned int &N1, ManagedArray<vec3<Scalar> > verts2, unsigned int &N2, vec3<Scalar> &v1, vec3<Scalar> &v2, vec3<Scalar> &a, vec3<Scalar> &b, bool& success, bool& overlap, const quat<Scalar> &qi, const quat<Scalar> &qj, const vec3<Scalar> &dr, bool in_recursion=false)
    //{
    //// Note that the vector v returned is the vector a - b, not b - a.
    //// Generally doesn't matter, but in the overlap case it's important to note
    //// that v1 = pos1 - b and v2 = a - pos2

    //// At any point only a subset of W is in use (identified by W_used), but
    //// the total possible is capped at ndim+1 because that is the largest
    //// number of affinely independent points in R^n.
    //constexpr unsigned int max_num_points = ndim + 1;
    //success = true;
    //overlap = true;

    //// Start with guess as vector pointing from the centroid of verts1 to the
    //// centroid of verts2.
    //vec3<Scalar> mean1, mean2;
    //for(unsigned int i = 0; i < N1; ++i)
        //{
        //mean1 += rotate(qi, verts1[i]);
        //}
    //for(unsigned int i = 0; i < N2; ++i)
        //{
        //mean2 += (rotate(qj, verts2[i]) + Scalar(-1.0)*dr);
        //}
    //mean1 /= Scalar(N1);
    //mean2 /= Scalar(N2);
    //v = mean1 - mean2; 

    //vec3<Scalar> W[max_num_points];
    //Scalar lambdas[max_num_points] = {0};
    //bool W_used[max_num_points] = {false};
    //unsigned int indices1[max_num_points] = {0};
    //unsigned int indices2[max_num_points] = {0};

    //for (unsigned int i = 0; i < max_num_points; ++i)
        //{
        //// We initialize W to avoid accidentally termianting if the new w is
        //// somehow equal to somthing saved in one of the uninitialized W[i].
        //W[i] = vec3<Scalar>();
        //}

    //// The first dimension shape of the deltas array is the total
    //// number of possible subsets, which is the cardinality
    //// of the power set (technically minus the empty set, but
    //// for indexing simplicity we just leave the 0th row empty).
    //// The second is the maximum number of affinely
    //// independent points, which is the dimension + 1
    //constexpr unsigned int max_power_set_size = (1 << (max_num_points));
    //Scalar deltas[max_power_set_size][max_num_points];
    //for (unsigned int i = 0; i < max_power_set_size; ++i)
        //{
        //for (unsigned int j = 0; j < max_num_points; j++)
            //{
            //deltas[i][j] = 0;
            //}
        //}

    //Scalar u(0), eps(1e-8), omega(1e-4); 
    //bool close_enough(false);
    //unsigned int max_iterations = N1 + N2 + 1;
    //unsigned int iteration = 0;
    //while (!close_enough)
        //{
        //iteration += 1;
        //if (iteration > max_iterations)
            //{
            //success = false;
            //break;
            //}
        //// support_{A-B}(-v) = support(A, -v) - support(B, v)
        //unsigned int i1 = support(verts1, N1, -v, qi, vec3<Scalar>());
        //unsigned int i2 = support(verts2, N2, v, qj, Scalar(-1.0)*dr);
        //vec3<Scalar> w = rotate(qi, verts1[i1]) - (rotate(qj, verts2[i2]) + Scalar(-1.0)*dr);

        //// Check termination conditions for degenerate cases:
        //// 1) If we are repeatedly finding the same point but can't get closer
        //// and can't terminate within machine precision.
        //// 2) If we are cycling between two points.
        //// In either case, because of the tracking with W_used, we can
        //// guarantee that the new w will be found in one of the W (but possibly
        //// in one of the unused slots.
        ////
        //// We don't bother with this on the GPU, the resulting warp divergence
        //// is worse for performance than just going through the max number of
        //// iterations on all threads.
//#ifndef NVCC
        //bool degenerate(false);
        //for (unsigned int i = 0; i < max_num_points; ++i)
            //{
            //if (w == W[i])
                //{
                //degenerate = true;
                //break;
                //}
            //}
//#endif

        //Scalar vnorm = sqrt(dot(v, v));
        //Scalar dvw = dot(v, w);
        //Scalar d = dvw/vnorm;
        //u = u > d ? u : d;
        //if (d > 0)
            //{
            //overlap = false;
            //}

//#ifdef NVCC
        //close_enough = ( ((vnorm - u) <= eps*vnorm) || (vnorm < omega) );
//#else
        //close_enough = ( degenerate || ((vnorm - u) <= eps*vnorm) || (vnorm < omega) );
//#endif
        //if (!close_enough)
            //{
            //unsigned int added_element(0);
            //for (; added_element < max_num_points; added_element++)
                //{
                //// At least one of these must be empty, otherwise we have an
                //// overlap. 
                //if (!W_used[added_element])
                    //{
                    //W[added_element] = w;
                    //W_used[added_element] = true;
                    //indices1[added_element] = i1;
                    //indices2[added_element] = i2;
                    //break;
                    //}
                //}
            //bool use_last(false);

            ///////////////////////////////////////////////
            //////////// BEGIN JOHNSON ALGORITHM //////////
            ///////////////////////////////////////////////
            //unsigned int added_index = 1 << added_element;

            //// If there is only one point in use, we can return immediately.
            //unsigned int num_used = 0;
            //for (unsigned int i = 0; i < max_num_points; ++i)
                //{
                //num_used += W_used[i];
                //}
            //if (num_used == 1)
                //{
                //for (unsigned int i = 0; i < max_num_points; ++i)
                    //{
                    //if (W_used[i])
                        //{
                        //deltas[1 << i][i] = 1;
                        //lambdas[i] = 1.0;
                        //}
                    //}
                //}
            //else
                //{
                //// The check_indexes array is simply linearly indexed, and it contains the
                //// subsets of W that we need to test. The sets are inserted as they are
                //// created over the course of the recursive algorithm. The sets are stored
                //// as unsigned integer representations of (ndim+1)-bits. For example, if W
                //// has 4 points, the subset consisting of the first 3 points is stored as
                //// 0111. The current_subset is used to select the current index to check, and
                //// next_subset_slot indicates the next open spot when a new set is found. To
                //// efficiently determine whether a particular subset is new or has been
                //// seen before, we maintain a separate boolean array comb_contains that is
                //// updated as new subsets are found. Note that comb_contains[0] is never
                //// used since it corresponds to the empty set, but is left indexed this way
                //// to simplify the access pattern using the bit-based indexing scheme.
                //unsigned int current_subset = 0, next_subset_slot = 0;
                //unsigned int check_indexes[max_power_set_size - 1] = {0};
                //bool comb_contains[max_power_set_size] = {false};

                //for (unsigned int i = 0; i < max_num_points; ++i)
                    //{
                    //if (W_used[i])
                        //{
                        //unsigned int index_i(1 << i);
                        //check_indexes[next_subset_slot] = index_i;
                        //comb_contains[index_i] = true;
                        //next_subset_slot += 1;

                        //// Base case for recursive algorithm is a set of size 1. While the distance
                        //// subalgorithm is recursive in computing sets of increasing size, the
                        //// outer GJK algorithm is not. Therefore, there is no guarantee that all
                        //// sets of size one will have been set by other calls to the Johnson
                        //// algorithm before getting her, so we need to establish the base case
                        //// explicitly by setting these here.
                        //deltas[index_i][i] = 1;
                        //}
                    //}

                //unsigned int desired_index = max_power_set_size;
                //while (current_subset < next_subset_slot)
                    //{
                    //unsigned int current_index = check_indexes[current_subset];

                    //// The vector k must be fixed, but can be chosen arbitrarily, but it. A
                    //// simple deterministic choice is the first used index use.
                    //unsigned int k = 0;
                    //for (; k < max_num_points; k++)
                        //{
                        //if ((1 << k) & current_index)
                            //{
                            //break;
                            //}
                        //}
                    //bool complete = true;
                    //for (unsigned int new_element = 0; new_element < max_num_points; new_element++)
                        //{
                        //// Add new elements that are in use and not contained in the current set.
                        //if (W_used[new_element] && !(current_index & (1 << new_element)))
                            //{
                            //// Generate the corresponding bit-based index for the new set.
                            //unsigned int new_index = current_index | (1 << new_element);
                            //// The only sets for which we will not have cached data are
                            //// sets that contain the element most recently added to W.
                            //if ((added_index & current_index) || (added_element == new_element))
                                //{
                                //Scalar total = 0;
                                //// Use bitwise checks of all possible elements to find set members
                                //for (unsigned int possible_element = 0; possible_element < max_num_points; possible_element++)
                                    //{
                                    //if ((1 << possible_element) & current_index)
                                        //{
                                         //total += deltas[current_index][possible_element]*(
                                            //dot(W[possible_element], W[k])-dot(W[possible_element], W[new_element]));
                                        //}
                                    //}
                                //deltas[new_index][new_element] = total;
                                //}
                            //if (!comb_contains[new_index])
                                //{
                                //comb_contains[new_index] = true;
                                //check_indexes[next_subset_slot] = new_index;
                                //next_subset_slot += 1;
                                //}

                            //// Part (ii) of termination condition: Delta_j(X U {y_j}) <= 0
                            //// for all j not in current_subset
                            //// Could add additional check beforehand using added_index
                            //if (deltas[new_index][new_element] > 0)
                                //{
                                //complete = false;
                                //}
                            //}
                        //}
                    //// Part (i) of termination condition: Delta_i(X) > 0 for all i in current_subset
                    //// Could add additional check beforehand using added_index
                    //for (unsigned int i = 0; i < max_num_points; ++i)
                        //{
                        //if (((1 << i) & current_index) && (deltas[current_index][i] <= 0))
                            //{
                            //complete = false;
                            //break;
                            //}
                        //}
                    //if (complete)
                        //{
                        //desired_index = current_index;
                        //break;
                        //}
                    //current_subset += 1;
                    //}

                //if (desired_index == max_power_set_size)
                    //{
                    //use_last = true;
                    //}
                //else
                    //{
                    //// The sum of relevant deltas is used to scale the lambdas.
                    //Scalar total(0);
                    //for (unsigned int i = 0; i < max_num_points; ++i)
                        //{
                        //total += deltas[desired_index][i];
                        //}

                    //for (unsigned int i = 0; i < max_num_points; ++i)
                        //{
                        //if ((1 << i) & desired_index)
                            //{
                            //W_used[i] = true;
                            //lambdas[i] = deltas[desired_index][i]/total;
                            //}
                        //else
                            //{
                            //W_used[i] = false;
                            //}
                        //}
                    //}
                //}
            ///////////////////////////////////////////////
            ///////////// END JOHNSON ALGORITHM ///////////
            ///////////////////////////////////////////////

            //if (use_last)
                //{
                //break;
                //}

            //v = vec3<Scalar>();
            //for (unsigned int i = 0; i < max_num_points; ++i)
                //{
                //if (W_used[i])
                    //{
                    //v += lambdas[i]*W[i];
                    //}
                //}
            //}
        //}

    //// If there's an overlap, then we call GJK again, except this time we just
    //// use the center point of each polytope against the other hull to find the
    //// deepest penetration point.
    //if (overlap && !in_recursion)
        //{
        //vec3<Scalar> a1, a2, b1, b2;
        //bool success1, success2, overlap1, overlap2;
        //gjk_with_overlaps<ndim>(pos1, pos2, &pos1, 1, verts2, N2, v1, v1, a1, b1, success1, overlap1, qi, qj, dr, true);
        //gjk_with_overlaps<ndim>(pos1, pos2, verts1, N1, &pos2, 1, v2, v2, a2, b2, success2, overlap2, qi, qj, dr, true);
        //a = a2;
        //b = b1;
//#ifndef NVCC
        //if (overlap1 or overlap2)
            //throw std::runtime_error("The particles are overlapping past their centers, potential won't work.");
        //if ( not (success1 and success2) )
            //throw std::runtime_error("Recursive GJK failed!");
//#endif
        //}
    //else
        //{
        //// A compiler bug causes the for loop below to never terminate when
        //// using max_num_points as the upper bound. Defining a new (equivalent)
        //// variable seems to fix it.
        //constexpr unsigned int new_limit = max_num_points;
        //v1 = v;
        //v2 = v;
        //a = vec3<Scalar>();
        //b = vec3<Scalar>();
        //for (unsigned int i = 0; i < new_limit; ++i)
            //{
            //if (W_used[i])
                //{
                //a += lambdas[i]*rotate(qi, verts1[indices1[i]]);
                //b += lambdas[i]*(rotate(qj, verts2[indices2[i]]) + Scalar(-1.0)*dr);
                //}
            //}
        //}
    //}


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
template <unsigned int ndim>
DEVICE inline void gjk(const ManagedArray<vec3<Scalar> > verts1, const unsigned int &N1, const ManagedArray<vec3<Scalar> > verts2, const unsigned int &N2, vec3<Scalar> &v, vec3<Scalar> &a, vec3<Scalar> &b, bool& success, bool& overlap, const quat<Scalar> &qi, const quat<Scalar> &qj, const vec3<Scalar> &dr)
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
    for(unsigned int i = 0; i < N1; ++i)
        {
        mean1 += rotate(qi, verts1[i]);
        }
    for(unsigned int i = 0; i < N2; ++i)
        {
        mean2 += (rotate(qj, verts2[i]) + Scalar(-1.0)*dr);
        }
    mean1 /= Scalar(N1);
    mean2 /= Scalar(N2);
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

    // The tolerances are compile-time constants.
    constexpr Scalar eps(1e-8), omega(1e-4);

    Scalar u(0);
    bool close_enough(false);
    unsigned int max_iterations = N1 + N2 + 1;
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
        unsigned int i1 = support(verts1, N1, -v, qi, vec3<Scalar>(0, 0, 0));
        unsigned int i2 = support(verts2, N2, v, qj, Scalar(-1.0)*dr);
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
                        deltas[1 << i][i] = 1;
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
                // to simplify the access pattern using the bit-based indexing scheme.
                unsigned int current_subset = 0, next_subset_slot = 0;
                unsigned int check_indexes[max_power_set_size - 1] = {0};
                unsigned long long int comb_contains = 0;

                for (unsigned int i = 0; i < max_num_points; ++i)
                    {
                    if (W_used & (1 << i))
                        {
                        unsigned int index_i(1 << i);
                        check_indexes[next_subset_slot] = index_i;
                        comb_contains |= (1 << index_i);
                        next_subset_slot += 1;

                        // Base case for recursive algorithm is a set of size 1. While the distance
                        // subalgorithm is recursive in computing sets of increasing size, the
                        // outer GJK algorithm is not. Therefore, there is no guarantee that all
                        // sets of size one will have been set by other calls to the Johnson
                        // algorithm before getting her, so we need to establish the base case
                        // explicitly by setting these here.
                        deltas[index_i][i] = 1;
                        }
                    }

                unsigned int desired_index = max_power_set_size;
                while (current_subset < next_subset_slot)
                    {
                    unsigned int current_index = check_indexes[current_subset];

                    // The vector k must be fixed, but can be chosen
                    // arbitrarily. A simple deterministic choice is the first
                    // used index use.
                    unsigned int k = 0;
                    for (; k < max_num_points; ++k)
                        {
                        if ((1 << k) & current_index)
                            {
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

                            Scalar delta_current = 0;

                            // The only sets for which we will not have cached data are
                            // sets that contain the element most recently added to W.
                            if ((added_index & current_index) || (added_element == new_element))
                                {
                                // Caching the W_new value for speed on the GPU.
                                const vec3<Scalar> W_new = W[new_element];

                                // Use bitwise checks of all possible elements to find set members
                                for (unsigned int possible_element = 0; possible_element < max_num_points; ++possible_element)
                                    {
                                    if ((1 << possible_element) & current_index)
                                        {
                                         const vec3<Scalar> W_possible = W[possible_element];
                                         const Scalar dot1 = dot(W_possible, W_k);
                                         const Scalar dot2 = dot(W_possible, W_new);
                                         delta_current += deltas[current_index][possible_element]*(dot1 - dot2);
                                        }
                                    }
                                deltas[new_index][new_element] = delta_current;
                                }
                            else
                                {
                                // Minimize data reads by using delta_current
                                // in the termination conditional below and
                                // only reading from deltas when we are using a
                                // cached value.
                                delta_current = deltas[new_index][new_element];
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
                            if (delta_current > 0)
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
                            if (((1 << i) & current_index) && (deltas[current_index][i] <= 0))
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
