//! Name the unit test module
#define BOOST_TEST_MODULE aabb_tree
#include "boost_utf_configure.h"
#include "hoomd/AABBTree.h"

#include <iostream>
#include <algorithm>

#include <boost/bind.hpp>
#include <boost/python.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>


#include "hoomd/VectorMath.h"
#include "hoomd/extern/saruprng.h"

using namespace hpmc;
using namespace hpmc::detail;

bool in(unsigned int i, const std::vector<unsigned int>& v)
    {
    std::vector<unsigned int>::const_iterator it;
    it = std::find(v.begin(), v.end(), i);
    return (it != v.end());
    }

BOOST_AUTO_TEST_CASE( basic )
    {
    // build a simple test AABB tree
    AABB aabbs[3];
    aabbs[0] = AABB(vec3<Scalar>(1,1,-1), vec3<Scalar>(3,3,1));
    aabbs[1] = AABB(vec3<Scalar>(0, 1, -1), vec3<Scalar>(1,5,1));
    aabbs[2] = AABB(vec3<Scalar>(0,0,-1), vec3<Scalar>(1,1,1));

    // construct the tree
    AABBTree tree;
    tree.buildTree(aabbs, 3);

    // try some test queries
    std::vector<unsigned int> hits;

    hits.clear();
    tree.query(hits, AABB(vec3<Scalar>(2,2,0), vec3<Scalar>(2.1, 2.1, 0.1)));
    BOOST_CHECK(in(0, hits));

    hits.clear();
    tree.query(hits, AABB(vec3<Scalar>(0.5,3,0), vec3<Scalar>(0.6, 3.1, 0.1)));
    BOOST_CHECK(in(1, hits));

    hits.clear();
    tree.query(hits, AABB(vec3<Scalar>(0.5,0.5,0), vec3<Scalar>(0.6, 0.6, 0.1)));
    BOOST_CHECK(in(2, hits));

    hits.clear();
    tree.query(hits, AABB(vec3<Scalar>(0.9,0.9,0), vec3<Scalar>(1.1, 1.1, 0.1)));
    BOOST_CHECK_EQUAL(hits.size(), 3);
    }


BOOST_AUTO_TEST_CASE( bigger )
    {
    const unsigned int N = 1000;
    Saru rng(1);

    // build a test AABB tree big enough to exercise the node splitting
    std::vector< vec3<Scalar> > points(N);
    AABB aabbs[N];
    for (unsigned int i = 0; i < N; i++)
        {
        points[i] = vec3<Scalar>(rng.f(), rng.f(), rng.f()) * Scalar(1000);
        aabbs[i] = AABB(points[i], Scalar(1.0));
        }

    // build the tree
    AABBTree tree;
    tree.buildTree(aabbs, N);

    // query each particle to ensure it can be found
    std::vector<unsigned int> hits;

    for (unsigned int i = 0; i < N; i++)
        {
        hits.clear();
        tree.query(hits, AABB(points[i], Scalar(0.01)));
        BOOST_CHECK(in(i, hits));
        }

    // now move all the points with the update method and ensure that they are still found
    for (unsigned int i = 0; i < N; i++)
        {
        points[i] += vec3<Scalar>(rng.f(), rng.f(), rng.f());
        aabbs[i] = AABB(points[i], Scalar(1.0));
        tree.update(i, aabbs[i]);
        }

    for (unsigned int i = 0; i < N; i++)
        {
        hits.clear();
        tree.query(hits, AABB(points[i], Scalar(0.01)));
        BOOST_CHECK(in(i, hits));
        }
    }
