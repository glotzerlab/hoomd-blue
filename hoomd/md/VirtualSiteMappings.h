//
// Created by girard01 on 2/8/23.
//

#ifndef HOOMD_VIRTUALSITEMAPPINGS_H
#define HOOMD_VIRTUALSITEMAPPINGS_H

#include <array>
#include <hoomd/HOOMDMath.h>

namespace hoomd::md {

    struct VSMap {
        static constexpr unsigned int n_sites = 0;
        struct param_type{};

        VSMap( uint64_t _site): site(_site) {}
/*
 * The base VSMap should never be used as template param, only its derived structs
 * Derived types should implement the following quantities:
 *
        void decomposeForce(Scalar4 *force_array) {};

        void decomposeVirial(Scalar *virialArray, uint64_t virial_pitch) {};

        void reconstructSite(Scalar4 *position_array) {};
*/
        const uint64_t site;
    };

namespace virtualsites {

#ifdef __HIPCC__
    __forceinline__ DEVICE void atAddScalar4(Scalar4* x, const Scalar4& y){
        atomicAdd(&(x->x), y.x);
        atomicAdd(&(x->y), y.y);
        atomicAdd(&(x->z), y.z);
        atomicAdd(&(x->w), y.w);
    }

    template<std::size_t N>
    void linearSum(Scalar4* ptr,
                   const Scalar4& f,
                   const std::array<uint64_t, N>& indices,
                   const std::array<Scalar, N>& coefficients){
    for(unsigned char s = 0; s < N; s++){
        atAddScalar4(&(ptr[indices[s]]), f * coefficients[s]);
        }
    }
#endif
    template<std::size_t N>
    void linearReconstruction(Scalar4* pos,
                              const uint64_t site,
                              const std::array<uint64_t, N>& indices,
                              const std::array<Scalar, N>& coefficients){
        Scalar3 site_pos = {0., 0., 0.};
        for(unsigned char s = 0; s < N; s++){
        const auto& ibase = pos[indices[s]];
        site_pos += make_scalar3(ibase.x, ibase.y, ibase.z) * coefficients[s];
        }
        pos[site].x = site_pos.x;
        pos[site].y = site_pos.y;
        pos[site].z = site_pos.z;
    }

    struct Type2 : public VSMap{
        struct param_type{
            Scalar a;
        };
        static constexpr unsigned int n_sites = 2;

        Type2(param_type params, std::array<uint64_t, n_sites> &_indices, uint64_t _site):
        VSMap(_site), a(params.a), indices(_indices){}

        void reconstructSite(Scalar4* positions){
            linearReconstruction(positions, site, indices, {1-a, a});
        }

        void decomposeForce(Scalar4* forces){
            // since a non-virtual particle may be used for construction of multiple
            // virtual sites, we would have to use an atomic op. CPU is serial wrt
            // this particular op.
            Scalar4 vsite_force = forces[site];
#ifndef __HIPCC__
            forces[indices[0]] += vsite_force * (1-a);
            forces[indices[1]] += vsite_force * a;
#else
            linearSum(forces, vsite_force, indices, {(1-a), a});
#endif
            forces[site] = make_scalar4(0., 0., 0., 0.);
        }

        void decomposeVirial(Scalar *virialArray, uint64_t virial_pitch) {}; // TODO: missing implementation, check required args as well

    protected:
        Scalar a;
        const std::array<uint64_t, n_sites> indices;
    };
}

}
#endif //HOOMD_VIRTUALSITEMAPPINGS_H
