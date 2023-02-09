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

        VSMap(std::array<uint64_t, n_sites> &_indices, uint64_t _site)
                : indices(_indices), site(_site) {}
/*
 * The base VSMap should never be used as template param, only its derived structs
 * Derived types should implement the following quantities:
 *
        void decomposeForce(Scalar4 *force_array) {};

        void decomposeVirial(Scalar *virialArray, uint64_t virial_pitch) {};

        void reconstructSite(Scalar4 *position_array) {};
*/
        const std::array<uint64_t, n_sites> indices;
        const uint64_t site;
    };

namespace virtualsites {
    struct Type2 : public VSMap{
        struct param_type{
            Scalar a;
        };

        Type2(param_type params, std::array<uint64_t, n_sites> &_indices, uint64_t _site):
        VSMap(_indices, _site), a(params.a){}

        void reconstructSite(Scalar4* positions){
            Scalar4 postype_i = positions[indices[0]];
            Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
            Scalar4 postype_j = positions[indices[1]];
            Scalar3 pos_j = make_scalar3(postype_j.x, postype_j.y, postype_j.z);
            Scalar3 pos_virtual = pos_i + (pos_j - pos_i) * a;
            positions[site].x = pos_virtual.x;
            positions[site].y = pos_virtual.y;
            positions[site].z = pos_virtual.z;
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
            // implementation note, we should probably define a function to perform the atomic add on Scalar4 to simplify boilerplate

            atomicAdd(&(forces[indices[0]].x), vsite_force.x * (1-a));
            atomicAdd(&(forces[indices[0]].y), vsite_force.y * (1-a));
            atomicAdd(&(forces[indices[0]].z), vsite_force.z * (1-a));
            atomicAdd(&(forces[indices[0]].w), vsite_force.w * (1-a));

            atomicAdd(&(forces[indices[1]].x), vsite_force.x * a);
            atomicAdd(&(forces[indices[1]].y), vsite_force.y * a);
            atomicAdd(&(forces[indices[1]].z), vsite_force.z * a);
            atomicAdd(&(forces[indices[1]].w), vsite_force.w * a);
#endif
            forces[site] = make_scalar4(0., 0., 0., 0.);
        }

        void decomposeVirial(Scalar *virialArray, uint64_t virial_pitch) {}; // TODO: missing implementation, check required args as well

    protected:
        Scalar a;
    };
}

}
#endif //HOOMD_VIRTUALSITEMAPPINGS_H
