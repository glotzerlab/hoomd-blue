//
// Created by girard01 on 2/8/23.
//

#ifndef HOOMD_VIRTUALSITEMAPPINGS_H
#define HOOMD_VIRTUALSITEMAPPINGS_H

#include <array>
#include <hoomd/HOOMDMath.h>

#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd::md {

    struct VSMap {
        static constexpr unsigned int n_sites = 0;
        struct param_type{};

        VSMap( uint64_t _site): site(_site) {}
/*
 * The base VSMap should never be used as template param, only its derived structs
 * Derived types should implement the following quantities:
 *
        void decomposeForce(Scalar4* forces, Scalar4* net_forces);

        void decomposeVirial(Scalar* virial,
                             Scalar* net_virial,
                             uint64_t virial_pitch,
                             uint64_t net_virial_pitch,
                             Scalar4* postype,
                             Scalar4* forces);

        void reconstructSite(Scalar4 *position_array);
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
    inline void linearSum(Scalar4* ptr,
                   const Scalar4& f,
                   const std::array<uint64_t, N>& indices,
                   const std::array<Scalar, N>& coefficients){
    for(unsigned char s = 0; s < N; s++){
        atAddScalar4(&(ptr[indices[s]]), f * coefficients[s]);
        }
    }
#endif
    template<std::size_t N>
    inline void linearReconstruction(Scalar4* pos,
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

    inline void DEVICE projectVirial(Scalar* virial,
                       Scalar* net_virial,
                       uint64_t virial_pitch,
                       uint64_t net_virial_pitch,
                       uint64_t source,
                       uint64_t target,
                       Scalar3 f,
                       Scalar3 dr_space){
        Scalar virialxx = net_virial[0 * net_virial_pitch + source];
        Scalar virialxy = net_virial[1 * net_virial_pitch + source];
        Scalar virialxz = net_virial[2 * net_virial_pitch + source];
        Scalar virialyy = net_virial[3 * net_virial_pitch + source];
        Scalar virialyz = net_virial[4 * net_virial_pitch + source];
        Scalar virialzz = net_virial[5 * net_virial_pitch + source];

        // subtract intra-body virial prt
#ifndef __HIPCC__
        virial[0 * virial_pitch + target] += virialxx - f.x * dr_space.x;
        virial[1 * virial_pitch + target] += virialxy - f.x * dr_space.y;
        virial[2 * virial_pitch + target] += virialxz - f.x * dr_space.z;
        virial[3 * virial_pitch + target] += virialyy - f.y * dr_space.y;
        virial[4 * virial_pitch + target] += virialyz - f.y * dr_space.z;
        virial[5 * virial_pitch + target] += virialzz - f.z * dr_space.z;
#else
        atomicAdd(virial + 0 * virial_pitch + target, virialxx - f.x * dr_space.x);
        atomicAdd(virial + 1 * virial_pitch + target, virialxy - f.x * dr_space.y);
        atomicAdd(virial + 2 * virial_pitch + target, virialxz - f.x * dr_space.z);
        atomicAdd(virial + 3 * virial_pitch + target, virialyy - f.y * dr_space.y);
        atomicAdd(virial + 4 * virial_pitch + target, virialyz - f.y * dr_space.z);
        atomicAdd(virial + 5 * virial_pitch + target, virialzz - f.z * dr_space.z);
#endif
    }

    template<std::size_t N>
    inline void projectLinearVirial(Scalar* virial,
                             Scalar* net_virial,
                             uint64_t virial_pitch,
                             uint64_t net_virial_pitch,
                             Scalar4* postype,
                             Scalar4* forces,
                             uint64_t site,
                             const std::array<uint64_t, N>& indices,
                             const std::array<Scalar, N>& coefficients){
        const auto pos_site = postype[site];
        Scalar3 r_site = make_scalar3(pos_site.x, pos_site.y, pos_site.z);
        for(unsigned char s = 0; s < N; s++){
            const auto& f = forces[indices[s]];
            const auto pos = postype[indices[s]];
            const auto r = make_scalar3(pos.x, pos.y, pos.z);
            projectVirial(virial, net_virial, virial_pitch, net_virial_pitch, site, indices[s],
                          make_scalar3(f.x, f.y, f.z) * coefficients[s], r - r_site);
        }
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

        void decomposeForce(Scalar4* forces, Scalar4* net_forces){
            // since a non-virtual particle may be used for construction of multiple
            // virtual sites, we would have to use an atomic op. CPU is serial wrt
            // this particular op.
            Scalar4 vsite_force = net_forces[site];
#ifndef __HIPCC__
            forces[indices[0]] += vsite_force * (1-a);
            forces[indices[1]] += vsite_force * a;
#else
            linearSum(forces, vsite_force, indices, {(1-a), a});
#endif
            net_forces[site] = make_scalar4(0., 0., 0., 0.);
        }

        void decomposeVirial(Scalar* virial,
                             Scalar* net_virial,
                             uint64_t virial_pitch,
                             uint64_t net_virial_pitch,
                             Scalar4* postype,
                             Scalar4* forces) {
            projectLinearVirial(virial,
                                net_virial,
                                virial_pitch,
                                net_virial_pitch,
                                postype,
                                forces,
                                site,
                                indices,
                                {1-a, a});
        };

    protected:
        Scalar a;
        const std::array<uint64_t, n_sites> indices;
    };
}

}
#endif //HOOMD_VIRTUALSITEMAPPINGS_H
