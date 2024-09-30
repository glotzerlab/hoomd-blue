// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _HPMC_COUNTERS_H_
#define _HPMC_COUNTERS_H_

#include "hoomd/HOOMDMath.h"

namespace hoomd
    {
namespace hpmc
    {
/*! \file IntegratorHPMCMonoGPU.cuh
    \brief Declaration of CUDA kernels drivers
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Storage for acceptance counters
/*! \ingroup hpmc_data_structs */
struct hpmc_counters_t
    {
    unsigned long long int translate_accept_count; //!< Count of accepted translation moves
    unsigned long long int translate_reject_count; //!< Count of rejected translation moves
    unsigned long long int rotate_accept_count;    //!< Count of accepted rotation moves
    unsigned long long int rotate_reject_count;    //!< Count of rejected rotation moves
    unsigned long long int overlap_checks;         //!< Count of the number of overlap checks
    unsigned int
        overlap_err_count; //!< Count of the number of times overlap checks encounter errors

    //! Construct a zero set of counters
    DEVICE hpmc_counters_t()
        {
        translate_accept_count = 0;
        translate_reject_count = 0;
        rotate_accept_count = 0;
        rotate_reject_count = 0;
        overlap_checks = 0;
        overlap_err_count = 0;
        }

#ifndef NVCC
    //! Get the translate acceptance
    /*! \returns The number of translation moves that are accepted and rejected.
     */
    std::pair<unsigned long long int, unsigned long long int> getTranslateCounts()
        {
        return std::make_pair(translate_accept_count, translate_reject_count);
        }

    //! Get the rotate acceptance
    /*! \returns The number of rotation moves that are accepted and rejected.
     */
    std::pair<unsigned long long int, unsigned long long int> getRotateCounts()
        {
        return std::make_pair(rotate_accept_count, rotate_reject_count);
        }

//! Get the number of moves
/*! \return The total number of moves
 */
#endif
    DEVICE unsigned long long int getNMoves()
        {
        return translate_accept_count + translate_reject_count + rotate_accept_count
               + rotate_reject_count;
        }
    };

//! Take the difference of two sets of counters
DEVICE inline hpmc_counters_t operator-(const hpmc_counters_t& a, const hpmc_counters_t& b)
    {
    hpmc_counters_t result;
    result.translate_accept_count = a.translate_accept_count - b.translate_accept_count;
    result.rotate_accept_count = a.rotate_accept_count - b.rotate_accept_count;
    result.translate_reject_count = a.translate_reject_count - b.translate_reject_count;
    result.rotate_reject_count = a.rotate_reject_count - b.rotate_reject_count;
    result.overlap_checks = a.overlap_checks - b.overlap_checks;
    result.overlap_err_count = a.overlap_err_count - b.overlap_err_count;
    return result;
    }

//! Sum of two sets of counters
DEVICE inline hpmc_counters_t operator+(const hpmc_counters_t& a, const hpmc_counters_t& b)
    {
    hpmc_counters_t result;
    result.translate_accept_count = a.translate_accept_count + b.translate_accept_count;
    result.rotate_accept_count = a.rotate_accept_count + b.rotate_accept_count;
    result.translate_reject_count = a.translate_reject_count + b.translate_reject_count;
    result.rotate_reject_count = a.rotate_reject_count + b.rotate_reject_count;
    result.overlap_checks = a.overlap_checks + b.overlap_checks;
    result.overlap_err_count = a.overlap_err_count + b.overlap_err_count;
    return result;
    }

//! Storage for NPT acceptance counters
/*! \ingroup hpmc_data_structs */
struct hpmc_boxmc_counters_t
    {
    unsigned long long int volume_accept_count;    //!< Count of accepted volume moves
    unsigned long long int volume_reject_count;    //!< Count of rejected volume moves
    unsigned long long int ln_volume_accept_count; //!< Count of accepted volume moves
    unsigned long long int ln_volume_reject_count; //!< Count of rejected volume moves
    unsigned long long int shear_accept_count;     //!< Count of accepted shear moves
    unsigned long long int shear_reject_count;     //!< Count of rejected shear moves
    unsigned long long int aspect_accept_count;    //!< Count of accepted aspect moves
    unsigned long long int aspect_reject_count;    //!< Count of rejected aspect moves
    //! Construct a zero set of counters
    DEVICE hpmc_boxmc_counters_t()
        {
        volume_accept_count = 0;
        volume_reject_count = 0;
        ln_volume_accept_count = 0;
        ln_volume_reject_count = 0;
        shear_accept_count = 0;
        shear_reject_count = 0;
        aspect_accept_count = 0;
        aspect_reject_count = 0;
        }

    //! Get the ln_shear acceptance
    /*! \returns The ratio of volume moves that are accepted, or 0 if there are no volume moves
     */
    DEVICE double getVolumeAcceptance()
        {
        if (volume_reject_count + volume_accept_count == 0)
            return 0.0;
        else
            return double(volume_accept_count) / double(volume_reject_count + volume_accept_count);
        }

    //! Get the ln(V) acceptance
    /*! \returns The ratio of volume moves that are accepted, or 0 if there are no volume moves
     */
    DEVICE double getLogVolumeAcceptance()
        {
        if (ln_volume_reject_count + ln_volume_accept_count == 0)
            return 0.0;
        else
            return double(ln_volume_accept_count)
                   / double(ln_volume_reject_count + ln_volume_accept_count);
        }

    //! Get the shear acceptance
    /*! \returns The ratio of shear moves that are accepted, or 0 if there are no shear moves
     */
    DEVICE double getShearAcceptance()
        {
        if (shear_reject_count + shear_accept_count == 0)
            return 0.0;
        else
            return double(shear_accept_count) / double(shear_reject_count + shear_accept_count);
        }

    //! Get tje aspect acceptance
    /*! \returns The ratio of aspect moves that are accepted, or 0 if there are no aspect moves
     */
    DEVICE double getAspectAcceptance()
        {
        if (aspect_reject_count + aspect_accept_count == 0)
            return 0.0;
        else
            return double(aspect_accept_count) / double(aspect_reject_count + aspect_accept_count);
        }

    //! Get the number of moves
    /*! \return The total number of moves
     */
    DEVICE unsigned long long int getNMoves()
        {
        return ln_volume_accept_count + ln_volume_reject_count + volume_accept_count
               + volume_reject_count + shear_accept_count + shear_reject_count + aspect_accept_count
               + aspect_reject_count;
        }
    };

//! Take the difference of two sets of counters
DEVICE inline hpmc_boxmc_counters_t operator-(const hpmc_boxmc_counters_t& a,
                                              const hpmc_boxmc_counters_t& b)
    {
    hpmc_boxmc_counters_t result;
    result.volume_accept_count = a.volume_accept_count - b.volume_accept_count;
    result.ln_volume_accept_count = a.ln_volume_accept_count - b.ln_volume_accept_count;
    result.shear_accept_count = a.shear_accept_count - b.shear_accept_count;
    result.aspect_accept_count = a.aspect_accept_count - b.aspect_accept_count;
    result.volume_reject_count = a.volume_reject_count - b.volume_reject_count;
    result.ln_volume_reject_count = a.ln_volume_reject_count - b.ln_volume_reject_count;
    result.shear_reject_count = a.shear_reject_count - b.shear_reject_count;
    result.aspect_reject_count = a.aspect_reject_count - b.aspect_reject_count;
    return result;
    }

//! Storage for implicit depletants acceptance counters
/*! \ingroup hpmc_data_structs */
struct hpmc_implicit_counters_t
    {
    unsigned long long int insert_count;        //!< Count of depletants inserted
    unsigned long long int insert_accept_count; //!< Count of depletants inserted successfully
    unsigned long long int
        insert_accept_count_sq; //!< Squared count of successful insertion attempts per depletant

    //! Construct a zero set of counters
    DEVICE hpmc_implicit_counters_t()
        {
        insert_count = 0;
        insert_accept_count = 0;
        insert_accept_count_sq = 0;
        }
    };

//! Storage for muVT acceptance counters
/*! \ingroup hpmc_data_structs */
struct hpmc_muvt_counters_t
    {
    unsigned long long int insert_accept_count;   //!< Count of accepted insertion moves
    unsigned long long int insert_reject_count;   //!< Count of rejected insertion moves
    unsigned long long int remove_accept_count;   //!< Count of accepted remove moves
    unsigned long long int remove_reject_count;   //!< Count of rejected remove moves
    unsigned long long int exchange_accept_count; //!< Count of accepted exchange moves
    unsigned long long int exchange_reject_count; //!< Count of rejected exchange moves
    unsigned long long int volume_accept_count;   //!< Count of accepted volume moves
    unsigned long long int volume_reject_count;   //!< Count of rejected volume moves

    //! Construct a zero set of counters
    DEVICE hpmc_muvt_counters_t()
        {
        insert_accept_count = 0;
        insert_reject_count = 0;
        remove_accept_count = 0;
        remove_reject_count = 0;
        exchange_accept_count = 0;
        exchange_reject_count = 0;
        volume_accept_count = 0;
        volume_reject_count = 0;
        }

    //! Get the insertion acceptance
    /*! \returns The ratio of insertion moves that are accepted, or 0 if there are no insertion
     * moves
     */
    DEVICE double getInsertAcceptance()
        {
        if (insert_reject_count + insert_accept_count == 0)
            return 0.0;
        else
            return double(insert_accept_count) / double(insert_reject_count + insert_accept_count);
        }

    //! Get the insert acceptance
    /*! \returns The number of insertion moves that are accepted and rejected.
     */
    std::pair<unsigned long long int, unsigned long long int> getInsertCounts()
        {
        return std::make_pair(insert_accept_count, insert_reject_count);
        }

    //! Get the acceptance for removing particles
    /*! \returns The ratio of deletion moves that are accepted, or 0 if there are no deletion moves
     */
    DEVICE double getRemoveAcceptance()
        {
        if (remove_reject_count + remove_accept_count == 0)
            return 0.0;
        else
            return double(remove_accept_count) / double(remove_reject_count + remove_accept_count);
        }

    //! Get the remove acceptance
    /*! \returns The number of removal moves that are accepted and rejected.
     */
    std::pair<unsigned long long int, unsigned long long int> getRemoveCounts()
        {
        return std::make_pair(remove_accept_count, remove_reject_count);
        }

    //! Get the exchange acceptance
    /*! \returns The number of exchange moves that are accepted and rejected.
     */
    std::pair<unsigned long long int, unsigned long long int> getExchangeCounts()
        {
        return std::make_pair(exchange_accept_count, exchange_reject_count);
        }

    //! Get the volume acceptance
    /*! \returns The number of volume moves that are accepted and rejected.
     */
    std::pair<unsigned long long int, unsigned long long int> getVolumeCounts()
        {
        return std::make_pair(volume_accept_count, volume_reject_count);
        }

    //! Get the acceptance for exchanging particle identities
    /*! \returns The ratio of exchange moves that are accepted, or 0 if there are no exchange moves
     */
    DEVICE double getExchangeAcceptance()
        {
        if (exchange_reject_count + exchange_accept_count == 0)
            return 0.0;
        else
            return double(exchange_accept_count)
                   / double(exchange_reject_count + exchange_accept_count);
        }

    //! Get the volume acceptance
    /*! \returns The ratio of volume moves that are accepted, or 0 if there are no volume moves
     */
    DEVICE double getVolumeAcceptance()
        {
        if (volume_reject_count + volume_accept_count == 0)
            return 0.0;
        else
            return double(volume_accept_count) / double(volume_reject_count + volume_accept_count);
        }

    //! Get the number of volume
    /*! \return The total number of moves
     */
    DEVICE unsigned long long int getNVolumeMoves()
        {
        return volume_accept_count + volume_reject_count;
        }

    //! Get the number of moves
    /*! \return The total number of moves
     */
    DEVICE unsigned long long int getNExchangeMoves()
        {
        return insert_accept_count + insert_reject_count + remove_accept_count + remove_reject_count
               + exchange_accept_count + exchange_reject_count;
        }
    };

//! Storage for cluster move acceptance counters
/*! \ingroup hpmc_data_structs */
struct hpmc_clusters_counters_t
    {
    unsigned long long int n_clusters;              //!< Number of constructed clusters
    unsigned long long int n_particles_in_clusters; //!< Number of particles in clusters

    //! Construct a zero set of counters
    DEVICE hpmc_clusters_counters_t()
        {
        n_clusters = 0;
        n_particles_in_clusters = 0;
        }

    //! Returns the number of particle in clusters
    DEVICE unsigned long long int getNParticlesInClusters() const
        {
        return n_particles_in_clusters;
        }

    //! Returns the average cluster size
    DEVICE double getAverageClusterSize() const
        {
        if (n_clusters)
            return (double)getNParticlesInClusters() / (double)n_clusters;
        else
            return 0.0;
        }
    };

struct hpmc_virtual_moves_counters_t
    {
    unsigned long long int accept_count;
    unsigned long long int reject_count;
    unsigned long long int total_num_particles_in_clusters;
    unsigned long long int total_num_moves_attempted;
    

    //! Construct a zero set of counters
    DEVICE hpmc_virtual_moves_counters_t()
        {
        accept_count = 0;
        reject_count = 0;
        total_num_particles_in_clusters = 0;
        total_num_moves_attempted = 0;
        }

    std::pair<unsigned long long int, unsigned long long int> getCounts()
        {
        return std::make_pair(accept_count, reject_count);
        }

    DEVICE double getAverageClusterSize()
        {
        if (accept_count + reject_count)
            {
            return (double)total_num_particles_in_clusters / ((double)accept_count + (double)reject_count);
            }
        else
            {
            return 0.0;
            }
        }
    DEVICE unsigned long long int getTotalNumParticlesInClusters()
        {
        return total_num_particles_in_clusters;
        }

    DEVICE unsigned long long int getTotalNumMovesAttempted()
        {
        return total_num_moves_attempted;
        }
    };

DEVICE inline hpmc_virtual_moves_counters_t operator-(const hpmc_virtual_moves_counters_t& a, hpmc_virtual_moves_counters_t& b)
    {
    hpmc_virtual_moves_counters_t result;
    result.accept_count = a.accept_count - b.accept_count;
    result.reject_count = a.reject_count - b.reject_count;
    result.total_num_particles_in_clusters = a.total_num_particles_in_clusters - b.total_num_particles_in_clusters;
    result.total_num_moves_attempted = a.total_num_moves_attempted - b.total_num_moves_attempted;
    return result;
    }

//! Take the difference of two sets of counters
DEVICE inline hpmc_implicit_counters_t operator-(const hpmc_implicit_counters_t& a,
                                                 const hpmc_implicit_counters_t& b)
    {
    hpmc_implicit_counters_t result;
    result.insert_count = a.insert_count - b.insert_count;
    result.insert_accept_count = a.insert_accept_count - b.insert_accept_count;
    result.insert_accept_count_sq = a.insert_accept_count_sq - b.insert_accept_count_sq;
    return result;
    }

//! Sum of two sets of counters
DEVICE inline hpmc_implicit_counters_t operator+(const hpmc_implicit_counters_t& a,
                                                 const hpmc_implicit_counters_t& b)
    {
    hpmc_implicit_counters_t result;
    result.insert_count = a.insert_count + b.insert_count;
    result.insert_accept_count = a.insert_accept_count + b.insert_accept_count;
    result.insert_accept_count_sq = a.insert_accept_count_sq + b.insert_accept_count_sq;
    return result;
    }

DEVICE inline hpmc_muvt_counters_t operator-(const hpmc_muvt_counters_t& a,
                                             const hpmc_muvt_counters_t& b)
    {
    hpmc_muvt_counters_t result;
    result.insert_accept_count = a.insert_accept_count - b.insert_accept_count;
    result.remove_accept_count = a.remove_accept_count - b.remove_accept_count;
    result.exchange_accept_count = a.exchange_accept_count - b.exchange_accept_count;
    result.volume_accept_count = a.volume_accept_count - b.volume_accept_count;
    result.insert_reject_count = a.insert_reject_count - b.insert_reject_count;
    result.remove_reject_count = a.remove_reject_count - b.remove_reject_count;
    result.exchange_reject_count = a.exchange_reject_count - b.exchange_reject_count;
    result.volume_reject_count = a.volume_reject_count - b.volume_reject_count;
    return result;
    }

//! Take the difference of two sets of counters
DEVICE inline hpmc_clusters_counters_t operator-(const hpmc_clusters_counters_t& a,
                                                 const hpmc_clusters_counters_t& b)
    {
    hpmc_clusters_counters_t result;
    result.n_clusters = a.n_clusters - b.n_clusters;
    result.n_particles_in_clusters = a.n_particles_in_clusters - b.n_particles_in_clusters;

    return result;
    }

//! Storage for event chain statistics (in addition to normal HPMC counters)
/*! \ingroup hpmc_data_structs */
struct hpmc_nec_counters_t
    {
    unsigned long long int chain_start_count;        //!< Count of chains started
    unsigned long long int chain_at_collision_count; //!< Count of collisions in a chain
    unsigned long long int
        chain_no_collision_count; //!< Count of updates, that did not end up in a collision (end of
                                  //!< chain or beyond search distance)
    unsigned long long int distance_queries; //!< Count of the number of distance queries
    unsigned int
        overlap_err_count; //!< Count of the number of times overlap checks encounter errors

    //! Construct a zero set of counters
    DEVICE hpmc_nec_counters_t()
        {
        chain_start_count = 0;
        chain_at_collision_count = 0;
        chain_no_collision_count = 0;
        distance_queries = 0;
        overlap_err_count = 0;
        }

    //! Get the number of moves
    /*! \return The total number of moves
     */
    DEVICE unsigned long long int getNMoves()
        {
        return chain_at_collision_count + chain_no_collision_count;
        }
    };

//! Take the difference of two sets of counters
DEVICE inline hpmc_nec_counters_t operator-(const hpmc_nec_counters_t& a,
                                            const hpmc_nec_counters_t& b)
    {
    hpmc_nec_counters_t result;
    result.chain_start_count = a.chain_start_count - b.chain_start_count;
    result.chain_at_collision_count = a.chain_at_collision_count - b.chain_at_collision_count;
    result.chain_no_collision_count = a.chain_no_collision_count - b.chain_no_collision_count;
    result.distance_queries = a.distance_queries - b.distance_queries;
    result.overlap_err_count = a.overlap_err_count - b.overlap_err_count;
    return result;
    }

//! Sum of two sets of counters
DEVICE inline hpmc_nec_counters_t operator+(const hpmc_nec_counters_t& a,
                                            const hpmc_nec_counters_t& b)
    {
    hpmc_nec_counters_t result;
    result.chain_start_count = a.chain_start_count + b.chain_start_count;
    result.chain_at_collision_count = a.chain_at_collision_count + b.chain_at_collision_count;
    result.chain_no_collision_count = a.chain_no_collision_count + b.chain_no_collision_count;
    result.distance_queries = a.distance_queries + b.distance_queries;
    result.overlap_err_count = a.overlap_err_count + b.overlap_err_count;
    return result;
    }

    } // end namespace hpmc
    } // end namespace hoomd

#endif // _HPMC_COUNTERS_H_
