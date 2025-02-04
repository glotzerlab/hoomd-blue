// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/VectorMath.h"

//! Declaration of evaluator function
__device__ float eval(const hoomd::vec3<float>& r_ij,
                      unsigned int type_i,
                      const hoomd::quat<float>& q_i,
                      float d_i,
                      float charge_i,
                      unsigned int type_j,
                      const hoomd::quat<float>& q_j,
                      float d_j,
                      float charge_j);

//! Function pointer type
typedef float (*eval_func)(const hoomd::vec3<float>& r_ij,
                           const unsigned int typ_i,
                           const hoomd::quat<float>& orientation_i,
                           const float diameter_i,
                           const float charge_i,
                           const unsigned int typ_j,
                           const hoomd::quat<float>& orientation_j,
                           const float diameter_j,
                           const float charge_j);
