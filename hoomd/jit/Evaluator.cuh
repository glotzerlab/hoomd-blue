#pragma once

#include "hoomd/VectorMath.h"

//! Signature for energy evaluators
typedef float (*eval_func)(const vec3<float>& r_ij,
    const unsigned int typ_i,
    const quat<float>& orientation_i,
    const float diameter_i,
    const float charge_i,
    const unsigned int typ_j,
    const quat<float>& orientation_j,
    const float diameter_j,
    const float charge_j);
