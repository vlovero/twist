#pragma once

#ifndef NDEBUG
#include "caliper/cali.h"

#define TWIST_CXX_MARK_FUNCTION cali::Function CALI_CREATE_VAR_NAME(__cali_ann, __func__)(__PRETTY_FUNCTION__)
#define TWIST_MARK_BEGIN(name) CALI_MARK_BEGIN(name)
#define TWIST_MARK_END(name) CALI_MARK_END(name)
#else
#define TWIST_CXX_MARK_FUNCTION
#define TWIST_MARK_BEGIN(name)
#define TWIST_MARK_END(name)
#endif