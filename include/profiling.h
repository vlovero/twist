#pragma once

#ifdef TWIST_PROFILING
#include "caliper/cali.h"

#ifdef _MSC_VER
#define TWIST_FUNC_NAME __FUNCSIG__
#else
#define TWIST_FUNC_NAME __PRETTY_FUNCTION__
#endif

#define TWIST_CXX_MARK_FUNCTION cali::Function CALI_CREATE_VAR_NAME(__cali_ann, __func__)(TWIST_FUNC_NAME)
#define TWIST_MARK_BEGIN(name) CALI_MARK_BEGIN(name)
#define TWIST_MARK_END(name) CALI_MARK_END(name)
#else
#define TWIST_CXX_MARK_FUNCTION
#define TWIST_MARK_BEGIN(name)
#define TWIST_MARK_END(name)
#endif