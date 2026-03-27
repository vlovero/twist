#pragma once

#include <array>
#include <cstddef>
#include <tuple>

#include "GL/gl1.h"
#include "GL/gl10.h"
#include "GL/gl11.h"
#include "GL/gl12.h"
#include "GL/gl13.h"
#include "GL/gl14.h"
#include "GL/gl15.h"
#include "GL/gl16.h"
#include "GL/gl2.h"
#include "GL/gl3.h"
#include "GL/gl4.h"
#include "GL/gl5.h"
#include "GL/gl6.h"
#include "GL/gl7.h"
#include "GL/gl8.h"
#include "GL/gl9.h"

typedef std::tuple<size_t, const double *, const double *, const double *, const double *, const double *> _method_t;
constexpr std::array<_method_t, 17> GL_methods = {
    _method_t{ 0, NULL, NULL, NULL, NULL, NULL },                                                                                                   // for easier indexing
    _method_t{ TWIST::GL1::c_nstages, &TWIST::GL1::c_A[0], &TWIST::GL1::c_b[0], &TWIST::GL1::c_c[0], &TWIST::GL1::c_P[0], &TWIST::GL1::c_chat[0] },       // GL1
    _method_t{ TWIST::GL2::c_nstages, &TWIST::GL2::c_A[0], &TWIST::GL2::c_b[0], &TWIST::GL2::c_c[0], &TWIST::GL2::c_P[0], &TWIST::GL2::c_chat[0] },       // GL2
    _method_t{ TWIST::GL3::c_nstages, &TWIST::GL3::c_A[0], &TWIST::GL3::c_b[0], &TWIST::GL3::c_c[0], &TWIST::GL3::c_P[0], &TWIST::GL3::c_chat[0] },       // GL3
    _method_t{ TWIST::GL4::c_nstages, &TWIST::GL4::c_A[0], &TWIST::GL4::c_b[0], &TWIST::GL4::c_c[0], &TWIST::GL4::c_P[0], &TWIST::GL4::c_chat[0] },       // GL4
    _method_t{ TWIST::GL5::c_nstages, &TWIST::GL5::c_A[0], &TWIST::GL5::c_b[0], &TWIST::GL5::c_c[0], &TWIST::GL5::c_P[0], &TWIST::GL5::c_chat[0] },       // GL5
    _method_t{ TWIST::GL6::c_nstages, &TWIST::GL6::c_A[0], &TWIST::GL6::c_b[0], &TWIST::GL6::c_c[0], &TWIST::GL6::c_P[0], &TWIST::GL6::c_chat[0] },       // GL6
    _method_t{ TWIST::GL7::c_nstages, &TWIST::GL7::c_A[0], &TWIST::GL7::c_b[0], &TWIST::GL7::c_c[0], &TWIST::GL7::c_P[0], &TWIST::GL7::c_chat[0] },       // GL7
    _method_t{ TWIST::GL8::c_nstages, &TWIST::GL8::c_A[0], &TWIST::GL8::c_b[0], &TWIST::GL8::c_c[0], &TWIST::GL8::c_P[0], &TWIST::GL8::c_chat[0] },       // GL8
    _method_t{ TWIST::GL9::c_nstages, &TWIST::GL9::c_A[0], &TWIST::GL9::c_b[0], &TWIST::GL9::c_c[0], &TWIST::GL9::c_P[0], &TWIST::GL9::c_chat[0] },       // GL9
    _method_t{ TWIST::GL10::c_nstages, &TWIST::GL10::c_A[0], &TWIST::GL10::c_b[0], &TWIST::GL10::c_c[0], &TWIST::GL10::c_P[0], &TWIST::GL10::c_chat[0] }, // GL10
    _method_t{ TWIST::GL11::c_nstages, &TWIST::GL11::c_A[0], &TWIST::GL11::c_b[0], &TWIST::GL11::c_c[0], &TWIST::GL11::c_P[0], &TWIST::GL11::c_chat[0] }, // GL11
    _method_t{ TWIST::GL12::c_nstages, &TWIST::GL12::c_A[0], &TWIST::GL12::c_b[0], &TWIST::GL12::c_c[0], &TWIST::GL12::c_P[0], &TWIST::GL12::c_chat[0] }, // GL12
    _method_t{ TWIST::GL13::c_nstages, &TWIST::GL13::c_A[0], &TWIST::GL13::c_b[0], &TWIST::GL13::c_c[0], &TWIST::GL13::c_P[0], &TWIST::GL13::c_chat[0] }, // GL13
    _method_t{ TWIST::GL14::c_nstages, &TWIST::GL14::c_A[0], &TWIST::GL14::c_b[0], &TWIST::GL14::c_c[0], &TWIST::GL14::c_P[0], &TWIST::GL14::c_chat[0] }, // GL14
    _method_t{ TWIST::GL15::c_nstages, &TWIST::GL15::c_A[0], &TWIST::GL15::c_b[0], &TWIST::GL15::c_c[0], &TWIST::GL15::c_P[0], &TWIST::GL15::c_chat[0] }, // GL15
    _method_t{ TWIST::GL16::c_nstages, &TWIST::GL16::c_A[0], &TWIST::GL16::c_b[0], &TWIST::GL16::c_c[0], &TWIST::GL16::c_P[0], &TWIST::GL16::c_chat[0] }  // GL16
};