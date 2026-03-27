#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NPY_ARRAY_MAX_DIMENSIONS 8


typedef struct _npy_array_t
{
    char *data;
    size_t shape[NPY_ARRAY_MAX_DIMENSIONS];
    int32_t ndim;
    char endianness;
    char typechar;
    size_t elem_size;
    bool fortran_order;
    /* Consider map_addr as a private member. Do modify this pointer! Used for unmap() */
    void *map_addr; /* pointer to the map if array is mmap()'ed -- else NULL */
} npy_array_t;


npy_array_t *npy_array_load(const char *filename);
npy_array_t *npy_array_mmap(const char *filename);
void npy_array_dump(const npy_array_t *m);
void npy_array_save(const char *filename, const npy_array_t *m);
void npy_array_free(npy_array_t *m);

/* _read_matrix() might be public in the future as a macro or something.
   Don't use it now as I will change name of it in case I make it public. */
typedef int64_t (*reader_func)(void *fp, void *buffer, uint64_t nbytes);
npy_array_t *_read_matrix(void *fp, reader_func read_func);


#define _NARG(...) _NARG_(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define _NARG_(...) _ARG_N(__VA_ARGS__)
#define _ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N

#define SHAPE(...) .shape = { __VA_ARGS__ }, .ndim = _NARG(__VA_ARGS__)
#define NPY_ARRAY_BUILDER(_data, _shape, ...)      \
    &(npy_array_t)                                 \
    {                                              \
        .data = (char *)_data, _shape, __VA_ARGS__ \
    }

#define NPY_DTYPE_FLOAT16 .typechar = 'f', .elem_size = 2
#define NPY_DTYPE_FLOAT32 .typechar = 'f', .elem_size = 4
#define NPY_DTYPE_FLOAT64 .typechar = 'f', .elem_size = 8
#define NPY_DTYPE_FLOAT128 .typechar = 'f', .elem_size = 16

#define NPY_DTYPE_INT8 .typechar = 'i', .elem_size = 1
#define NPY_DTYPE_INT16 .typechar = 'i', .elem_size = 2
#define NPY_DTYPE_INT32 .typechar = 'i', .elem_size = 4
#define NPY_DTYPE_INT64 .typechar = 'i', .elem_size = 8

#define NPY_DTYPE_UINT8 .typechar = 'u', .elem_size = 1
#define NPY_DTYPE_UINT16 .typechar = 'u', .elem_size = 2
#define NPY_DTYPE_UINT32 .typechar = 'u', .elem_size = 4
#define NPY_DTYPE_UINT64 .typechar = 'u', .elem_size = 8

#define NPY_DTYPE_COMPLEX64 .typechar = 'c', .elem_size = 8
#define NPY_DTYPE_COMPLEX128 .typechar = 'c', .elem_size = 16
#define NPY_DTYPE_COMPLEX256 .typechar = 'c', .elem_size = 32

#define NPY_DTYPE_BOOL .typechar = 'b', .elem_size = 1
