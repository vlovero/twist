cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

project(twist
    VERSION 0.1.0
    DESCRIPTION "Traveling Wave Analysis Tools"
    LANGUAGES CXX
)

include(CheckCXXCompilerFlag)
include(CheckLanguage)

check_language(Fortran)
check_cxx_compiler_flag(-Wno-vla-cxx-extension TWIST_CAN_DISABLE_VLA_WARNING)

if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    message("Found and enabled fortran")
endif()

# ####################################################################################
# Universal definitions for everything ###############################################
# ####################################################################################
option(TWIST_SANITIZE OFF)
option(TWIST_THREAD_SANITIZE OFF)
option(BUILD_BENCHMARKS OFF)
option(BUILD_TESTS OFF)
option(WITH_WAVE_NUMBER OFF)
option(TWIST_GLOBAL_INSTALL ON)
option(WITH_LOCAL_CONTINUATION OFF)
option(USE_BETTER_AUX OFF)

set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/bin")

if(TWIST_SANITIZE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=bounds")
elseif(TWIST_THREAD_SANITIZE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fdiagnostics-color=always -Wall -Wextra")
if (TWIST_CAN_DISABLE_VLA_WARNING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-vla-cxx-extension")
endif()

if((UNIX AND NOT APPLE) OR MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--start-group")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" OR "${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ####################################################################################
# collect all static libs ############################################################
# ####################################################################################
file(GLOB_RECURSE SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
set(static_libs
    ${CMAKE_SOURCE_DIR}/build/lib/libcholmod.a
    ${CMAKE_SOURCE_DIR}/build/lib/libamd.a
    ${CMAKE_SOURCE_DIR}/build/lib/libcamd.a
    ${CMAKE_SOURCE_DIR}/build/lib/libccolamd.a
    ${CMAKE_SOURCE_DIR}/build/lib/libcolamd.a
    ${CMAKE_SOURCE_DIR}/build/lib/libfmt.a
    ${CMAKE_SOURCE_DIR}/build/lib/libgmp.a
    ${CMAKE_SOURCE_DIR}/build/lib/libhdf5.a
    ${CMAKE_SOURCE_DIR}/build/lib/libhdf5_cpp.a
    ${CMAKE_SOURCE_DIR}/build/lib/libhdf5_hl.a
    ${CMAKE_SOURCE_DIR}/build/lib/libhdf5_hl_cpp.a
    ${CMAKE_SOURCE_DIR}/build/lib/libhdf5_tools.a
    ${CMAKE_SOURCE_DIR}/build/lib/libopenblas.a
    ${CMAKE_SOURCE_DIR}/build/lib/libsimdjson.a
    ${CMAKE_SOURCE_DIR}/build/lib/libsuitesparseconfig.a
    ${CMAKE_SOURCE_DIR}/build/lib/libsymengine.a
    ${CMAKE_SOURCE_DIR}/build/lib/libumfpack.a
    ${CMAKE_SOURCE_DIR}/build/lib/libfftw3.a
    ${CMAKE_SOURCE_DIR}/build/lib/libcaliper.a
)

# ####################################################################################
# define main libs and execs #########################################################
# ####################################################################################
find_package(OpenMP REQUIRED)

message("Fortran directories: ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")
message("Fortran libraries  : ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}")

set(SCRIPT_FILES visualize plot interim animate simviz)

foreach(script_file ${SCRIPT_FILES})
    file(READ "${CMAKE_SOURCE_DIR}/include/_python_scripts/_${script_file}.py" FILE_CONTENT)
    set(OUTPUT_FILE "${CMAKE_SOURCE_DIR}/src/misc/${script_file}_script.cpp")
    configure_file(
        "${CMAKE_SOURCE_DIR}/src/misc/${script_file}_script.cpp.in"
        ${OUTPUT_FILE}
        @ONLY
    )
    list(APPEND SRC_FILES ${OUTPUT_FILE})
endforeach()

add_executable(twist "${CMAKE_SOURCE_DIR}/twist.cpp")
add_library(_twistlib STATIC ${SRC_FILES})

target_compile_definitions(_twistlib PUBLIC PYTHON_EXEC="${PYTHON_EXEC}")

if(WITH_WAVE_NUMBER)
    target_compile_definitions(_twistlib PUBLIC TEST_WAVE_NUMBER_PARAM)
endif()
    
if (WITH_LOCAL_CONTINUATION)
    target_compile_definitions(_twistlib PUBLIC TWIST_INCLUDE_LOCAL_CONTINUATION)
endif()

if (USE_BETTER_AUX)
    target_compile_definitions(_twistlib PUBLIC USE_BETTER_AUX)
endif()

# libs/includes for twist lib
target_link_directories(_twistlib PUBLIC ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(_twistlib ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
target_link_libraries(_twistlib OpenMP::OpenMP_CXX)
target_link_libraries(_twistlib ${static_libs})

# target_link_libraries(_twistlib ${LIBGFORTRAN_PATH})
target_include_directories(_twistlib PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_include_directories(_twistlib PUBLIC "${CMAKE_SOURCE_DIR}/build/include")
target_include_directories(_twistlib PUBLIC "${CMAKE_SOURCE_DIR}/build/include/magic_enum")

# link/includes for twist executable
target_link_libraries(twist PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(twist PRIVATE _twistlib)
target_include_directories(twist PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_include_directories(twist PUBLIC "${CMAKE_SOURCE_DIR}/build/include")
target_include_directories(twist PUBLIC "${CMAKE_SOURCE_DIR}/build/include/magic_enum")

if(TWIST_GLOBAL_INSTALL AND NOT WIN32)
    set_target_properties(twist PROPERTIES RUNTIME_OUTPUT_DIRECTORY "/usr/local/bin")
endif()

install(TARGETS twist)

# ####################################################################################
# define any test execs ##############################################################
# ####################################################################################
if(BUILD_TESTS)
    if(NOT DEFINED ENV{TWIST_TESTS})
        # compile ALL tests
        set(WHICH_TESTS
            roc
            test_colmat
            test_dggev4
            test_N
            test_twist
        )
    else()
        set(WHICH_TESTS $ENV{TWIST_TESTS})
    endif()

    foreach(test_name ${WHICH_TESTS})
        if(NOT EXISTS "${CMAKE_SOURCE_DIR}/test/${test_name}.cpp")
            message("${test_name} is not a valid test")
            continue()
        endif()

        add_executable(${test_name} "${CMAKE_SOURCE_DIR}/test/${test_name}.cpp")

        # target_compile_options(${test_name} PRIVATE -O3)
        target_compile_definitions(${test_name} PUBLIC PYTHON_EXEC="${PYTHON_EXEC}")
        target_link_libraries(${test_name} OpenMP::OpenMP_CXX)
        target_link_libraries(${test_name} ${static_libs})
        target_link_libraries(${test_name} _twistlib)
        target_include_directories(${test_name} PUBLIC "${CMAKE_SOURCE_DIR}/include")
        target_include_directories(${test_name} PUBLIC "${CMAKE_SOURCE_DIR}/build/include")
        target_include_directories(${test_name} PUBLIC "${CMAKE_SOURCE_DIR}/build/include/magic_enum")
    endforeach()
endif()

# ####################################################################################
# define any benchmark execs #########################################################
# ####################################################################################
if(BUILD_BENCHMARKS)
    if(NOT DEFINED ENV{TWIST_BENCHMARKS})
        # compile ALL benchmarks
        set(WHICH_BENCHMARKS
            bench_dggev
            bench_lu
            bench_colmat
            bench_dlatrn
            bench_sparsemat_growth
            bench_scalability
        )
    else()
        set(WHICH_BENCHMARKS $ENV{TWIST_BENCHMARKS})
    endif()

    foreach(benchmark_name ${WHICH_BENCHMARKS})
        if(NOT EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/${benchmark_name}.cpp")
            message("${benchmark_name} is not a valid benchmark")
            continue()
        endif()

        add_executable(${benchmark_name} "${CMAKE_SOURCE_DIR}/benchmarks/${benchmark_name}.cpp")
        target_compile_definitions(${benchmark_name} PUBLIC PYTHON_EXEC="${PYTHON_EXEC}")
        target_link_libraries(${benchmark_name} OpenMP::OpenMP_CXX)
        target_link_libraries(${benchmark_name} ${static_libs})
        target_link_libraries(${benchmark_name} _twistlib)
        target_link_libraries(${benchmark_name} ${CMAKE_SOURCE_DIR}/build/lib/libbenchmark.a)
        target_include_directories(${benchmark_name} PUBLIC "${CMAKE_SOURCE_DIR}/include")
        target_include_directories(${benchmark_name} PUBLIC "${CMAKE_SOURCE_DIR}/build/include")
        target_include_directories(${benchmark_name} PUBLIC "${CMAKE_SOURCE_DIR}/build/include/magic_enum")
    endforeach()
endif()