function build_dependency {
    grep -F ${1} successful_dependencies.txt &>/dev/null
    if [[ $? -ne 0 ]]; then
        cd ${1}
        mkdir build
        cd build
        cmake ${@: 2}
        if [[ $? -ne 0 ]]; then
            exit 1
        fi
        
        if [[ $TWIST_ENABLE_PARALLEL_BUILD -eq 1 ]]; then
            # only enable parallel builds when ninja is available
            # make frequently crashes when building in parallel
            cmake --build . --target install --parallel
        else
            cmake --build . --target install
        fi

        if [[ $? -ne 0 ]]; then
            exit 1
        fi
        cd ../..
        echo -e "${1}" >> successful_dependencies.txt
    else
        echo -e "\033[1;31m[${1}]\033[0m has already been built"
    fi
}

function check_available_command {
    command -v $1 &>/dev/null
    if [[ $? -ne 0 ]]; then
        echo -e "\033[31mError:\033[0m ${1} wasn't found. Make sure ${1} executable is in your path."
        exit 1
    fi
    echo -e "\033[1;31m[${1}]\033[0m was found"
}

mkdir build
cd build

PREFIX=`pwd`

# default value(s) for cmake stuff
TWIST_SANITIZE=OFF
TWIST_THREAD_SANITIZE=OFF
WITH_WAVE_NUMBER=OFF
TWIST_GLOBAL_INSTALL=ON
TWIST_LOCAL_OPENMP=0
TWIST_REMOVE_ALL_GIT_DEPENDENCIES=0
TWIST_ENABLE_PARALLEL_BUILD=0

BUILD_TYPE=Debug

for i in "$@"; do
    case "$1" in
        --with-benchmarks)
            shift
            BUILD_BENCHMARKS=ON
            git clone --depth=1 -b "v1.9.4" https://github.com/google/benchmark.git
            build_dependency benchmark \
                -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
                -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} ..
            ;;
        --with-tests)
            shift
            BUILD_TESTS=ON
            ;;
        --with-sanitizers)
            shift
            TWIST_SANITIZE=ON
            ;;
        --with-thread-sanitizers)
            shift
            TWIST_THREAD_SANITIZE=ON
            ;;
        --with-wave-number)
            shift
            WITH_WAVE_NUMBER=ON
            ;;
        --release)
            shift
            BUILD_TYPE=Release
            ;;
        --small)
            shift
            BUILD_TYPE=MinSizeRel
            ;;
        --optimized)
            shift
            BUILD_TYPE=Release
            ;;
        --profiling)
            shift
            BUILD_TYPE=RelWithDebInfo
            ;;
        --local-install)
            shift
            TWIST_GLOBAL_INSTALL=OFF
            ;;
        --local-openmp)
            shift
            TWIST_LOCAL_OPENMP=1
            ;;
        --remove-git-dependencies)
            shift
            TWIST_REMOVE_ALL_GIT_DEPENDENCIES=1
            ;;
        -*|--*)
			echo "Unknown option $i"
			exit 1
			;;
    esac
done

TWIST_ALL_GIT_DEPENDENCIES=("benchmark" "openmp" "Caliper" "fftw" "OpenBLAS" "indicators" "argparse" "ordered-map" "symengine" "hdf5" "simdjson" "SuiteSparse" "fmt")

# download all dependencies
grep -F "__downloaded__" successful_dependencies.txt &>/dev/null
if [[ $? -ne 0 ]]; then
    git clone --depth=1 https://github.com/vlovero/OpenBLAS.git
    git clone --depth=1 https://github.com/vlovero/argparse.git
    git clone --depth=1 -b "v7.8.2" https://github.com/DrTimothyAldenDavis/SuiteSparse.git
    git clone --depth=1 -b "11.0.2" https://github.com/fmtlib/fmt.git
    git clone --depth=1 -b "v3.10.1" https://github.com/simdjson/simdjson.git
    git clone --depth=1 -b "hdf5_1.14.5" https://github.com/HDFGroup/hdf5.git
    git clone --depth=1 -b "v0.14.0" https://github.com/symengine/symengine.git
    git clone --depth=1 -b "v1.1.0" https://github.com/Tessil/ordered-map.git
    git clone --depth=1 -b "v2.3" https://github.com/p-ranav/indicators.git
    git clone --depth=1 -b "v0.9.6" https://github.com/Neargye/magic_enum.git
    git clone --depth=1 https://github.com/LLNL/Caliper.git

    # gmp isn't on github so use wget and extract it
    wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
    tar -xf gmp-6.3.0.tar.xz
    # zlib isn't standard on windows so just download local lib
    # for cmake to build with hdf5
    wget -P hdf5 https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz

    # the github fftw needs to much preprocessing so
    # just download the ready to go version from the
    # website
    wget https://fftw.org/fftw-3.3.10.tar.gz
    tar -xzf fftw-3.3.10.tar.gz
    mv fftw-3.3.10 fftw
    
    echo -e "__downloaded__" >> successful_dependencies.txt
else
    echo "all depencies have been downloaded"
fi

# this is to get rid of linking warnings on mac
if [[ $(uname) == "Darwin" ]]; then
    export MACOSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion)
    export OpenMP_ROOT=$(brew --prefix libomp)
fi

# use openmp when building openblas
export USE_OPENMP=1

# check for cmake
check_available_command cmake
check_available_command wget
check_available_command git

# use ninja if available
command -v ninja &>/dev/null
if [[ $? -eq 0 ]]; then
    export CLICOLOR_FORCE=1
    export CMAKE_GENERATOR=Ninja
    TWIST_ENABLE_PARALLEL_BUILD=1
fi


if [[ $TWIST_LOCAL_OPENMP -eq 1 ]]; then
    git clone -n --depth=1 --filter=tree:0 https://github.com/llvm/llvm-project.git
    if [[ $? -eq 0 ]]; then
        cd llvm-project
        git sparse-checkout set --no-cone /openmp /cmake
        git checkout
        cd ..
    fi
    cd llvm-project
    build_dependency openmp \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} -DBUILD_SHARED_LIBS=OFF \
        -DLIBOMP_ENABLE_SHARED=OFF ..
    export OpenMP_ROOT=$PREFIX
    cd ..
fi

# check for OpenMP
cd ../checks/openmp
cmake .
if [[ $? -ne 0 ]]; then
    rm -rf CMakeFiles
    rm cmake_install.cmake
    rm CMakeCache.txt
    echo -e "\033[31mError:\033[0m cmake couldn't find OpenMP. Make sure it is installed."
    exit 1
fi

# go back to main build directory
cd ../../build

grep -F gmp successful_dependencies.txt &>/dev/null
if [[ $? -ne 0 ]]; then
    cd gmp-6.3.0
    ./configure --prefix=$PREFIX --with-pic
    make -j24
    make install
    rm ${PREFIX}/lib/libgmp*.so
    rm ${PREFIX}/lib/libgmp*.dylib
    cd ..
    echo -e "gmp" >> successful_dependencies.txt
else
    echo -e "\033[1;31m[gmp]\033[0m has already been built"
fi

grep -F magic_enum successful_dependencies.txt &>/dev/null
if [[ $? -ne 0 ]]; then
    mkdir -p include
    cp -r magic_enum/include/magic_enum include/
    echo -e "magic_enum" >> successful_dependencies.txt
else
    echo -e "\033[1;31m[magic_enum]\033[0m has already been built"
fi

build_dependency Caliper \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} -DBUILD_SHARED_LIBS=OFF ..

build_dependency fftw \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF \
    -DENABLE_SSE=ON -DENABLE_SSE2=ON -DENABLE_AVX=ON -DENABLE_AVX2=ON -DDISABLE_FORTRAN=ON \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..

build_dependency OpenBLAS \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..

build_dependency indicators \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_CXX_STANDARD=20 -DINDICATORS_SAMPLES=OFF -DINDICATORS_DEMO=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} ..

build_dependency argparse \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DARGPARSE_BUILD_SAMPLES=off -DARGPARSE_BUILD_TESTS=off -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} ..

build_dependency ordered-map \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} ..

build_dependency symengine \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DBUILD_TESTS=no -DBUILD_BENCHMARKS=no -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON \
    -DCMAKE_FIND_USE_PACKAGE_ROOT_PATH=1 -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=0 -DGMP_ROOT=${PREFIX} \
    -DCMAKE_PREFIX_PATH=${PREFIX} -DGMP_INCLUDE_DIRS=${PREFIX}/include -DGMP_LIBRARIES=${PREFIX}/lib/libgmp.a \
    -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} \
    ..

build_dependency hdf5 \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DHDF5_BUILD_CPP_LIB=ON -DBUILD_TESTING:BOOL=OFF -DHDF5_BUILD_TOOLS:BOOL=ON -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} \
    -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" \
    -DZLIB_TGZ_NAME=${PREFIX}/hdf5/zlib-1.3.1.tar.gz \
    -DHDF5_ENABLE_SZIP_SUPPORT:BOOL=OFF \
    -DHDF5_USE_ZLIB_STATIC:BOOL=OFF \
    -DZLIB_USE_LOCALCONTENT:BOOL=ON \
    -DTGZPATH:PATH=${PREFIX}/hdf5 \
    ..

build_dependency simdjson \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DSIMDJSON_ENABLE_THREADS=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} ..

build_dependency SuiteSparse \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DBLA_STATIC=ON -DLAPACK_LIBRARIES=${PREFIX}/lib/libopenblas.a -DBLAS_LIBRARIES=${PREFIX}/lib/libopenblas.a \
    -DLAPACK_INCLUDE_DIRS=${PREFIX}/include/openblas  -DBLAS_INCLUDE_DIRS=${PREFIX}/include/openblas \
    -DSUITESPARSE_ENABLE_PROJECTS="umfpack" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} ..

build_dependency fmt \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DFMT_TEST=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} ..

# check if python exists
if [[ ! -v PYTHON_EXEC ]]; then
    PYTHON_EXEC=python3
elif [[ -z "$PYTHON_EXEC" ]]; then
    PYTHON_EXEC=python3
else
    echo "PYTHON_EXEC has the value: $PYTHON_EXEC"
fi

command -v ${PYTHON_EXEC} &>/dev/null
if [[ $? -ne 0 ]]; then
    echo "python interpretter could not be found"
    exit 1
fi
PYTHON_EXEC=`${PYTHON_EXEC} -c "import sys; print(sys.executable)"`

# try to import matplotlib
${PYTHON_EXEC} -c "import matplotlib" &>/dev/null
if [[ $? -eq 0 ]]; then
    echo "matplotlib was found on ${PYTHON_EXEC}"
else
    echo -e "\033[31mError:\033[0m matplotlib is not available on $(command -v python3)"
    echo "Try setting the PYTHON_EXEC environment variable to a python interpreter path that has matplotlib and h5py installed."
    exit 1
fi

# try to import h5py
${PYTHON_EXEC} -c "import h5py" &>/dev/null
if [[ $? -eq 0 ]]; then
    echo "h5py was found on ${PYTHON_EXEC}"
else
    echo -e "\033[31mError:\033[0m h5py is not available on $(command -v python3)"
    echo "Try setting the PYTHON_EXEC environment variable to a python interpreter path that has matplotlib and h5py installed."
    exit 1
fi

mkdir TWIST_${BUILD_TYPE}
cd TWIST_${BUILD_TYPE}

cmake \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} \
    -DPYTHON_EXEC=${PYTHON_EXEC} -DBUILD_BENCHMARKS=$BUILD_BENCHMARKS \
    -DTWIST_SANITIZE=$TWIST_SANITIZE -DWITH_WAVE_NUMBER=$WITH_WAVE_NUMBER \
    -DCMAKE_POLICY_VERSION_MINIMUM=4.0 -DBUILD_TESTS=$BUILD_TESTS \
    -DTWIST_GLOBAL_INSTALL=$TWIST_GLOBAL_INSTALL -DTWIST_THREAD_SANITIZE=$TWIST_THREAD_SANITIZE \
    ../.. 
cmake --build . --target install --parallel
mv compile_commands.json ../compile_commands.json

if [[ $TWIST_REMOVE_ALL_GIT_DEPENDENCIES -eq 1 ]]; then
    cd $PREFIX
    for depency in "${TWIST_ALL_GIT_DEPENDENCIES[@]}"; do
        echo "Deleting: $depency"
        rm -rf $depency
    done
fi

if [[ $(uname) == "Darwin" ]]; then
    unset MACOSX_DEPLOYMENT_TARGET
fi