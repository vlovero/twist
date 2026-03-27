import os
import sys
import pathlib
import shutil
import platform
import tarfile
from time import sleep
from glob import glob
from argparse import ArgumentParser, RawTextHelpFormatter
from subprocess import Popen, DEVNULL, check_output
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager


RESET = "\033[0m"
RED = "\033[31m"
BOLD_RED = "\033[1;31m"
CMAKE_ONOFF = ("OFF", "ON")
SUCCESS_FILE = "successful_dependencies.txt"

ALL_GIT_REPOS = (
    ("vlovero/OpenBLAS", None),
    ("vlovero/argparse", None),
    ("DrTimothyAldenDavis/SuiteSparse", "v7.8.2"),
    ("fmtlib/fmt", "11.0.2"),
    ("simdjson/simdjson", "v3.10.1"),
    ("HDFGroup/hdf5", "2.1.0"),
    ("symengine/symengine", "v0.14.0"),
    ("Tessil/ordered-map", "v1.1.0"),
    ("p-ranav/indicators", "v2.3"),
    ("Neargye/magic_enum", "v0.9.6"),
    ("LLNL/Caliper", None),
)
GMP_URL = "https://gmplib.org/download/gmp/gmp-6.3.0.tar.gz"
ZLIB_URL = "https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz"
FFTW_URL = "https://fftw.org/fftw-3.3.10.tar.gz"

ALL_CMAKE_BUILDS = {
    "Caliper": "-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX={PREFIX} -DBUILD_SHARED_LIBS=OFF ..",
    "fftw": "-DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DENABLE_SSE=ON -DENABLE_SSE2=ON -DENABLE_AVX=ON -DENABLE_AVX2=ON -DDISABLE_FORTRAN=ON -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH={PREFIX} -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..",
    "OpenBLAS": "-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH={PREFIX} -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..",
    "indicators": "-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_STANDARD=20 -DINDICATORS_SAMPLES=OFF -DINDICATORS_DEMO=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH={PREFIX} ..",
    "argparse": "-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DARGPARSE_BUILD_SAMPLES=off -DARGPARSE_BUILD_TESTS=off -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH={PREFIX} ..",
    "ordered-map": "-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH={PREFIX} ..",
    "symengine": "-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_TESTS=no -DBUILD_BENCHMARKS=no -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_FIND_USE_PACKAGE_ROOT_PATH=1 -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=0 -DGMP_ROOT={PREFIX} -DCMAKE_PREFIX_PATH={PREFIX} -DGMP_INCLUDE_DIRS={PREFIX}/include -DGMP_LIBRARIES={PREFIX}/lib/libgmp.a -DCMAKE_INSTALL_PREFIX:PATH={PREFIX} ..",
    "hdf5": '-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DHDF5_ALLOW_UNSUPPORTED=ON -DHDF5_PROVIDES_PARALLEL=ON -DHDF5_BUILD_CPP_LIB=ON -DBUILD_TESTING:BOOL=OFF -DHDF5_BUILD_TOOLS:BOOL=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH={PREFIX} -DHDF5_ALLOW_EXTERNAL_SUPPORT:STRING="TGZ" -DZLIB_TGZ_NAME={PREFIX}/hdf5/zlib-1.3.1.tar.gz -DHDF5_ENABLE_SZIP_SUPPORT:BOOL=OFF -DHDF5_USE_ZLIB_STATIC:BOOL=OFF -DZLIB_USE_LOCALCONTENT:BOOL=ON -DTGZPATH:PATH={PREFIX}/hdf5 ..',
    "simdjson": "-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DSIMDJSON_ENABLE_THREADS=OFF -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH={PREFIX} ..",
    "SuiteSparse": "-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBLA_STATIC=ON -DLAPACK_LIBRARIES={PREFIX}/lib/libopenblas.a -DBLAS_LIBRARIES={PREFIX}/lib/libopenblas.a -DLAPACK_INCLUDE_DIRS={PREFIX}/include/openblas  -DBLAS_INCLUDE_DIRS={PREFIX}/include/openblas -DSUITESPARSE_ENABLE_PROJECTS=umfpack -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH={PREFIX} ..",
    "fmt": "-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DFMT_TEST=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH={PREFIX} ..",
}

TWIST_CMAKE_ARGS = "-DCMAKE_BUILD_TYPE={BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH={PREFIX} -DPYTHON_EXEC={PYTHON_EXEC} -DBUILD_BENCHMARKS={BUILD_BENCHMARKS} -DTWIST_SANITIZE={TWIST_SANITIZE} -DWITH_WAVE_NUMBER={WITH_WAVE_NUMBER} -DCMAKE_POLICY_VERSION_MINIMUM=4.0 -DBUILD_TESTS={BUILD_TESTS} -DTWIST_GLOBAL_INSTALL={TWIST_GLOBAL_INSTALL} -DTWIST_THREAD_SANITIZE={TWIST_THREAD_SANITIZE} -DWITH_LOCAL_CONTINUATION={WITH_LOCAL_CONTINUATION} -DUSE_BETTER_AUX={USE_BETTER_AUX} ../.."
_BETTER_AUX_EXAMPLE = '''"auxillary": {
    "aux1": {
        "inputs": ["arg1", "arg2", "..."],
        "output": "..."
    }
}'''

@contextmanager
def dir_as(destination):
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)


def log_error(msg):
    print(f"{RED}Error:{RESET} {msg}")


def log_status(name, built=True):
    if built:
        print(f"{BOLD_RED}[{name}]{RESET} has already been built")
    else:
        print(f"{BOLD_RED}[{name}]{RESET} was found")


def download_github_repo(author_and_name, branch=None):
    cmd = ["git", "clone", "--depth=1"]
    if branch is not None:
        cmd.extend(["-b", branch])
    cmd.append(f"https://github.com/{author_and_name}.git")
    process = Popen(cmd, stdout=DEVNULL, stderr=DEVNULL)
    process.wait()


def check_python_dependency(package, why, env, show_output):
    try:
        __import__(package)
        log_status(package, False)
    except ImportError:
        log_error(f"{package} is not available for {sys.executable}")
        answer = input("install? (y/n) ")
        if answer.lower()[:1] == "y":
            run_process(package, "install", env, [sys.executable, "-m", "pip", "install", package], show_output)
            print(f"installed {package}")
        else:
            print(f"{package} is needed for {why}")
            exit(1)


def download_from_url(url, path):
    requests = __import__("requests")

    fname = os.path.basename(url)
    with dir_as(path):
        request = requests.get(url)
        with open(fname, "wb") as file:
            file.write(request.content)


def download_all_repos():
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        # do all git clones in parallel
        futures = {pool.submit(download_github_repo, what, branch): what for what, branch in ALL_GIT_REPOS}
        for future in as_completed(futures):
            print(f"finished downloading {futures[future]}")

        # do zip file downloads in parallel next
        futures = {
            pool.submit(download_from_url, GMP_URL, "."): GMP_URL,
            pool.submit(download_from_url, ZLIB_URL, "."): ZLIB_URL,
            pool.submit(download_from_url, FFTW_URL, "."): FFTW_URL,
        }
        for future in as_completed(futures):
            print(f"finished downloading {futures[future]}")

    # add sleep here becuase python processes can sometimes
    # finish before OS has registered downloaded file
    sleep(1e-2)
    
    shutil.move("zlib-1.3.1.tar.gz", "hdf5/zlib-1.3.1.tar.gz")
    # unzip gmp and fftw
    unzip_kwargs = {} if sys.version_info.minor < 12 else dict(filter=None)
    with tarfile.open("gmp-6.3.0.tar.gz", mode="r|gz") as file:
        file.extractall(**unzip_kwargs)

    with tarfile.open("fftw-3.3.10.tar.gz", mode="r|gz") as file:
        file.extractall(**unzip_kwargs)
    shutil.move("fftw-3.3.10", "fftw")


def download_local_openmp(prefix, env, show_output):
    global ALL_CMAKE_BUILDS
    # clone llvm
    cmd = ["git", "clone", "-b", "llvmorg-21.1.8", "-n", "--depth=1", "--filter=tree:0", "https://github.com/llvm/llvm-project.git"]
    run_process("openmp", "clone", env, cmd, show_output, _exit=False)
    with dir_as("llvm-project"):
        # clone openmp and needed cmake stuff
        cmd = ["git", "sparse-checkout", "set", "--no-cone", "/openmp", "/cmake", "CMakeLists.txt"]
        run_process("openmp", "sparse-checkout", env, cmd, show_output, _exit=False)
        run_process("openmp", "checkout", env, ["git", "checkout"], show_output, _exit=False)


    # instructions to build openmpmp
    ALL_CMAKE_BUILDS = {
        "llvm-project/openmp": "-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} -DBUILD_SHARED_LIBS=OFF -DLIBOMP_ENABLE_SHARED=OFF ..",
        **ALL_CMAKE_BUILDS,
    }
    env["OpenMP_ROOT"] = prefix


def run_process(name, what, env, cmd, show_output, _exit=True):
    if show_output:
        process = Popen(cmd, env=env)
    else:
        process = Popen(cmd, env=env, stdout=DEVNULL, stderr=DEVNULL)
    process.wait()
    if process.poll() != 0:
        if not show_output:
            log_error(f"{name} failed to {what}. Rerun with --verbose to see output")
        if _exit:
            exit(1)
        else:
            return 1


def run_base_checks(env, show_output):
    # check for cli tools that are needed
    for cmd in ("cmake", "git", "gfortran"):
        if shutil.which(cmd) is None:
            log_error(f"{cmd} wasn't found. Make sure the {cmd} executable is in your PATH")
            exit(1)
        log_status(cmd, built=False)

    # check for python packages that are needed for plotting
    check_python_dependency("requests", "downloading dependencies", env, False)
    check_python_dependency("matplotlib", "visualize command", env, False)
    check_python_dependency("h5py", "visualize command", env, False)

    # cmake checks
    for name in ("openmp",):
        with dir_as(f"checks/{name}"):
            if run_process("OpenMP", "be found", env, ["cmake", "."], show_output, _exit=False) is not None:
                # clear cache to avoid awkward bugs
                os.remove("CMakeCache.txt")
                os.remove("cmake_install.cmake")
                shutil.rmtree("CMakeFiles")
                log_error("cmake failed to locate OpenMP. Consider --local-openmp flag")
            

def build_cmake_dependency(name: str, env: dict, cache: set, parallel: bool, show_output: bool, cmake_args):
    if name in cache:
        log_status(name, True)
        return

    os.makedirs(f"{name}/build", exist_ok=True)
    with dir_as(f"{name}/build"):
        print(f"Working on: {name}")
        cmd = ["cmake", *cmake_args]
        run_process(name, "configure", env, cmd, show_output)

        cmd = ["cmake", "--build", ".", "--target", "install"]
        if parallel:
            cmd.append("--parallel")

        run_process(name, "build", env, cmd, show_output)


def build_gmp(prefix: str, env: dict, cache: set, show_output: bool):
    if "gmp" in cache:
        log_status("gmp", True)
        return

    with dir_as("gmp-6.3.0"):
        print("Working on: gmp")
        cmd = ["./configure", f"--{prefix=!s}", "--with-pic"]
        run_process("gmp", "configure", env, cmd, show_output)
        run_process("gmp", "build", env, ["make"], show_output)
        run_process("gmp", "install", env, ["make", "install"], show_output)

    paths = glob(os.path.join(prefix, "lib/libgmp*.so"))
    paths.extend(glob(os.path.join(prefix, "lib/libgmp*.dylib")))
    for path in paths:
        os.remove(path)

    cache.add("gmp")


def build_magic_enum(prefix: str, env: dict, cache: set, show_output: bool):
    if "magic_enum" in cache:
        log_status("magic_enum", True)
        return

    include_path = os.path.join(prefix, "include")
    os.makedirs(include_path, exist_ok=True)
    shutil.copytree("magic_enum/include/magic_enum", include_path, dirs_exist_ok=True)
    cache.add("magic_enum")


def setup():
    prefix = pathlib.Path(os.getcwd()) / "build"
    prefix.mkdir(exist_ok=True)
    env = dict(os.environ)
    parallel_builds = False

    env["USE_OPENMP"] = "1"

    if platform.system() == "Darwin":
        env["MACOSX_DEPLOYMENT_TARGET"] = check_output(["sw_vers", "-productVersion"]).decode().strip()
        env["OpenMP_ROOT"] = check_output(["brew", "--prefix", "libomp"]).decode().strip()

    if shutil.which("ninja") is not None:
        env["CLICOLOR_FORCE"] = "1"
        env["CMAKE_GENERATOR"] = "Ninja"
        parallel_builds = True

    run_base_checks(env, prefix)

    return prefix, env, parallel_builds


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

    # build types
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--debug", action="store_true", default=False, help="build TWIST with flags for debugging")
    group.add_argument("--optimized", action="store_true", default=False, help="build TWIST with flags for optimizing")
    group.add_argument("--profiling", action="store_true", default=False, help="build TWIST with flags for optimizing and debugging")
    group.add_argument("--small", action="store_true", default=False, help="build TWIST with flags for smaller binary")

    # what to build
    parser.add_argument("--with-tests", action="store_true", default=False, help="also build tests")
    parser.add_argument("--with-benchmarks", action="store_true", default=False, help="also build benchmarks")
    parser.add_argument(
        "--with-wave-number",
        action="store_true",
        default=False,
        help="build TWIST with ability to do continuation in wave number ('wave_number' will be the parameter name)",
    )
    parser.add_argument(
        "--with-local-continuation",
        action="store_true",
        default=False,
        help=f"{BOLD_RED}[UNFINISHED FEATURE]:{RESET} build TWIST with ability to do continuation in the local dynamics ('local-continuation' will be a new command)",
    )
    parser.add_argument(
        "--use-better-aux",
        action="store_true",
        default=False,
        help="build TWIST using a more robust auxillary function system.\n"
            "THIS CHANGES THE FUNCTIONALITY OF THE SPEC FILE.\n"
            f"The new spec file auxillary function entries have the syntax\n{_BETTER_AUX_EXAMPLE}",
    )

    # debugging compiler flags
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--with-sanitizers", action="store_true", default=False, help="runtime sanitizers for debugging unknown crashes")
    group.add_argument(
        "--with-thread-sanitizers",
        action="store_true",
        default=False,
        help="runtime sanitizers for debugging unknown crashes (only for multithreading)",
    )

    # extra options
    parser.add_argument(
        "--local-install", action="store_true", default=False, help=f"install TWIST to {os.getcwd()}/build/bin/ instead of /usr/local/bin/"
    )
    parser.add_argument("--local-openmp", action="store_true", default=False, help="download, build, and use this OpenMP instead of system")
    parser.add_argument("--remove-git-dependencies", action="store_true", default=False, help="delete downloaded git repos")
    parser.add_argument("--clean-bin-folder", action="store_true", default=False, help="delete all binaries created by dependencies")
    parser.add_argument("--verbose", action="store_true", default=False, help="show output for everything")

    args = parser.parse_args()

    if not args.local_install:
        if not os.access("/usr/local/bin", os.W_OK | os.F_OK):
            print("/usr/local/bin either does not exist or you do not have permission to write to it. Use the --local-install flag")
            exit(1)

    BUILD_BENCHMARKS = CMAKE_ONOFF[args.with_benchmarks]
    BUILD_TESTS = CMAKE_ONOFF[args.with_tests]
    TWIST_SANITIZE = CMAKE_ONOFF[args.with_sanitizers]
    TWIST_THREAD_SANITIZE = CMAKE_ONOFF[args.with_thread_sanitizers]
    WITH_WAVE_NUMBER = CMAKE_ONOFF[args.with_wave_number]
    TWIST_GLOBAL_INSTALL = CMAKE_ONOFF[not args.local_install]
    WITH_LOCAL_CONTINUATION = CMAKE_ONOFF[args.with_local_continuation]
    USE_BETTER_AUX = CMAKE_ONOFF[args.use_better_aux]
    if args.debug:
        BUILD_TYPE = "Debug"
    elif args.optimized:
        BUILD_TYPE = "Release"
    elif args.profiling:
        BUILD_TYPE = "RelWithDebInfo"
    else:
        BUILD_TYPE = "MinSizeRel"

    # try to use python3 by default just in case python 2 is being used
    # extension stuff is for windows
    PYTHON_EXEC = sys.executable
    exe_dir = os.path.dirname(PYTHON_EXEC)
    name, ext = os.path.splitext(os.path.basename(PYTHON_EXEC))
    if name.lower() == "python":
        possible_py3 = os.path.join(exe_dir, "python3" + ext)
        if os.path.exists(possible_py3):
            PYTHON_EXEC = possible_py3
    # again just for windows...
    PYTHON_EXEC = PYTHON_EXEC.replace('\\', '/')

    prefix, env, parallel_builds = setup()

    with dir_as(prefix):
        if not os.path.exists(SUCCESS_FILE):
            with open(SUCCESS_FILE, "w") as _:
                pass

        with open(SUCCESS_FILE, "r") as cache_file:
            cache = set(map(lambda line: line.strip(), cache_file.readlines()))

        if "__downloaded__" not in cache:
            download_all_repos()
            cache.add("__downloaded__")
            with open(SUCCESS_FILE, "w") as cache_file:
                cache_file.write("\n".join(cache))

        if args.with_benchmarks:
            download_github_repo("google/benchmark", "v1.9.4")
            ALL_CMAKE_BUILDS["benchmark"] = (
                "-DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DCMAKE_INSTALL_PREFIX:PATH={PREFIX} .."
            )

        if args.local_openmp:
            download_local_openmp(prefix, env, args.verbose)

        build_gmp(prefix.as_posix(), env, cache, args.verbose)
        build_magic_enum(prefix.as_posix(), env, cache, args.verbose)

        for name, cmake_str in ALL_CMAKE_BUILDS.items():
            cmake_args = cmake_str.format(PREFIX=prefix.as_posix()).split()
            build_cmake_dependency(name, env, cache, parallel_builds, args.verbose, cmake_args)
            cache.add(name)

            # not efficient but whatever
            with open(SUCCESS_FILE, "w") as cache_file:
                cache_file.write("\n".join(cache))

        TWIST_dir = prefix / f"TWIST_{BUILD_TYPE}"
        TWIST_dir.mkdir(exist_ok=True)
        with dir_as(str(TWIST_dir)):
            cmake_args = TWIST_CMAKE_ARGS.format(
                BUILD_TYPE=BUILD_TYPE,
                PREFIX=prefix.as_posix(),
                PYTHON_EXEC=PYTHON_EXEC,
                BUILD_BENCHMARKS=BUILD_BENCHMARKS,
                BUILD_TESTS=BUILD_TESTS,
                TWIST_GLOBAL_INSTALL=TWIST_GLOBAL_INSTALL,
                TWIST_SANITIZE=TWIST_SANITIZE,
                WITH_WAVE_NUMBER=WITH_WAVE_NUMBER,
                TWIST_THREAD_SANITIZE=TWIST_THREAD_SANITIZE,
                WITH_LOCAL_CONTINUATION=WITH_LOCAL_CONTINUATION,
                USE_BETTER_AUX=USE_BETTER_AUX
            ).split()
            run_process("TWIST", "configure", env, ["cmake", *cmake_args], True)
            cmd = ["cmake", "--build", ".", "--target", "install"]
            if parallel_builds:
                cmd.append("--parallel")
            run_process("TWIST", "build", env, cmd, True)
            shutil.move("compile_commands.json", "../compile_commands.json")

        if args.remove_git_dependencies:
            for name in [*ALL_CMAKE_BUILDS.keys(), "gmp-6.3.0", "magic_enum"]:
                if os.path.isdir(name):
                    print(f"deleting {name}")
                    shutil.rmtree(name)
                else:
                    print(f"already removed {name}")
            for name in ("fftw-3.3.10.tar.gz", "gmp-6.3.0.tar.gz"):
                if os.path.exists(name):
                    os.remove(name)
        if args.clean_bin_folder:
            with dir_as("bin"):
                for file in glob("h5*") + glob("cali*") + glob("mirror*"):
                    os.remove(file)
