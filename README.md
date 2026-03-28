# TWIST: A Tool to Accurately and Efficiently Analyze Traveling Wave Solutions in Reaction-Diffusion Systems

**Vincent Lovero**  
*Mathematics Department, University of California, Davis, USA*  

---

## 1. What is TWIST

Traveling Wave Investigation Software Tool (TWIST) is a command line tool for the computation, continuation, and bifurcation analysis of traveling wave solutions in reaction diffusion systems. TWIST was originally designed to handle high-dimensional, electrophysiological models of cardiac tissue. Therefore, TWIST is capable of computing dispersion curves and spectra of traveling wave solutions with near machine precision accuracy. Existing software that is capable of generating dispersion curves was not designed for complicated, high-dimensional systems, nor were they designed for fast and accurate computation of spectra.

The goal of TWIST was not only to be capable of performing the necessary computations, but to also be fast and easy to use. TWIST uses algorithms outlined in the manual that are ideal for large, ill-conditioned systems such as those found in models of cardiac tissue. TWIST is capable of automating almost every step in the computation of dispersion curves and aims to not require a user to have any knowledge of the inner workings of the software.

### TWIST offers the following capabilities:

- Automatically generate, perform coordinate transforms, and compile a model from simple text
- Compute traveling wave solutions (and an initial wave)
- Continue traveling wave solutions through parameter space
- Perform bifurcation analysis on families of traveling wave solutions
- Simulation traveling wave computed solutions in time
- Plot traveling wave solutions, dispersion curves, and kymographs of simulations

TWIST does all of the above quickly and accurately.

## 2. Building and Installing TWIST

### Dependencies

Before building TWIST, you need:

1. **git**
2. **cmake** (version 3.24 or later)
3. **C++20 compiler** (GCC, Clang, or Apple-Clang)
4. **gfortran**
5. **OpenMP**
6. **Python**

### Installing OpenMP

- **MacOS**: `brew install libomp`
- **Linux**: Usually pre-installed with compiler, or `sudo apt install libomp-dev`

### Installing Python

Visit https://www.python.org/ and follow installation instructions for your platform.

### Build Instructions

The build process takes 5-30 minutes and only needs to be done once.

#### Standard Build (sudo may be required):
```bash
python3 build.py --optimized
```

#### Local Installation (no sudo required):
```bash
python3 build.py --optimized --local-install
```

This installs TWIST to `./build/bin` instead of `/usr/local/bin/`.

### What Gets Built

TWIST automatically downloads and builds these dependencies:

 - OpenBLAS
 - UMFPack
 - libfmt
 - simdjson
 - HDF5
 - SymEngine
 - ordered map
 - argparse
 - indicators
 - magic enum
 - gmp
 - FFTW
 - Caliper

### Verify Installation

Test that TWIST installed correctly:
```bash
twist --help
```

If this displays lots of help information, installation was successful.

### Build Output Locations

- Models source code: `.cache/models/src`
- Compiled models: `.cache/models/lib`
- Default binary: `/usr/local/bin/twist` (or `./build/bin/twist` with `--local-install`)

### Troubleshooting

- **Windows users**: Use WSL or MinGW-w64 from MSYS2
- **MacOS "syntax error"**: Use `zsh` instead of `bash`
- **Permission denied**: Use `--local-install` flag or run with sudo

## NOTE FOR WINDOWS USERS
To build TWIST, the GMP and gfortran runtime libraries are needed.
These can be problematic on windows, therefore it is highly recommended
to use the MSYS2 development environment. The following (untested)
instructions should be used:
1. Install MSYS2 following the instructions [here](https://www.msys2.org/)
2. From the MSYS2 terminal, install python, gcc, g++, gfortran, ninja, make, and cmake using the following two commands:
```bash
pacman -Syu
pacman -S git mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-gcc-fortran mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-make mingw-w64-ucrt-x86_64-python mingw-w64-ucrt-x86_64-python-pip
```

## NOTE ON IMPROVING EASE OF USE
If using the `--local-install` flag, it can be cumbersome to write `./build/bin/twist`
every time. Therefore, a run commands (rc) file is included (for UNIX shells).
This rc file will add the location of the twist executable to `PATH`
and enable autocompletions to further improve ease of use of twist.
To enable these settings run
```bash
source .twistrc
```
from inside the twist directory.

_Note_: autocompletions for zsh are much better than for bash.
