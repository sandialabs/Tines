# Building TINES

TINES requires Kokkos and the code uses CBLAS and LAPACKE interfaces from OpenBLAS or Intel MKL. For testing, we use GTEST library. For convenience, we explain how to build the TINES using the following environment variables.

```
/// repositories
export KOKKOS_REPOSITORY_PATH=/where/you/clone/kokkos/git/repo
export TINES_REPOSITORY_PATH=/where/you/clone/tines/git/repo
export GTEST_REPOSITORY_PATH=/where/you/clone/gtest/git/repo

/// build directories
export KOKKOS_BUILD_PATH=/where/you/build/kokkos
export TINES_BUILD_PATH=/where/you/build/tines
export GTEST_BUILD_PATH=/where/you/build/gtest

/// install directories
export KOKKOS_INSTALL_PATH=/where/you/install/kokkos
export TINES_INSTALL_PATH=/where/you/install/tines
export GTEST_INSTALL_PATH=/where/you/install/gtest
export OPENBLAS_INSTALL_PATH=/where/you/install/openblas
export LAPACKE_INSTALL_PATH=/where/you/install/lapacke
```

## Download Libraries

Clone Kokkos and TINES repositories.

```
git clone https://github.com/kokkos/kokkos.git ${KOKKOS_REPOSITORY_PATH};
git clone https://github.com/google/googletest.git ${GTEST_REPOSITORY_PATH}
git clone getz.ca.sandia.gov:/home/gitroot/math_utils ${TINES_REPOSITORY_PATH};
```

Here, we assume that TPLs (OpenBLAS and LAPACKE) are compiled and installed separately as these TPLs can be easily built using a distribution tool e.g., apt, yum, macports. We also recommend to turn off the threading capability of these TPLs as OpenMP is used for processing batch parallelism.

## Building Libraries and Configuring TINES

### Kokkos

This example builds Kokkos on Intel SandyBridge architectures and install it to ``${KOKKOS_INSTALL_PATH}``. For other available options, see [Kokkos github pages](https://github.com/kokkos/kokkos).

```
cd ${KOKKOS_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${CXX}"  \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE=OFF \
    -D Kokkos_ARCH_SNB=ON \
    ${KOKKOS_REPOSITORY_PATH}
make -j install
```

To compile for NVIDIA GPUs, one can customize the cmake script. Note that we use ``nvcc_wrapper`` provided by Kokkos as its compiler. The architecture flag in the following example indicates that the host architecture is Intel SandyBridge and the GPU is a NVIDIA Volta 70 generation. With Kokkos 3.1, the CUDA architecture flag is optional (the script automatically detects a correct CUDA arch flag).
```
cd ${KOKKOS_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${KOKKOS_REPOSITORY_PATH}/bin/nvcc_wrapper"  \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_CUDA:BOOL=ON \
    -D Kokkos_ENABLE_CUDA_UVM:BOOL=OFF \
    -D Kokkos_ENABLE_CUDA_LAMBDA:BOOL=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE=OFF \
    -D Kokkos_ARCH_VOLTA70=ON \
    -D Kokkos_ARCH_SNB=ON \
    ${KOKKOS_REPOSITORY_PATH}
make -j install
```

It its worth for noting that (1) the serial execution space is enabled for Kokkos when the serial execution space is used for example and unit tests and (2) when CUDA is enabled, we do not explicitly use universal virtual memory (UVM). A user can enable UVM if it is used for the application. However, TINES does not assume that the host code can access device memory.

### TINES

Compiling TINES follows Kokkos configuration settings, also available at ``${KOKKOS_INSTALL_PATH}``. The OpenBLAS and LAPACKE libraries are required on the host device. These provide optimized dense linear algebra tools. When an Intel compiler is available, one can replace these libraries with Intel MKL by adding an option ``TINES_ENABLE_MKL=ON``. On Mac OSX, we use the OpenBLAS library managed by **macports**. This version of the OpenBLAS has different header names and we need to distinguish this version of the code from others which are typically used in Linux distributions. To discern the two version of the code, cmake looks for ``cblas_openblas.h`` to tell that the installed version is from MacPorts. This mechanism can be broken if MacPorts' OpenBLAS is changed later. The MacPorts OpenBLAS version also include LAPACKE interface and one can remove ``LAPACKE_INSTALL_PATH`` from the configure script. SACADO library is a header only library and it is included in the TINES distributions.

```
cd ${KOKKOS_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX=${TINES_INSTALL_PATH} \
    -D CMAKE_CXX_COMPILER="${CXX}" \
    -D CMAKE_CXX_FLAGS="-g" \
    -D TINES_ENABLE_DEBUG=OFF \
    -D TINES_ENABLE_VERBOSE=OFF \
    -D TINES_ENABLE_TEST=ON \
    -D TINES_ENABLE_EXAMPLE=ON \
    -D KOKKOS_INSTALL_PATH="${HOME}/Work/lib/kokkos/install/butter/release" \
    -D GTEST_INSTALL_PATH="${HOME}/Work/lib/gtest/install/butter/release" \
    -D OPENBLAS_INSTALL_PATH="${OPENBLAS_INSTALL_PATH}" \
    -D LAPACKE_INSTALL_PATH="${LAPACKE_INSTALL_PATH}" \
    ${TINES_REPOSITORY_PATH}/src
make -j install
```

For GPUs, the compiler is changed with ``nvcc_wrapper`` by adding ``-D CMAKE_CXX_COMPILER="${KOKKOS_INSTALL_PATH}/bin/nvcc_wrapper"``.

### GTEST

We use GTEST as our testing infrastructure. GTEST can be compiled and installed using the following cmake script.

```
cd ${GTEST_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${GTEST_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${CXX}"  \
    ${GTEST_REPOSITORY_PATH}
make -j install
```
