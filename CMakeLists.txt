#=============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2023, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
#
#=============================================================================
cmake_minimum_required(VERSION 3.19)

project(aimet)

# TODO Needed below variable CMAKE_CXX_FLAGS_ORG for tensorflow 2.10 verasion.
# We have use -std=c++17 -O0 -ggdb -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 for TF2.10
set(CMAKE_CXX_FLAGS_ORG "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -ggdb -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -DPYBIND11_BUILD_ABI=\\\"_cxxabi1011\\\"")

set(AIMET_INSTALL_DIR ${CMAKE_BINARY_DIR}/staging/universal)

if(NOT DEFINED AIMET_PYTHONPATH)
  set(AIMET_PYTHONPATH "PYTHONPATH=${CMAKE_BINARY_DIR}/artifacts" CACHE STRING "python path")
endif()
set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/common/src/python")

set(WHL_EDITABLE_MODE OFF CACHE BOOL "Enable editable mode, wheels would have symlinks to C++ part instead of copies")
set(WHL_PREP_DIR "${CMAKE_BINARY_DIR}/whlprep" CACHE STRING "A path to store extra files which should be included in the wheels")
set(WHL_PREP_AIMET_COMMON_DIR "${WHL_PREP_DIR}/aimet_common" CACHE STRING "A path to store extra files which should be included in the aimet_common wheel")
set(WHL_PREP_AIMET_TORCH_DIR "${WHL_PREP_DIR}/aimet_torch" CACHE STRING "A path to store extra files which should be included in the aimet_torch wheel")
set(WHL_PREP_AIMET_TENSORFLOW_DIR "${WHL_PREP_DIR}/aimet_tensorflow" CACHE STRING "A path to store extra files which should be included in the aimet_tensorflow wheel")
set(WHL_PREP_AIMET_ONNX_DIR "${WHL_PREP_DIR}/aimet_onnx" CACHE STRING "A path to store extra files which should be included in the aimet_onnx wheel")
mark_as_advanced(WHL_PREP_DIR WHL_PREP_AIMET_COMMON_DIR WHL_PREP_AIMET_TORCH_DIR WHL_PREP_AIMET_TENSORFLOW_DIR WHL_PREP_AIMET_ONNX_DIR)

# Set the software version from version.txt file (if not already set)
if(NOT DEFINED SW_VERSION)
  file(STRINGS "packaging/version.txt" SW_VERSION)
  message(STATUS "Set SW_VERSION = ${SW_VERSION} from ${CMAKE_CURRENT_SOURCE_DIR}/packaging/version.txt")
else()
  message(STATUS "SW_VERSION already set to ${SW_VERSION}.")
endif()

find_package(Threads)
# Disabling openMP since statically linked OpenMP in PyTorch (v1.5.0) seems to conflict with aimet usage
#find_package(OpenMP REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# -------------------------------
# Conditional build for CUDA
# -------------------------------
if (NOT (DEFINED ENABLE_CUDA))
  message("Compiling with CUDA not explicitly disabled. Enabling implicitly")
  set(ENABLE_CUDA ON CACHE BOOL "")
endif(NOT (DEFINED ENABLE_CUDA))

set(CUDA_VER_STRING "cpu")
if (ENABLE_CUDA)
  message("Compiling with CUDA enabled")

  find_package(CUDAToolkit)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 72)
  endif()
  message(STATUS "** Initial CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES} **")

  enable_language(CUDA)

  # Find CUDA toolkit version
  find_package(CUDAToolkit QUIET)
  # Truncate the string for use in version string ex. 11.6.124 --> cu116
  string(REGEX REPLACE "^([0-9]+)\\.([0-9]+)\\.*[0-9]*" "cu\\1\\2" CUDA_VER_STRING ${CUDAToolkit_VERSION})
  message(STATUS "Found CUDA toolkit version ${CUDAToolkit_VERSION}, using ${CUDA_VER_STRING}")
  
else(ENABLE_CUDA)
  message("Compiling with CUDA disabled")

endif(ENABLE_CUDA)

find_package(Python3 COMPONENTS Interpreter Development)
message("Found python: ${Python3_FOUND}, at ${Python3_LIBRARIES}")

find_package(PkgConfig)
pkg_search_module(LAPACKE REQUIRED lapacke)
set(OLD_FIND_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_STATIC_LIBRARY_SUFFIX}")
pkg_search_module(OPENCV opencv) # in case opencv will not be found, it would be built form source
set(CMAKE_FIND_LIBRARY_SUFFIXES "${OLD_FIND_LIBRARY_SUFFIXES}")

# ----------------------------------
# Conditional build for ONNX
# ----------------------------------

if (NOT (DEFINED ENABLE_ONNX))
  message("AIMET Onnx build not explicitly disabled.")
  set(ENABLE_ONNX OFF CACHE BOOL "")

endif(NOT (DEFINED ENABLE_ONNX))

if (ENABLE_ONNX)
  execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "import onnx; print(onnx.__version__)"
                  OUTPUT_VARIABLE ONNX_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(FMWORK_VERSION ${ONNX_VERSION})
  message(STATUS "Found ONNX version: ${ONNX_VERSION}")

  if (NOT ENABLE_TORCH)
    message(STATUS "Enabling Torch as it is required by ONNX")
    set(ENABLE_TORCH ON CACHE BOOL "" FORCE)
  endif(NOT ENABLE_TORCH)

  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/onnx/src/python")
  get_filename_component(ONNX_DIR ${ONNX_CMAKE_DIR}/../../ ABSOLUTE)
else (ENABLE_ONNX)
  message("AIMET Onnx build disabled")
endif (ENABLE_ONNX)

# ----------------------------------
# Conditional build for PyTorch
# ----------------------------------
if (NOT (DEFINED ENABLE_TORCH))
  message(STATUS "AIMET Torch build not explicitly disabled. Enabling implicitly")
  set(ENABLE_TORCH ON CACHE BOOL "")

endif(NOT (DEFINED ENABLE_TORCH))
message(STATUS "AIMET - ENABLE_TORCH: ${ENABLE_TORCH}" )

if (ENABLE_TORCH)
  execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "import torch; print(torch.__version__)"
                  OUTPUT_VARIABLE TORCH_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (NOT ENABLE_ONNX)
    set(FMWORK_VERSION ${TORCH_VERSION})
  endif()
  message(STATUS "Found Torch version: ${TORCH_VERSION}")

  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/torch/src/python")
  execute_process(COMMAND ${Python3_EXECUTABLE} "-c" "import torch;print(torch.utils.cmake_prefix_path)"
                  RESULT_VARIABLE TORCH_NOT_FOUND
                  OUTPUT_VARIABLE TORCH_CMAKE_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  )

  # Update CMAKE_CUDA_ARCHITECTURES with the supported architectures that this pytorch version was
  # compiled for:
  #   1. Remove sm_ prefixes from the CUDA architecture names.
  #   2. Change python list into a CMake list.
  #execute_process(COMMAND ${Python3_EXECUTABLE} "-c" "import torch; print(';'.join(arch.split('_')[1] for arch in torch.cuda.get_arch_list()))"
  #                RESULT_VARIABLE TORCH_NOT_FOUND
  #                OUTPUT_VARIABLE CMAKE_CUDA_ARCHITECTURES
  #                OUTPUT_STRIP_TRAILING_WHITESPACE
  #                )
  #message(STATUS "** Updated CMAKE_CUDA_ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES} **")

  # We remove certain architectures that are not supported
  #set(UNSUPPORTED_CUDA_ARCHITECTURES_TORCH 90)
  #list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES ${UNSUPPORTED_CUDA_ARCHITECTURES_TORCH})
  #message(STATUS "** Removed unsupported archs (${UNSUPPORTED_CUDA_ARCHITECTURES_TORCH}), \
  #                Now CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES} **")

  # Set torch cuda architecture list variable
  # Convert to the proper format (Reference: https://stackoverflow.com/a/74962874)
  #   - Insert "." between the digits of the architecture version (ex. 50 --> 5.0)
  #   - Repleace semi-colons in list with spaces
  #set(TORCH_CUDA_ARCH_LIST ${CMAKE_CUDA_ARCHITECTURES})
  #list(TRANSFORM TORCH_CUDA_ARCH_LIST REPLACE "([0-9])([0-9])" "\\1.\\2")
  #string(REPLACE ";" " " TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST}")
  #message(STATUS "** Updated TORCH_CUDA_ARCH_LIST to ${TORCH_CUDA_ARCH_LIST} **")

  find_package(Torch REQUIRED PATHS ${TORCH_CMAKE_DIR} NO_DEFAULT_PATH)
  get_filename_component(TORCH_DIR ${TORCH_CMAKE_DIR}/../../ ABSOLUTE)

else (ENABLE_TORCH)
  message(STATUS "AIMET Torch build disabled inside AIMET")
endif (ENABLE_TORCH)

# ----------------------------------
# PyBind11
# ----------------------------------
execute_process(COMMAND ${Python3_EXECUTABLE} "-c" "import pybind11;print(pybind11.get_include())"
                RESULT_VARIABLE PYBIND11_NOT_FOUND
                OUTPUT_VARIABLE PYBIND11_INCLUDE_DIRS
                OUTPUT_STRIP_TRAILING_WHITESPACE
                )
# If we enable PyTorch builds then use the pybind11 headers that are part of the torch pip install
# So we don't have a version mismatch - between PyTorch custom C++ op code and PyMO
find_path(PYBIND11_INCLUDE "pybind11.h"
          PATHS ${TORCH_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIRS}
          PATHS ${ONNX_INCLUDE_DIRS} ${PYBIND11_INCLUDE_DIRS}
          PATH_SUFFIXES "pybind11"
          REQUIRED
          NO_DEFAULT_PATH
        )
get_filename_component(PYBIND11_INC_DIRS ${PYBIND11_INCLUDE} DIRECTORY)

add_library(PYBIND11 SHARED IMPORTED)

set_target_properties(PYBIND11 PROPERTIES
        IMPORTED_LOCATION ${Python3_LIBRARIES}
        INTERFACE_INCLUDE_DIRECTORIES "${PYBIND11_INC_DIRS}"
        INTERFACE_LINK_LIBRARIES Python3::Module
        )

# ----------------------------------
# Conditional build for TensorFlow
# ----------------------------------

if (NOT (DEFINED ENABLE_TENSORFLOW))
  message("AIMET TensorFlow build not explicitly disabled. Enabling implicitly")
  set(ENABLE_TENSORFLOW ON CACHE BOOL "")

endif(NOT (DEFINED ENABLE_TENSORFLOW))

if (ENABLE_TENSORFLOW)
  # Get Tensorflow version
  execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "import tensorflow as tf; print(tf.__version__)"
                  OUTPUT_VARIABLE TF_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(FMWORK_VERSION ${TF_VERSION})

  if(NOT DEFINED TF_LIB_DIR)
    # Get location of TensorFlow library
    execute_process(COMMAND "${Python3_EXECUTABLE}" "-c" "import tensorflow as tf; print(tf.sysconfig.get_lib())"
                    OUTPUT_VARIABLE TF_LIB_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Found TensorFlow version: ${TF_VERSION} TensorFlow library path: ${TF_LIB_DIR}")
  else(TF_LIB_DIR)
    message(STATUS "TensorFlow version and TensorFlow library already set.")
  endif(NOT DEFINED TF_LIB_DIR)

  # Use different cmake version for TF2.10
  if (${TF_VERSION} MATCHES "2.10.*")
    message(STATUS "TensorFlow version is 2.10. Setting cmake cxx flags with c++17")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_ORG} -std=c++17 -O0 -ggdb -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 -DPYBIND11_BUILD_API=\\\"_cxxabi1011\\\"")
  endif (${TF_VERSION} MATCHES "2.10.*")

  # Find the TensorFlow library file
  find_library(TF_LIB_FILE NAMES libtensorflow_framework.so.1 libtensorflow_framework.so.2 HINTS ${TF_LIB_DIR})
  if(NOT TF_LIB_FILE)
    message(FATAL_ERROR "TensorFlow library NOT found at ${TF_LIB_DIR}")
  endif()

  # find the _pywrap_tensorflow_internal.so library. Used for custom ops
  find_library(PYWRAP_TF_INTERNAL NAMES _pywrap_tensorflow_internal.so HINTS ${TF_LIB_DIR}/python/)

  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/TrainingExtensions/tensorflow/src/python")
  list(APPEND TENSORFLOW_INCLUDES "${TF_LIB_DIR}/include")
  add_library(TensorFlow SHARED IMPORTED)
  set_target_properties(TensorFlow PROPERTIES
        IMPORTED_LOCATION
          "${TF_LIB_FILE}"
        INTERFACE_INCLUDE_DIRECTORIES
          "${TENSORFLOW_INCLUDES}"
        )

  add_library(PyWrapTensorFlowInternal SHARED IMPORTED)
  set_target_properties(PyWrapTensorFlowInternal PROPERTIES
        IMPORTED_LOCATION "${PYWRAP_TF_INTERNAL}"
        )

else (ENABLE_TENSORFLOW)
  message("AIMET TensorFlow build disabled")

endif (ENABLE_TENSORFLOW)

# Export PYTHONPATH to the parent cmake scope (if present)
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  set(AIMET_PYTHONPATH "${AIMET_PYTHONPATH}" PARENT_SCOPE)
else()
  message(STATUS "Set ${AIMET_PYTHONPATH} in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

find_program(PATCHELF_EXE patchelf)
if (PATCHELF_EXE)
    message(STATUS "Found patchelf in '${PATCHELF_EXE}'")
else()
    # FIXME
    # Better to include patchefl into docker image, although seems it is not trivial
    include(FetchContent)
    FetchContent_Declare(patchelf
        URL "https://github.com/NixOS/patchelf/releases/download/0.15.0/patchelf-0.15.0-x86_64.tar.gz"
    )
    if(NOT patchelf_POPULATED)
        FetchContent_Populate(patchelf)
        set(PATCHELF_EXE ${patchelf_SOURCE_DIR}/bin/patchelf)
        message(STATUS "patchelf: ${PATCHELF_EXE}")
    endif()
endif()

# -------------------------------
# Generate pip packages
# -------------------------------
# Set the packaging path (if not already set)
if(NOT DEFINED AIMET_PACKAGE_PATH)
  set(AIMET_PACKAGE_PATH "\"${AIMET_INSTALL_DIR}\"")
  message(STATUS "Set AIMET_PACKAGE_PATH = ${AIMET_PACKAGE_PATH}")
endif(NOT DEFINED AIMET_PACKAGE_PATH)

set(remote_url_cmake_opt "-DREMOTE_URL=\"\"")
if (PIP_INDEX EQUAL "reporelease")
  execute_process(COMMAND git config --get remote.origin.url WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE remote_url)
  if(NOT remote_url STREQUAL "")
    string(REGEX REPLACE "\n$" "" remote_url "${remote_url}")
    # Remove the ".git" suffix from the remote repo URL
    string(REGEX REPLACE "\\.[^.]*$" "" remote_url ${remote_url})
    string(REGEX REPLACE ".*@" "" remote_post ${remote_url})
    set(remote_url "https://${remote_post}")
    message(STATUS "Repo Remote URL = ${remote_url}")
    set(remote_url_cfg "\"${remote_url}\"")
    set(sw_version_cfg "\"${SW_VERSION}\"")

    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/packaging/setup_cfg.py" "version=${sw_version_cfg}\n")
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/packaging/setup_cfg.py" "remote_url=${remote_url_cfg}")

    set(remote_url_cmake_opt "-DREMOTE_URL=${remote_url}")
  else()
    message(FATAL_ERROR "Repo Remote URL is blank. Unable to create AIMET wheel package")
  endif()
endif()

add_custom_target(packageaimet
  # Run the install target first
  COMMAND "${CMAKE_COMMAND}" --build "${PROJECT_BINARY_DIR}" --target install

  # Now run the packaging target to generate wheel files
  COMMAND ${CMAKE_COMMAND} -DSW_VERSION=${SW_VERSION} -DCUDA_VER_STRING=${CUDA_VER_STRING} -DPIP_INDEX=${PIP_INDEX} ${remote_url_cmake_opt} -DPYTHON3_EXECUTABLE=${Python3_EXECUTABLE} -DAIMET_PACKAGE_PATH=${AIMET_PACKAGE_PATH} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DENABLE_CUDA=${ENABLE_CUDA} -DENABLE_TENSORFLOW=${ENABLE_TENSORFLOW} -DENABLE_TORCH=${ENABLE_TORCH} -DENABLE_ONNX=${ENABLE_ONNX} -DPATCHELF_EXE=${PATCHELF_EXE} -DFMWORK_VERSION=${FMWORK_VERSION} -P ${CMAKE_CURRENT_SOURCE_DIR}/packaging/package_aimet.cmake
)

add_subdirectory(ThirdParty)

if (NOT (DEFINED ENABLE_TESTS))
  message("Test build not explicitly disabled. Enabling implicitly")
  set(ENABLE_TESTS ON CACHE BOOL "")
endif(NOT (DEFINED ENABLE_TESTS))

if (ENABLE_TESTS)
    message("Compiling test targets enabled")
    enable_testing()
    add_subdirectory(NightlyTests)
else (ENABLE_TESTS)
    message("Compiling test targets disabled")
endif()

add_subdirectory(ModelOptimizations)
add_subdirectory(TrainingExtensions)
add_subdirectory(Examples)
add_subdirectory(Docs)
if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Benchmarks")
  add_subdirectory(Benchmarks)
endif()

# -------------------------------
# Packaging - deprecated
# -------------------------------
set(CPACK_PACKAGE_NAME "aimet")
set(CPACK_PACKAGE_VERSION_MAJOR 0)
set(CPACK_PACKAGE_VERSION_MINOR 1)
set(CPACK_PACKAGE_VERSION_PATCH 1)
set(CPACK_SYSTEM_NAME "build-0.0.0.0")
set(CPACK_GENERATOR "TGZ")
set(CPACK_INSTALLED_DIRECTORIES ${AIMET_INSTALL_DIR} .)
include(CPack)

# -------------------------------
# Upload pip packages
# -------------------------------

# Check the pip config file path and set it to a default value (if not set)
if(NOT DEFINED PIP_CONFIG_FILE)
  set(PIP_CONFIG_FILE "None")
endif()

# Check the pip index name and set it to a default value (if not set)
if(NOT DEFINED PIP_INDEX)
  set(PIP_INDEX "None")
endif()

# Check the pip certificate path and set it to a default value (if not set)
if(NOT DEFINED PIP_CERT_FILE)
  set(PIP_CERT_FILE "None")
endif()

#TODO For some reason, this package upload target does NOT work as expected and needs to be debugged
add_custom_target(upload
    # Now run the packaging target to upload the pip package
    COMMAND ${CMAKE_COMMAND} -DPIP_CONFIG_FILE=${PIP_CONFIG_FILE} -DPIP_INDEX=${PIP_INDEX} -DPIP_CERT_FILE=${PIP_CERT_FILE} -P ${CMAKE_CURRENT_SOURCE_DIR}/packaging/upload_aimet.cmake
    DEPENDS packageaimet
)
