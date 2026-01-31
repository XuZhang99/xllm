#!/bin/bash

# Color definitions
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Configuration
REPO_ROOT=`pwd`
INSTALL_PREFIX="${HOME}/.local/yalantinglibs"

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages and exit
print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
    exit 1
}

# Function to check command success
check_success() {
    if [ $? -ne 0 ]; then
        print_error "$1"
    fi
}

# Create install directory if it doesn't exist
if [ ! -d "${INSTALL_PREFIX}" ]; then
    mkdir -p "${INSTALL_PREFIX}"
    check_success "Failed to create install directory: ${INSTALL_PREFIX}"
fi

echo -e "${YELLOW}Installing to: ${INSTALL_PREFIX}${NC}"


# Install yalantinglibs
print_section "Installing yalantinglibs"

# Check if thirdparties directory exists
if [ ! -d "${REPO_ROOT}/third_party/Mooncake/thirdparties" ]; then
    mkdir -p "${REPO_ROOT}/third_party/Mooncake/thirdparties"
    check_success "Failed to create Mooncake/thirdparties directory"
fi

# Change to thirdparties directory
cd "${REPO_ROOT}/third_party/Mooncake/thirdparties"
check_success "Failed to change to Mooncake/thirdparties directory"

# Check if yalantinglibs is already installed
if [ -d "yalantinglibs" ]; then
    echo -e "${YELLOW}yalantinglibs directory already exists. Removing for fresh install...${NC}"
    rm -rf yalantinglibs
    check_success "Failed to remove existing yalantinglibs directory"
fi

# Clone yalantinglibs
echo "Cloning yalantinglibs from https://gitcode.com/gh_mirrors/ya/yalantinglibs.git"
git clone https://gitcode.com/gh_mirrors/ya/yalantinglibs.git
check_success "Failed to clone yalantinglibs"

# Build and install yalantinglibs
cd yalantinglibs
check_success "Failed to change to yalantinglibs directory"

# Checkout version 0.5.5
echo "Checking out yalantinglibs version 0.5.5..."
git checkout 0.5.5
check_success "Failed to checkout yalantinglibs version 0.5.5"

mkdir -p build
check_success "Failed to create build directory"

cd build
check_success "Failed to change to build directory"

echo "Configuring yalantinglibs..."
cmake .. -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF -DYLT_ENABLE_IBV=ON 
check_success "Failed to configure yalantinglibs"

echo "Building yalantinglibs (using $(nproc) cores)..."
cmake --build . -j$(nproc)
check_success "Failed to build yalantinglibs"

echo "Installing yalantinglibs..."
cmake --install .
check_success "Failed to install yalantinglibs"

# Patch the config.cmake file
CONFIG_FILE="${INSTALL_PREFIX}/lib/cmake/yalantinglibs/config.cmake"
if [ -f "${CONFIG_FILE}" ]; then
    sed -i '54s/target_link_libraries(${ylt_target_name} -libverbs)/target_link_libraries(${ylt_target_name} INTERFACE -libverbs)/' "${CONFIG_FILE}"
    check_success "Failed to patch yalantinglibs config.cmake"
fi

print_success "yalantinglibs installed successfully to ${INSTALL_PREFIX}"


# Install tvm-ffi
print_section "Installing tvm-ffi"

TVM_FFI_REPO="https://github.com/apache/tvm-ffi.git"
TVM_FFI_BUILD_TYPE="RelWithDebInfo"
TVM_FFI_TMP_DIR="$(mktemp -d -t tvm-ffi-XXXXXXXX)"
TVM_FFI_SRC_DIR="${TVM_FFI_TMP_DIR}/tvm-ffi"
TVM_FFI_BUILD_DIR="${TVM_FFI_SRC_DIR}/build_cpp"

echo "Cloning tvm-ffi from ${TVM_FFI_REPO}"
git clone --depth 1 --recurse-submodules "${TVM_FFI_REPO}" "${TVM_FFI_SRC_DIR}"
check_success "Failed to clone tvm-ffi"

echo "Updating tvm-ffi submodules..."
git -C "${TVM_FFI_SRC_DIR}" submodule update --init --recursive
check_success "Failed to update tvm-ffi submodules"

echo "Configuring tvm-ffi..."
cmake -S "${TVM_FFI_SRC_DIR}" -B "${TVM_FFI_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${TVM_FFI_BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
check_success "Failed to configure tvm-ffi"

echo "Building tvm-ffi target tvm_ffi_shared..."
cmake --build "${TVM_FFI_BUILD_DIR}" --parallel --config "${TVM_FFI_BUILD_TYPE}" --target tvm_ffi_shared
check_success "Failed to build tvm-ffi (tvm_ffi_shared)"

echo "Installing tvm-ffi..."
cmake --install "${TVM_FFI_BUILD_DIR}" --config "${TVM_FFI_BUILD_TYPE}"
check_success "Failed to install tvm-ffi"

rm -rf "${TVM_FFI_TMP_DIR}"
check_success "Failed to cleanup tvm-ffi temporary directory"

print_success "tvm-ffi installed successfully to ${INSTALL_PREFIX}"
