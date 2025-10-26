#!/bin/bash
set -e
LLVM_DIR=/usr/local/opt/llvm
MLIR_DIR=/usr/local/opt/llvm/lib/cmake/mlir

mkdir -p build && cd build
cmake -DLLVM_DIR=$LLVM_DIR/lib/cmake/llvm \
      -DMLIR_DIR=$MLIR_DIR \
      ..
make 

# mkdir -p build && cd build
# cmake -DLLVM_DIR=$LLVM_DIR/lib/cmake/llvm -DMLIR_DIR=$MLIR_DIR ..
# make
# make install

# Step 1
# cd ..
# # cd ..
# # python3 -c "from iam import Tensor; t=Tensor([2,3]); print(f'✓ Works: {t.shape()}, size={t.size()}')"
# echo -e "\n=== Testing ==="
# # Add python directory to PYTHONPATH so it can find iam package
# export PYTHONPATH="${PWD}/python:${PYTHONPATH}"

# python3 -c "from iam import Tensor; t=Tensor([2,3]); print(f'✓ Shape: {t.shape()}, Size: {t.size()}')"

# echo -e "\n=== Running unit tests ==="
# python3 -m pytest tests/unit/test_core.py -v

# Step 2
# mkdir -p build && cd build
# cmake -DLLVM_DIR=/usr/local/opt/llvm/lib/cmake/llvm \
#       -DMLIR_DIR=/usr/local/opt/llvm/lib/cmake/mlir \
#       ..
# make

./mlir/test_iam_dialect