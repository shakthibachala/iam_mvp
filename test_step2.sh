#!/bin/bash
set -e

LLVM_DIR=/usr/local/opt/llvm
MLIR_DIR=/usr/local/opt/llvm/lib/cmake/mlir

echo "=== Building Step 2: IAM Dialect ==="
mkdir -p build && cd build
cmake -DLLVM_DIR=$LLVM_DIR/lib/cmake/llvm \
      -DMLIR_DIR=$MLIR_DIR \
      ..
make 

echo -e "\n=== Testing IAM Dialect ==="
./mlir/test_iam_dialect

echo -e "\n✓ Step 2 Complete: IAM dialect registered"