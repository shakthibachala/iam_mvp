#include "iam/Dialect/IAMDialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::iam;

// Include generated dialect definitions
#include "IAMDialect.cpp.inc"

// Initialize is called by generated constructor
void IAMDialect::initialize() {
  // Operations will be added in Steps 3-5
}