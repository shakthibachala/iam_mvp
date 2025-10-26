#pragma once

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace iam {

// Forward declare - full definition in .inc file
class IAMDialect;

} // namespace iam
} // namespace mlir

// Include generated dialect declarations (has the class definition)
#include "IAMDialect.h.inc"