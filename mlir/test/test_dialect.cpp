#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Dialect.h"
#include "iam/Dialect/IAMDialect.hpp"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

int main() {
  MLIRContext context;
  
  // Test: Load IAM dialect
  context.getOrLoadDialect<iam::IAMDialect>();
  
  // Test: Verify it's registered
  Dialect* dialect = context.getLoadedDialect("iam");
  if (!dialect) {
    llvm::errs() << "FAIL: IAM dialect not loaded\n";
    return 1;
  }
  
  llvm::outs() << "✓ IAM dialect loaded successfully\n";
  llvm::outs() << "  Namespace: " << dialect->getNamespace() << "\n";
  
  return 0;
}
