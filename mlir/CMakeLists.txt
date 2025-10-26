# IAM MLIR Dialect

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# TableGen support
set(LLVM_TARGET_DEFINITIONS tablegen/IAMBase.td)
mlir_tablegen(IAMDialect.h.inc -gen-dialect-decls)
mlir_tablegen(IAMDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRIAMIncGen)

add_library(IAMDialect STATIC
  lib/iam/Dialect/IAMDialect.cpp
)

add_dependencies(IAMDialect MLIRIAMIncGen)

# Add build directory to include path (for generated .inc files)
target_include_directories(IAMDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(IAMDialect
  PUBLIC
  MLIRIR
  MLIRSupport
)

# Test executable
add_executable(test_iam_dialect
  test/test_dialect.cpp
)

target_link_libraries(test_iam_dialect
  PRIVATE
  IAMDialect
  MLIRIR
  MLIRSupport
)