//===- IAMBase.td - IAM dialect base -----------*- tablegen -*-===//

#ifndef IAM_BASE
#define IAM_BASE

include "mlir/IR/OpBase.td"

def IAM_Dialect : Dialect {
  let name = "iam";
  let summary = "Interpretable AI with MLIR";
  let cppNamespace = "::mlir::iam";
}

#endif // IAM_BASE