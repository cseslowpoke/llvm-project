#ifndef LLVM_TRANSFORMS_UTILS_OFFLOADFUNCATTRIBUTEPASS_H
#define LLVM_TRANSFORMS_UTILS_OFFLOADFUNCATTRIBUTEPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// OffloadFuncAttributePass - Adds NVPTX kernel function attributes (e.g.
/// "nvvm.maxntid") based on host-side launch information from JSON input.
class OffloadFuncAttributePass : public PassInfoMixin<OffloadFuncAttributePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif
