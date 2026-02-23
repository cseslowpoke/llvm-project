#ifndef LLVM_TRANSFORMS_UTILS_OFFLOADPARAMATTRIBUTEPASS_H
#define LLVM_TRANSFORMS_UTILS_OFFLOADPARAMATTRIBUTEPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// OffloadParamAttributePass - Adds alignment and other attributes to kernel
/// pointer parameters based on host-side analysis results from JSON input.
class OffloadParamAttributePass : public PassInfoMixin<OffloadParamAttributePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif
