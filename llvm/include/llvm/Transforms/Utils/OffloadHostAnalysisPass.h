#ifndef LLVM_TRANSFORMS_UTILS_OFFLOADHOSTANALYSISPASS_H
#define LLVM_TRANSFORMS_UTILS_OFFLOADHOSTANALYSISPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class OffloadHostAnalysisPass : public PassInfoMixin<OffloadHostAnalysisPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif
