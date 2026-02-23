#ifndef LLVM_TRANSFORMS_UTILS_OFFLOADASSUMEINJECTIONPASS_H
#define LLVM_TRANSFORMS_UTILS_OFFLOADASSUMEINJECTIONPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class OffloadAssumeInjectionPass : public PassInfoMixin<OffloadAssumeInjectionPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif
