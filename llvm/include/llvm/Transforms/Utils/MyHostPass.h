#ifndef LLVM_TRANSFORMS_UTILS_MYHOSTPASS_H
#define LLVM_TRANSFORMS_UTILS_MYHOSTPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class MyHostPass : public PassInfoMixin<MyHostPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif
