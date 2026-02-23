#ifndef LLVM_TRANSFORMS_UTILS_MYDEVICEPASS_H
#define LLVM_TRANSFORMS_UTILS_MYDEVICEPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class MyDevicePass : public PassInfoMixin<MyDevicePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif
