#ifndef LLVM_TRANSFORMS_UTILS_MYDEVICEPASS2_H
#define LLVM_TRANSFORMS_UTILS_MYDEVICEPASS2_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// MyDevicePass2 - Restores readonly/nocapture attributes for kernel pointer
/// arguments that are only read from. This pass should run AFTER MyDevicePass
/// to fix attribute inference issues caused by llvm.assume insertion.
class MyDevicePass2 : public PassInfoMixin<MyDevicePass2> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif
