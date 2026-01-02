#ifndef LLVM_TRANSFORMS_UTILS_CUDAASSUMEALIGNPASS_H
#define LLVM_TRANSFORMS_UTILS_CUDAASSUMEALIGNPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class CudaAssumeAlignPass : public PassInfoMixin<CudaAssumeAlignPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static StringRef name() { return "CudaAssumeAlignPass"; }
};

} // namespace llvm

#endif
