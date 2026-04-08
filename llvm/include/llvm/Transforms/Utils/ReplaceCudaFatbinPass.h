#ifndef LLVM_TRANSFORMS_UTILS_REPLACECUDAFATBINPASS_H
#define LLVM_TRANSFORMS_UTILS_REPLACECUDAFATBINPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// ReplaceCudaFatbinPass - Replaces the dummy fatbin data embedded in host
/// LLVM IR (in the .nv_fatbin section) with the real fatbin file contents.
/// This enables a two-phase host compilation pipeline where the first phase
/// compiles with a dummy fatbin to generate all CUDA registration code, and
/// the second phase patches in the real fatbin without re-parsing the source.
class ReplaceCudaFatbinPass : public PassInfoMixin<ReplaceCudaFatbinPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif
