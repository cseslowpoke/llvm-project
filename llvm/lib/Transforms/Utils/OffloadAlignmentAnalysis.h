//===- OffloadAlignmentAnalysis.h - Alignment analysis for offload -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal implementation for analyzing alignment properties of kernel
// arguments in offload contexts. This is used by OffloadHostAnalysisPass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_UTILS_OFFLOADALIGNMENTANALYSIS_H
#define LLVM_LIB_TRANSFORMS_UTILS_OFFLOADALIGNMENTANALYSIS_H

#include "llvm/ADT/SmallPtrSet.h"
#include <cstdint>
#include <string>

namespace llvm {

class AssumptionCache;
class DominatorTree;
class Function;
class Value;

/// Result of alignment analysis for a single kernel argument.
struct KernelArgAlignmentInfo {
  bool IsCudaMallocDerived = false;
  std::string CudaMallocOutParamName;
  uint64_t KnownAlignment = 0;
};

/// Internal analyzer for alignment properties of kernel arguments.
/// This class is not part of the public API.
class OffloadAlignmentAnalyzer {
public:
  OffloadAlignmentAnalyzer(Function &F, AssumptionCache &AC, DominatorTree &DT)
      : F(F), AC(AC), DT(DT) {}

  /// Analyze alignment for a value that is stored into a kernel argument slot.
  KernelArgAlignmentInfo analyzeArgumentAlignment(Value *SlotValue);

private:
  Function &F;
  AssumptionCache &AC;
  DominatorTree &DT;

  SmallPtrSet<const Value *, 16> CudaMallocOutParams;
  bool CudaMallocOutParamsCached = false;

  void buildCudaMallocOutParams();
  const Value *isDerivedFromCudaMallocOut(Value *V);
};

} // namespace llvm

#endif // LLVM_LIB_TRANSFORMS_UTILS_OFFLOADALIGNMENTANALYSIS_H
