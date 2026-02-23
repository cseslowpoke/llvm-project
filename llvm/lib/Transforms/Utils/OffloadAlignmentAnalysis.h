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
/// This struct holds information about whether a kernel argument pointer
/// originates from cudaMalloc and what alignment can be inferred.
struct KernelArgAlignmentInfo {
  /// True if the argument value is derived from a cudaMalloc output parameter.
  /// cudaMalloc guarantees at least 256-byte alignment for allocated memory.
  bool IsCudaMallocDerived = false;

  /// Name of the cudaMalloc output parameter (e.g., the pointer-to-pointer
  /// passed to cudaMalloc) if IsCudaMallocDerived is true.
  std::string CudaMallocOutParamName;

  /// The best known alignment (in bytes) for this argument, computed via
  /// LLVM's getKnownAlignment. This considers llvm.assume hints and other
  /// alignment information available at the store site.
  uint64_t KnownAlignment = 0;
};

/// Internal analyzer for alignment properties of kernel arguments.
///
/// This class analyzes values passed to CUDA kernels via cudaLaunchKernel
/// to determine:
/// 1. Whether the value originates from cudaMalloc (which guarantees alignment)
/// 2. What alignment can be inferred using LLVM's value tracking
///
/// This is an internal implementation detail of OffloadHostAnalysisPass.
/// It is NOT part of the public LLVM API - the header lives in lib/, not include/.
class OffloadAlignmentAnalyzer {
public:
  /// Construct an analyzer for the given function.
  /// @param F   The function containing cudaLaunchKernel calls to analyze.
  /// @param AC  AssumptionCache for the function (used by getKnownAlignment).
  /// @param DT  DominatorTree for the function (used by getKnownAlignment).
  OffloadAlignmentAnalyzer(Function &F, AssumptionCache &AC, DominatorTree &DT)
      : F(F), AC(AC), DT(DT) {}

  /// Analyze alignment for a value stored into a kernel argument slot.
  ///
  /// Given a value that is stored into the kernel args array (passed as
  /// argument 5 to cudaLaunchKernel), this method:
  /// 1. Checks if the value is an alloca (temporary slot for the argument)
  /// 2. Finds stores into that alloca to get the actual argument value
  /// 3. Traces the value back to see if it derives from cudaMalloc
  /// 4. Computes the known alignment using LLVM's getKnownAlignment
  ///
  /// @param SlotValue The value stored into the kernel args array slot.
  /// @return Alignment information for this kernel argument.
  KernelArgAlignmentInfo analyzeArgumentAlignment(Value *SlotValue);

private:
  Function &F;
  AssumptionCache &AC;
  DominatorTree &DT;

  /// Cached set of cudaMalloc output parameters (the pointer-to-pointer
  /// first argument of cudaMalloc calls) found in the function.
  SmallPtrSet<const Value *, 16> CudaMallocOutParams;
  bool CudaMallocOutParamsCached = false;

  /// Scan the function for cudaMalloc calls and cache their output parameters.
  void buildCudaMallocOutParams();

  /// Check if a value is derived from a cudaMalloc output parameter.
  /// Uses a worklist-based traversal through loads, GEPs, PHIs, and selects.
  /// @param V The value to check.
  /// @return The cudaMalloc output parameter if found, nullptr otherwise.
  const Value *isDerivedFromCudaMallocOut(Value *V);
};

} // namespace llvm

#endif // LLVM_LIB_TRANSFORMS_UTILS_OFFLOADALIGNMENTANALYSIS_H
