//===- OffloadAliasAnalysis.h - Alias analysis for offload ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal implementation for analyzing alias properties of kernel arguments
// in offload contexts. This is used by OffloadHostAnalysisPass to determine
// which kernel arguments can be marked as noalias.
//
// The key insight is that each cudaMalloc call allocates a unique memory
// region, so pointers derived from different cudaMalloc calls cannot alias.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_UTILS_OFFLOADALIASANALYSIS_H
#define LLVM_LIB_TRANSFORMS_UTILS_OFFLOADALIASANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <cstdint>
#include <string>

namespace llvm {

class Function;
class Value;
class CallBase;

/// Result of alias analysis for a single kernel argument.
/// This struct holds information about whether a kernel argument pointer
/// originates from a unique cudaMalloc allocation.
struct KernelArgAliasInfo {
  /// True if the argument value is derived from a cudaMalloc output parameter.
  bool IsFromUniqueCudaMalloc = false;

  /// Unique ID for the cudaMalloc allocation site. Arguments with different
  /// AllocationIDs are guaranteed to not alias (they come from different
  /// cudaMalloc calls).
  unsigned AllocationID = 0;

  /// Name of the cudaMalloc output parameter for debugging purposes.
  std::string CudaMallocOutParamName;
};

/// Internal analyzer for alias properties of kernel arguments.
///
/// This class analyzes values passed to CUDA kernels via cudaLaunchKernel
/// to determine:
/// 1. Whether the value originates from cudaMalloc
/// 2. Which specific cudaMalloc call it came from (allocation ID)
///
/// Two arguments with different allocation IDs are guaranteed to not alias,
/// because each cudaMalloc allocates a distinct memory region.
///
/// This is an internal implementation detail of OffloadHostAnalysisPass.
/// It is NOT part of the public LLVM API - the header lives in lib/, not
/// include/.
class OffloadAliasAnalyzer {
public:
  /// Construct an analyzer for the given function.
  /// @param F The function containing cudaLaunchKernel calls to analyze.
  OffloadAliasAnalyzer(Function &F) : F(F) {}

  /// Analyze alias properties for a value stored into a kernel argument slot.
  ///
  /// Given a value that is stored into the kernel args array (passed as
  /// argument 5 to cudaLaunchKernel), this method:
  /// 1. Checks if the value is an alloca (temporary slot for the argument)
  /// 2. Finds stores into that alloca to get the actual argument value
  /// 3. Traces the value back to see if it derives from cudaMalloc
  /// 4. Assigns a unique allocation ID based on which cudaMalloc it came from
  ///
  /// @param SlotValue The value stored into the kernel args array slot.
  /// @return Alias information for this kernel argument.
  KernelArgAliasInfo analyzeArgumentAlias(Value *SlotValue);

private:
  Function &F;

  /// Map from cudaMalloc output parameter to unique allocation ID.
  /// Each cudaMalloc call gets a unique ID starting from 1.
  DenseMap<const Value *, unsigned> CudaMallocToID;
  unsigned NextAllocationID = 1;
  bool CudaMallocIDsCached = false;

  /// Scan the function for cudaMalloc calls and assign unique IDs.
  void buildCudaMallocIDs();

  /// Check if a value is derived from a cudaMalloc output parameter.
  /// Uses a worklist-based traversal through loads, GEPs, PHIs, and selects.
  /// @param V The value to check.
  /// @return The cudaMalloc output parameter if found, nullptr otherwise.
  const Value *isDerivedFromCudaMallocOut(Value *V);
};

} // namespace llvm

#endif // LLVM_LIB_TRANSFORMS_UTILS_OFFLOADALIASANALYSIS_H
