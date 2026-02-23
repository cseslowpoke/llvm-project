//===- OffloadAliasAnalysis.cpp - Alias analysis for offload --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OffloadAliasAnalysis.h"

#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "offload-host-analysis"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

/// Scan the function for cudaMalloc calls and assign unique IDs.
///
/// cudaMalloc signature: cudaError_t cudaMalloc(void **devPtr, size_t size)
/// Each cudaMalloc call allocates a unique memory region, so we assign
/// a unique ID to each call site. Arguments derived from different
/// cudaMalloc calls are guaranteed to not alias.
void OffloadAliasAnalyzer::buildCudaMallocIDs() {
  if (CudaMallocIDsCached)
    return;

  CudaMallocToID.clear();
  NextAllocationID = 1;

  for (Instruction &I : instructions(F)) {
    auto *CBI = dyn_cast<CallBase>(&I);
    if (!CBI)
      continue;
    Function *Callee = CBI->getCalledFunction();
    if (!Callee)
      continue;
    if (Callee->getName() != "cudaMalloc")
      continue;
    if (CBI->arg_size() < 1)
      continue;

    // Store the first argument (devPtr) after stripping pointer casts.
    // This is the location where cudaMalloc writes the allocated pointer.
    // Assign a unique ID to this allocation site.
    const Value *OutParam = CBI->getArgOperand(0)->stripPointerCasts();
    if (!CudaMallocToID.count(OutParam)) {
      CudaMallocToID[OutParam] = NextAllocationID++;
      LLVM_DEBUG(dbgs() << "OffloadAliasAnalyzer: cudaMalloc out param ID="
                        << (NextAllocationID - 1) << " for: ";
                 OutParam->print(dbgs()); dbgs() << "\n");
    }
  }
  CudaMallocIDsCached = true;
}

/// Trace a value back through the IR to check if it derives from cudaMalloc.
///
/// This uses a worklist-based backward traversal to handle:
/// - LoadInst: Check if loading from a cudaMalloc output parameter
/// - GetElementPtrInst: Follow the base pointer
/// - PHINode: Check all incoming values
/// - SelectInst: Check both true and false values
///
/// The traversal is limited to 128 steps to avoid infinite loops.
const Value *OffloadAliasAnalyzer::isDerivedFromCudaMallocOut(Value *V) {
  buildCudaMallocIDs();

  SmallPtrSet<const Value *, 32> Visited;
  SmallVector<Value *, 16> Worklist;
  Worklist.push_back(V);

  unsigned Steps = 0;
  while (!Worklist.empty() && Steps++ < 128) {
    Value *Cur = Worklist.pop_back_val();
    Cur = Cur->stripPointerCasts();
    if (!Visited.insert(Cur).second)
      continue;

    // If this is a load, check if it's loading from a cudaMalloc output param.
    // Pattern: %ptr = load ptr, ptr %devPtr  (where %devPtr was passed to
    // cudaMalloc)
    if (auto *LI = dyn_cast<LoadInst>(Cur)) {
      const Value *PtrOp = LI->getPointerOperand()->stripPointerCasts();
      if (CudaMallocToID.count(PtrOp))
        return PtrOp; // Found! This value comes from cudaMalloc.
      Worklist.push_back(const_cast<Value *>(PtrOp));
      continue;
    }

    // Follow GEP base pointers (e.g., array element access)
    if (auto *GEP = dyn_cast<GetElementPtrInst>(Cur)) {
      Worklist.push_back(GEP->getPointerOperand());
      continue;
    }

    // For PHI nodes, check all incoming values
    if (auto *PN = dyn_cast<PHINode>(Cur)) {
      for (Value *In : PN->incoming_values())
        Worklist.push_back(In);
      continue;
    }

    // For select instructions, check both branches
    if (auto *SI = dyn_cast<SelectInst>(Cur)) {
      Worklist.push_back(SI->getTrueValue());
      Worklist.push_back(SI->getFalseValue());
      continue;
    }
  }

  return nullptr;
}

/// Analyze alias properties for a kernel argument slot value.
///
/// The kernel args array in cudaLaunchKernel is typically set up like:
///   %args = alloca [N x ptr]           ; Array of argument pointers
///   %slot0 = alloca ptr                 ; Temporary slot for arg 0
///   store ptr %actual_arg, ptr %slot0   ; Store actual argument
///   %elem0 = getelementptr [N x ptr], ptr %args, i64 0, i64 0
///   store ptr %slot0, ptr %elem0        ; Store slot pointer into args array
///
/// This method receives %slot0 (the SlotValue) and:
/// 1. Finds stores into %slot0 to get %actual_arg
/// 2. Checks if %actual_arg derives from cudaMalloc
/// 3. Returns the allocation ID if found
KernelArgAliasInfo
OffloadAliasAnalyzer::analyzeArgumentAlias(Value *SlotValue) {
  KernelArgAliasInfo Result;

  // The slot value should be an alloca that holds the actual argument pointer.
  auto *SlotAlloca = dyn_cast<AllocaInst>(SlotValue->stripPointerCasts());
  if (!SlotAlloca)
    return Result;

  // Scan all stores in the function to find stores into this slot alloca.
  for (Instruction &I : instructions(F)) {
    auto *SI = dyn_cast<StoreInst>(&I);
    if (!SI)
      continue;
    // Check if this store writes to our slot alloca
    if (SI->getPointerOperand()->stripPointerCasts() != SlotAlloca)
      continue;

    // Get the value being stored (the actual kernel argument)
    Value *RHS = SI->getValueOperand();

    // Check if this value derives from cudaMalloc
    const Value *CudaMallocOut = isDerivedFromCudaMallocOut(RHS);
    if (CudaMallocOut) {
      Result.IsFromUniqueCudaMalloc = true;
      Result.AllocationID = CudaMallocToID[CudaMallocOut];
      if (CudaMallocOut->hasName())
        Result.CudaMallocOutParamName = CudaMallocOut->getName().str();

      LLVM_DEBUG(dbgs() << "         alias: from cudaMalloc ID="
                        << Result.AllocationID << " (";
                 CudaMallocOut->print(dbgs()); dbgs() << ")\n");

      // Return on first match - if there are multiple stores with different
      // sources, we conservatively cannot determine noalias
      return Result;
    }
  }

  LLVM_DEBUG(dbgs() << "         alias: not from unique cudaMalloc\n");
  return Result;
}
