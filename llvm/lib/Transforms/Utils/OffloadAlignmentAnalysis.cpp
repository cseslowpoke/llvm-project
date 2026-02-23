//===- OffloadAlignmentAnalysis.cpp - Alignment analysis for offload -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OffloadAlignmentAnalysis.h"

#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "offload-host-analysis"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

/// Scan the function for cudaMalloc calls and cache their output parameters.
///
/// cudaMalloc signature: cudaError_t cudaMalloc(void **devPtr, size_t size)
/// The first argument (devPtr) is a pointer-to-pointer that receives the
/// allocated device memory address. We track these to identify kernel
/// arguments that originate from cudaMalloc allocations.
void OffloadAlignmentAnalyzer::buildCudaMallocOutParams() {
  if (CudaMallocOutParamsCached)
    return;

  CudaMallocOutParams.clear();
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
    CudaMallocOutParams.insert(CBI->getArgOperand(0)->stripPointerCasts());
  }
  CudaMallocOutParamsCached = true;
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
const Value *OffloadAlignmentAnalyzer::isDerivedFromCudaMallocOut(Value *V) {
  buildCudaMallocOutParams();

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
    // Pattern: %ptr = load ptr, ptr %devPtr  (where %devPtr was passed to cudaMalloc)
    if (auto *LI = dyn_cast<LoadInst>(Cur)) {
      const Value *PtrOp = LI->getPointerOperand()->stripPointerCasts();
      if (CudaMallocOutParams.contains(PtrOp))
        return PtrOp;  // Found! This value comes from cudaMalloc.
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

/// Analyze alignment for a kernel argument slot value.
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
/// 3. Computes alignment using getKnownAlignment
KernelArgAlignmentInfo
OffloadAlignmentAnalyzer::analyzeArgumentAlignment(Value *SlotValue) {
  KernelArgAlignmentInfo Result;

  // The slot value should be an alloca that holds the actual argument pointer.
  auto *SlotAlloca = dyn_cast<AllocaInst>(SlotValue->stripPointerCasts());
  if (!SlotAlloca)
    return Result;

  const DataLayout &DL = F.getParent()->getDataLayout();
  bool FoundStoreIntoSlot = false;
  bool AnyCudaMallocDerived = false;
  uint64_t BestKnownAlignment = 0;
  std::string BestCudaMallocOutName;

  // Scan all stores in the function to find stores into this slot alloca.
  // There may be multiple stores (e.g., in different branches), so we track
  // the best (highest) alignment found.
  for (Instruction &I : instructions(F)) {
    auto *SI = dyn_cast<StoreInst>(&I);
    if (!SI)
      continue;
    // Check if this store writes to our slot alloca
    if (SI->getPointerOperand()->stripPointerCasts() != SlotAlloca)
      continue;

    FoundStoreIntoSlot = true;

    // Get the value being stored (the actual kernel argument)
    Value *RHS = SI->getValueOperand();

    // Check if this value derives from cudaMalloc
    const Value *CudaMallocOut = isDerivedFromCudaMallocOut(RHS);
    if (CudaMallocOut) {
      AnyCudaMallocDerived = true;
      LLVM_DEBUG(dbgs() << "         rhs derives from cudaMalloc out param: ";
                 CudaMallocOut->print(dbgs()); dbgs() << "\n");

      // Use LLVM's getKnownAlignment to compute alignment at this store site.
      // This considers llvm.assume intrinsics and other alignment hints.
      Align KnownAlign = getKnownAlignment(RHS, DL, SI, &AC, &DT);
      LLVM_DEBUG(dbgs() << "         rhs known alignment: " << KnownAlign.value() << "\n");

      // Keep track of the best alignment seen across all stores
      if (KnownAlign.value() > BestKnownAlignment) {
        BestKnownAlignment = KnownAlign.value();
        if (CudaMallocOut->hasName())
          BestCudaMallocOutName = CudaMallocOut->getName().str();
      }
    }
  }

  if (!FoundStoreIntoSlot) {
    LLVM_DEBUG(dbgs() << "         slot: no store into this alloca found\n");
  } else if (!AnyCudaMallocDerived) {
    LLVM_DEBUG(dbgs() << "         slot: not cudaMalloc-derived (based on stores in this "
                         "function)\n");
  }

  Result.IsCudaMallocDerived = AnyCudaMallocDerived;
  Result.CudaMallocOutParamName = BestCudaMallocOutName;
  Result.KnownAlignment = BestKnownAlignment;

  return Result;
}
