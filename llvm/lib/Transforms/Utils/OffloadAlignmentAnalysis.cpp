//===- OffloadAlignmentAnalysis.cpp - Alignment analysis for offload -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OffloadAlignmentAnalysis.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

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
    CudaMallocOutParams.insert(CBI->getArgOperand(0)->stripPointerCasts());
  }
  CudaMallocOutParamsCached = true;
}

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

    if (auto *LI = dyn_cast<LoadInst>(Cur)) {
      const Value *PtrOp = LI->getPointerOperand()->stripPointerCasts();
      if (CudaMallocOutParams.contains(PtrOp))
        return PtrOp;
      Worklist.push_back(const_cast<Value *>(PtrOp));
      continue;
    }

    if (auto *GEP = dyn_cast<GetElementPtrInst>(Cur)) {
      Worklist.push_back(GEP->getPointerOperand());
      continue;
    }

    if (auto *PN = dyn_cast<PHINode>(Cur)) {
      for (Value *In : PN->incoming_values())
        Worklist.push_back(In);
      continue;
    }

    if (auto *SI = dyn_cast<SelectInst>(Cur)) {
      Worklist.push_back(SI->getTrueValue());
      Worklist.push_back(SI->getFalseValue());
      continue;
    }
  }

  return nullptr;
}

KernelArgAlignmentInfo
OffloadAlignmentAnalyzer::analyzeArgumentAlignment(Value *SlotValue) {
  KernelArgAlignmentInfo Result;

  auto *SlotAlloca = dyn_cast<AllocaInst>(SlotValue->stripPointerCasts());
  if (!SlotAlloca)
    return Result;

  const DataLayout &DL = F.getParent()->getDataLayout();
  bool FoundStoreIntoSlot = false;
  bool AnyCudaMallocDerived = false;
  uint64_t BestKnownAlignment = 0;
  std::string BestCudaMallocOutName;

  for (Instruction &I : instructions(F)) {
    auto *SI = dyn_cast<StoreInst>(&I);
    if (!SI)
      continue;
    if (SI->getPointerOperand()->stripPointerCasts() != SlotAlloca)
      continue;

    FoundStoreIntoSlot = true;

    Value *RHS = SI->getValueOperand();
    const Value *CudaMallocOut = isDerivedFromCudaMallocOut(RHS);
    if (CudaMallocOut) {
      AnyCudaMallocDerived = true;
      errs() << "         rhs derives from cudaMalloc out param: ";
      CudaMallocOut->print(errs());
      errs() << "\n";

      Align KnownAlign = getKnownAlignment(RHS, DL, SI, &AC, &DT);
      errs() << "         rhs known alignment: " << KnownAlign.value() << "\n";

      if (KnownAlign.value() > BestKnownAlignment) {
        BestKnownAlignment = KnownAlign.value();
        if (CudaMallocOut->hasName())
          BestCudaMallocOutName = CudaMallocOut->getName().str();
      }
    }
  }

  if (!FoundStoreIntoSlot) {
    errs() << "         slot: no store into this alloca found\n";
  } else if (!AnyCudaMallocDerived) {
    errs() << "         slot: not cudaMalloc-derived (based on stores in this "
              "function)\n";
  }

  Result.IsCudaMallocDerived = AnyCudaMallocDerived;
  Result.CudaMallocOutParamName = BestCudaMallocOutName;
  Result.KnownAlignment = BestKnownAlignment;

  return Result;
}
