//===----------------------------------------------------------------------===//
//
// Copyright (c) 2025 Lai-YT
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/BlockGridDimensionAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <algorithm>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

using namespace llvm;

namespace {

/// \brief Find all calls to a function with the name \p FuncName in the range
/// of instructions or users \p Range.
template <typename RangeTy>
SmallVector<CallInst *> findAllCallsTo(const RangeTy &Range,
                                       StringRef FuncName) {
  SmallVector<CallInst *> Calls;
  auto Match = [&](CallInst *Call) {
    if (!Call)
      return false;
    Function *Callee = Call->getCalledFunction();
    return Callee && Callee->getName() == FuncName;
  };
  if constexpr (std::is_pointer_v<std::decay_t<decltype(*Range.begin())>>) {
    for (auto *Element : Range) {
      if (auto *Call = dyn_cast<CallInst>(Element); Match(Call))
        Calls.push_back(Call);
    }
  } else {
    for (auto &Element : Range) {
      if (auto *Call = dyn_cast<CallInst>(&Element); Match(Call))
        Calls.push_back(Call);
    }
  }
  return Calls;
}

} // namespace

#define DEBUG_TYPE "block-grid-dimension-analysis"

AnalysisKey BlockGridDimensionAnalysis::Key;

namespace {

Function *findStubOfConfig(CallInst *CallToCudaPushCallConfiguration) {
  LLVM_DEBUG(dbgs() << "Looking for stub function corresponding to "
                    << *CallToCudaPushCallConfiguration << "...\n");
  // The return value of `__cudaPushCallConfiguration` is checked immediately
  // after the call. If non-zero, the code branches to the block that contains
  // the call to the stub function. Example:
  //
  //   %call = call i32 @__cudaPushCallConfiguration(...)
  //   %tobool = icmp ne i32 %call, 0
  //   br i1 %tobool, label %kcall.end, label %kcall.configok
  //
  // kcall.configok:
  //   call void @_Z22__device_stub__kernel1PA32_fPfS1_(...)
  //   br label %kcall.end
  //
  // By following the use chain of the return value, we can locate the basic
  // block containing the stub function call.
  // Find the icmp comparing the call result to 0. After -O3 InstCombine the
  // canonical form may be either `icmp ne %call, 0` (then success branch is
  // the false successor) or `icmp eq %call, 0` (then success branch is the
  // true successor). Both encode the same condition with branches swapped.
  ICmpInst *Cmp = nullptr;
  for (User *U : CallToCudaPushCallConfiguration->users()) {
    if (auto *I = dyn_cast<ICmpInst>(U)) {
      Cmp = I;
      break;
    }
  }
  if (!Cmp)
    return nullptr;
  ICmpInst::Predicate Pred = Cmp->getPredicate();
  if (Pred != ICmpInst::ICMP_NE && Pred != ICmpInst::ICMP_EQ)
    return nullptr;
  auto *Br = dyn_cast<BranchInst>(Cmp->user_back());
  if (!Br || !Br->isConditional())
    return nullptr;
  // Success branch is the one taken when the call returned 0.
  BasicBlock *BB = (Pred == ICmpInst::ICMP_NE) ? Br->getSuccessor(1)
                                               : Br->getSuccessor(0);
  if (!BB)
    return nullptr;
  // Skip leading intrinsic / debug / lifetime calls that may precede the
  // actual stub call (especially at -O3 with debug info preserved). Also
  // skip CUDA runtime helpers (__cudaPopCallConfiguration, cudaLaunchKernel)
  // that appear when the stub has been inlined into the caller.
  for (auto &&Inst : *BB) {
    auto *Call = dyn_cast<CallInst>(&Inst);
    if (!Call)
      continue;
    Function *Stub = Call->getCalledFunction();
    if (!Stub || Stub->isIntrinsic())
      continue;
    StringRef Name = Stub->getName();
    if (Name == "__cudaPopCallConfiguration" || Name == "cudaLaunchKernel")
      continue;
    LLVM_DEBUG(dbgs() << "Found stub function: " << Stub->getName() << "\n");
    return Stub;
  }
  LLVM_DEBUG(dbgs() << "No stub function found.\n");
  return nullptr;
}

raw_ostream &operator<<(raw_ostream &OS, const Dim3 &Dim) {
  // It's common that the dimensions are 1, 32, or 64.
  // We pad with space to 2 digits for alignment.
  OS << formatv("({0,2}, {1,2}, {2,2})", Dim.X, Dim.Y, Dim.Z);
  return OS;
}

BlockGridDim findBlockGridDim(CallInst *CallToCudaPushCallConfiguration) {
  // The type `dim3` is defined as:
  //     struct dim3 {
  //         unsigned int x, y, z;
  //     };
  // However, such struct is coerced to `{ i64, i32 }` when passed to
  // `__cudaPushCallConfiguration`. This is likely due to the C calling
  // convention:
  //
  // [6] %gridDim.tmp = alloca %struct.dim3, align 4
  //     %blockDim = alloca %struct.dim3, align 4
  // [4] %gridDim.coerce = alloca { i64, i32 }, align 4
  //     %blockDim.coerce = alloca { i64, i32 }, align 4
  //     ...
  //   ; We can get the values of the grid and block dimensions from the
  //   ; call to the constructor of `dim3`.
  // [7] call void @_ZN4dim3C2Ejjj(
  //         ptr noundef nonnull align 4 dereferenceable(12) %gridDim.tmp,
  //         i32 noundef 1, i32 noundef 1, i32 noundef 1)
  //     call void @_ZN4dim3C2Ejjj(
  //         ptr noundef nonnull align 4 dereferenceable(12) %blockDim,
  //         i32 noundef 32, i32 noundef 32, i32 noundef 1)
  //   ; The coercion is done by copying the values of the struct
  //   ; to the coerced type through `llvm.memcpy`.
  // [5] call void @llvm.memcpy.p0.p0.i64(
  //         ptr align 4 %gridDim.coerce, ptr align 4 %gridDim.tmp,
  //         i64 12, i1 false)
  //   ; The coerced values is then loaded and passed as the parameters.
  // [3] %2 = getelementptr inbounds { i64, i32 },
  //         ptr %gridDim.coerce, i32 0, i32 0
  // [2] %3 = load i64, ptr %2, align 4
  //     %4 = getelementptr inbounds { i64, i32 },
  //         ptr %gridDim.coerce, i32 0, i32 1
  //     %5 = load i32, ptr %4, align 4
  //     call void @llvm.memcpy.p0.p0.i64(
  //         ptr align 4 %blockDim.coerce, ptr align 4 %blockDim,
  //         i64 12, i1 false)
  //     %6 = getelementptr inbounds { i64, i32 },
  //         ptr %blockDim.coerce, i32 0, i32 0
  //     %7 = load i64, ptr %6, align 4
  //     %8 = getelementptr inbounds { i64, i32 },
  //         ptr %blockDim.coerce, i32 0, i32 1
  //     %9 = load i32, ptr %8, align 4
  //   ; The first two parameters are gridDim and the second two
  //   ; parameters are blockDim.
  //     %call10 = call i32 @__cudaPushCallConfiguration(
  // [1]     i64 %3, i32 %5,
  //         i64 %7, i32 %9,
  //         i64 noundef 0, ptr noundef null)
  //
  // By tracing how the values are loaded and constructed, we can recover the
  // gridDim and blockDim values from the call arguments.
  // The [*] comments indicate the path we track to find the gridDim, from [1]
  // to [7].
  //
  // NOTE: Deriving launch dimensions solely from the constructor call is
  // error-prone, since the underlying `dim3` object may be modified after
  // construction and before being copied into the coerced value passed to
  // `__cudaPushCallConfiguration` (e.g., via field stores or a subsequent
  // `llvm.memcpy`/`llvm.memset`).
  // If such modifications are possible between the constructor and the capture
  // used for the launch configuration, the analysis conservatively treats the
  // corresponding dimension as unknown. As a result, launch dimensions must be
  // constructed entirely via the constructor to be considered known.

  // Fast path for optimized IR (e.g. -O3): the call args may already be
  // ConstantInts after SROA/InstCombine fold away the alloca+load+memcpy
  // dance. dim3 {x,y} is coerced into i64 (low 32 = x, high 32 = y); z is a
  // separate i32 arg. Decode directly when possible.
  auto FindDimFromConstantArgs = [](Value *XY, Value *Z) -> std::optional<Dim3> {
    auto *XYC = dyn_cast<ConstantInt>(XY);
    auto *ZC = dyn_cast<ConstantInt>(Z);
    if (!XYC && !ZC)
      return std::nullopt;
    Dim3 D;
    if (XYC) {
      uint64_t V = XYC->getZExtValue();
      D.X = static_cast<unsigned>(V & 0xFFFFFFFFu);
      D.Y = static_cast<unsigned>(V >> 32);
    }
    if (ZC)
      D.Z = static_cast<unsigned>(ZC->getZExtValue());
    return D;
  };

  auto FindDim = [](Value *ZArg) -> Dim3 {
    auto *DimZPtr = dyn_cast_or_null<LoadInst>(ZArg);
    if (!DimZPtr)
      return {};
    auto &DimZ = *DimZPtr;
    // [3]
    auto *DimZGEPPtr = dyn_cast<GetElementPtrInst>(DimZ.getPointerOperand());
    if (!DimZGEPPtr)
      return {};
    auto &DimZGEP = *DimZGEPPtr;
    // [4]
    auto *DimCoercePtr = dyn_cast<AllocaInst>(DimZGEP.getPointerOperand());
    if (!DimCoercePtr)
      return {};
    auto &DimCoerce = *DimCoercePtr;
    LLVMContext &Ctx = DimCoerce.getContext();
    Type *ArgTys[] = {
        PointerType::getUnqual(Ctx),
        PointerType::getUnqual(Ctx),
        IntegerType::getInt64Ty(Ctx),
    };
    auto *LLVMMemcpy = Intrinsic::getDeclaration(DimCoerce.getModule(),
                                                 Intrinsic::memcpy, ArgTys);
    auto *LLVMMemset = Intrinsic::getDeclaration(DimCoerce.getModule(),
                                                 Intrinsic::memset, ArgTys);
    // The call to the `llvm.memcpy` is one of the users of the coerced
    // struct, and should be the only one that is a call.
    CallInst *CallToMemCpy = nullptr;
    for (auto &&User : DimCoerce.users()) {
      // [5]
      if (auto *Call = dyn_cast<CallInst>(User)) {
        assert(Call->getCalledFunction() == LLVMMemcpy &&
               "The call to the memcpy should be the only user that is a call");
        assert(Call->getOperand(0) == &DimCoerce &&
               "The first operand of the memcpy should be the coerced struct");
        LLVM_DEBUG(dbgs() << "  Found call to llvm.memcpy: " << *Call << "\n");
        CallToMemCpy = Call;
      }
    }
    if (!CallToMemCpy)
      return {};
    // [6]
    auto *DimTmpPtr = dyn_cast<AllocaInst>(CallToMemCpy->getArgOperand(1));
    if (!DimTmpPtr)
      return {};
    auto &DimTmp = *DimTmpPtr;
    // If the dim is constructed separately, e.g.,
    //     dim3 gridDim(32);
    //     ...
    //     kernel<<<gridDim, ...>>>(...);
    // There will be an additional `llvm.memcpy` call to copy the values from
    // `gridDim` to the temporary struct `DimTmp`.
    // Identify this by checking if the temporary struct is used as the
    // destination of `llvm.memcpy`.
    Value *Dim = nullptr;
    CallInst *CallToCopyToTemp = nullptr;
    for (auto &&User : DimTmp.users()) {
      if (auto *Call = dyn_cast<CallInst>(User);
          Call && Call->getCalledFunction() == LLVMMemcpy &&
          Call->getArgOperand(0) == &DimTmp) {
        LLVM_DEBUG(dbgs() << "  Found copy to temp struct: " << *Call << "\n");
        Dim = Call->getArgOperand(1);
        CallToCopyToTemp = Call;
      }
    }

    Value *DimObj = (Dim ?: &DimTmp);
    Instruction *CaptureInst =
        (Dim ? static_cast<Instruction *>(CallToCopyToTemp)
             : static_cast<Instruction *>(CallToMemCpy));
    assert(CaptureInst && "Expected a capture instruction for dim3 value");

    auto IsDerivedFromDimObj = [&](Value *Ptr) {
      if (!Ptr || !Ptr->getType()->isPointerTy())
        return false;
      return getUnderlyingObject(Ptr) == DimObj;
    };

    auto IsModifiedBetweenCtorAndCapture = [&](CallInst *CtorCall) {
      if (!CtorCall || !CtorCall->getFunction())
        return false;

      for (auto &I : instructions(*CtorCall->getFunction())) {
        if (&I == CtorCall || &I == CaptureInst)
          continue;

        bool IsModified = false;
        if (auto *Store = dyn_cast<StoreInst>(&I)) {
          IsModified = IsDerivedFromDimObj(Store->getPointerOperand());
        } else if (auto *Call = dyn_cast<CallInst>(&I)) {
          if (Function *Callee = Call->getCalledFunction()) {
            if ((Callee == LLVMMemcpy || Callee == LLVMMemset) &&
                Call->arg_size() >= 1) {
              IsModified = IsDerivedFromDimObj(Call->getArgOperand(0));
            }
          }
        }
        if (IsModified)
          return true;
      }
      return false;
    };

    for (CallInst *CtorCallForDim :
         findAllCallsTo(DimObj->users(), "_ZN4dim3C2Ejjj")) {
      LLVM_DEBUG(dbgs() << "  Found dim3 constructor call: " << *CtorCallForDim
                        << "\n");
      if (IsModifiedBetweenCtorAndCapture(CtorCallForDim)) {
        LLVM_DEBUG(
            dbgs()
            << "  dim3 is modified after construction; treating as unknown.\n");
        return {};
      }

      auto GetZExtValueOrUnknown = [](Value *V) {
        auto *CI = dyn_cast<ConstantInt>(V);
        return CI ? std::make_optional(CI->getZExtValue()) : std::nullopt;
      };

      Dim3 Dim = {
          .X = GetZExtValueOrUnknown(CtorCallForDim->getArgOperand(1)),
          .Y = GetZExtValueOrUnknown(CtorCallForDim->getArgOperand(2)),
          .Z = GetZExtValueOrUnknown(CtorCallForDim->getArgOperand(3)),
      };
      LLVM_DEBUG(dbgs() << "  Found dimension: " << Dim << "\n");
      return Dim;
    }
    LLVM_DEBUG(dbgs() << "  No call to the constructor of dim3 found.\n");
    return {};
  };

  Value *GridXY = CallToCudaPushCallConfiguration->getArgOperand(0);
  Value *GridZ = CallToCudaPushCallConfiguration->getArgOperand(1);
  Value *BlockXY = CallToCudaPushCallConfiguration->getArgOperand(2);
  Value *BlockZ = CallToCudaPushCallConfiguration->getArgOperand(3);

  LLVM_DEBUG(dbgs() << "Looking for gridDim...\n");
  Dim3 GridDim = FindDimFromConstantArgs(GridXY, GridZ).value_or(FindDim(GridZ));
  LLVM_DEBUG(dbgs() << "Looking for blockDim...\n");
  Dim3 BlockDim =
      FindDimFromConstantArgs(BlockXY, BlockZ).value_or(FindDim(BlockZ));
  return {BlockDim, GridDim};
}

BlockGridDim unionBlockGridDim(const BlockGridDim &LHS,
                               const BlockGridDim &RHS) {
  auto UnionComponent =
      [](std::optional<unsigned> LHS,
         std::optional<unsigned> RHS) -> std::optional<unsigned> {
    if (!LHS || !RHS)
      return std::nullopt;
    return std::max(*LHS, *RHS);
  };
  auto UnionDim3 = [&](const Dim3 &LHS, const Dim3 &RHS) -> Dim3 {
    return {
        .X = UnionComponent(LHS.X, RHS.X),
        .Y = UnionComponent(LHS.Y, RHS.Y),
        .Z = UnionComponent(LHS.Z, RHS.Z),
    };
  };
  return {
      .BlockDim = UnionDim3(LHS.BlockDim, RHS.BlockDim),
      .GridDim = UnionDim3(LHS.GridDim, RHS.GridDim),
  };
}

} // namespace

BlockGridDimensionAnalysis::Result
BlockGridDimensionAnalysis::run(Module &M, ModuleAnalysisManager &MAM) {
  // Only run on the host side for X86 targets.
  if (!Triple(M.getTargetTriple()).isX86()) {
    LLVM_DEBUG(dbgs() << "Not an X86 target. Skipped.\n");
    return Result();
  }

  // The kernel launch parameters are set through the
  // call to `__cudaPushCallConfiguration`:
  //     int __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
  //         size_t sharedMem, cudaStream_t stream)
  auto *CudaPushCallConfiguration =
      M.getFunction("__cudaPushCallConfiguration");
  if (!CudaPushCallConfiguration) {
    LLVM_DEBUG(dbgs() << "No __cudaPushCallConfiguration function found."
                         "Seems like there's no kernel launch in the module. "
                         "Skipped.\n");
    return Result();
  }

  // Group push-call sites by the function that contains them. We key results
  // by the caller function so that downstream consumers (e.g.,
  // OffloadHostAnalysisPass) can join on `cudaLaunchKernel`'s containing
  // function, which equals the caller of push when the stub has been inlined
  // (typical at -O3) and equals the stub itself when it has not.
  DenseMap<Function *, SmallVector<CallInst *, 2>> ByCaller;
  DenseMap<Function *, SmallVector<Function *, 2>> CallerToStubs;
  for (auto *User : CudaPushCallConfiguration->users()) {
    auto *Call = dyn_cast<CallInst>(User);
    if (!Call) {
      LLVM_DEBUG(
          dbgs() << "Skipped non-call user of __cudaPushCallConfiguration: "
                 << *User << "\n");
      continue;
    }
    Function *Caller = Call->getFunction();
    if (!Caller)
      continue;
    ByCaller[Caller].push_back(Call);
    if (Function *Stub = findStubOfConfig(Call))
      CallerToStubs[Caller].push_back(Stub);
  }

  // Compute the union of grid/block dims across multiple push sites in the
  // same caller, then publish the result under both the caller name and (when
  // available and distinct) the resolved stub names.
  Result BlockGridDims;
  for (auto &&[Caller, CallsToConfiguration] : ByCaller) {
    LLVM_DEBUG(dbgs() << "For caller function '" << Caller->getName() << "':\n");
    bool HasAny = false;
    BlockGridDim Union;
    for (CallInst *CallToConfiguration : CallsToConfiguration) {
      auto This = findBlockGridDim(CallToConfiguration);
      Union = HasAny ? unionBlockGridDim(Union, This) : This;
      HasAny = true;
    }
    BlockGridDims[Caller->getName().str()] = Union;
    auto It = CallerToStubs.find(Caller);
    if (It != CallerToStubs.end()) {
      for (Function *Stub : It->second) {
        if (Stub != Caller)
          BlockGridDims[Stub->getName().str()] = Union;
      }
    }
  }
  return BlockGridDims;
}

PreservedAnalyses
BlockGridDimensionAnalysisPrinter::run(Module &M, ModuleAnalysisManager &MAM) {
  OS << "Block and grid dimensions analysis for module '" << M.getName()
     << "':\n";
  auto &BlockGridDims = MAM.getResult<BlockGridDimensionAnalysis>(M);
  for (auto &&[StubName, BlockGridDim] : BlockGridDims) {
    OS << "  Stub: " << StubName << "\n";
    OS << "    Block: ";
    OS << BlockGridDim.BlockDim << "\n";
    OS << "    Grid:  ";
    OS << BlockGridDim.GridDim << "\n";
  }
  return PreservedAnalyses::all();
}

namespace llvm::json {

static Value toJSON(const Dim3 &Dim) {
  return Object{
      {"X", Dim.X},
      {"Y", Dim.Y},
      {"Z", Dim.Z},
  };
}

static Value toJSON(const BlockGridDimensionAnalysis::Result &Result) {
  Array Array;
  for (const auto &[StubName, BGD] : Result) {
    Array.push_back(Object{
        {"Stub", StubName},
        {"BlockDim", toJSON(BGD.BlockDim)},
        {"GridDim", toJSON(BGD.GridDim)},
    });
  }
  return Array;
}

static bool fromJSON(const Value &V, Dim3 &Out, Path P) {
  ObjectMapper O(V, P);
  if (!O)
    return false;
  return O.map("X", Out.X) && O.map("Y", Out.Y) && O.map("Z", Out.Z);
}

static bool fromJSON(const Value &V, BlockGridDim &Out, Path P) {
  ObjectMapper O(V, P);
  if (!O)
    return false;
  return O.map("BlockDim", Out.BlockDim) && O.map("GridDim", Out.GridDim);
}

static bool fromJSON(const Value &V, BlockGridDimensionAnalysis::Result &Out,
                     Path P) {
  if (auto *A = V.getAsArray()) {
    for (size_t I = 0; I < A->size(); ++I) {
      if (auto *O = (*A)[I].getAsObject()) {
        auto StubName = O->getString("Stub");
        if (!StubName)
          return false;
        if (!fromJSON((*A)[I], Out[(*StubName).str()], P.index(I)))
          return false;
      } else {
        P.report("expected object");
        return false;
      }
    }
    return true;
  }
  P.report("expected array");
  return false;
}

} // namespace llvm::json

PreservedAnalyses BlockGridDimensionAnalysisJSONExporter::run(
    Module &M, ModuleAnalysisManager &MAM) {
  auto &BlockGridDims = MAM.getResult<BlockGridDimensionAnalysis>(M);
  OS << formatv("{0:2}", json::toJSON(BlockGridDims)) << "\n";
  return PreservedAnalyses::all();
}

Expected<BlockGridDimensionAnalysis::Result>
BlockGridDimensionAnalysisJSONImporter::fromJSON(StringRef JSON) {
  if (JSON.empty())
    return BlockGridDimensionAnalysis::Result();

  Expected<BlockGridDimensionAnalysis::Result> BGD =
      json::parse<BlockGridDimensionAnalysis::Result>(JSON);
  if (!BGD)
    return BGD.takeError();
  return *BGD;
}

Expected<BlockGridDimensionAnalysis::Result>
BlockGridDimensionAnalysisJSONImporter::fromFile(
    llvm::StringRef Filename) {
  auto BufferOrErr = MemoryBuffer::getFile(Filename, /*IsText=*/true);
  if (!BufferOrErr)
    return errorCodeToError(BufferOrErr.getError());
  return fromJSON((*BufferOrErr)->getBuffer());
}
