#include "llvm/Transforms/Utils/MyHostPass.h"

#include "llvm/IR/Module.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Demangle/Demangle.h"

#include <optional>
#include <string>

using namespace llvm;

static cl::opt<std::string> MyHostPassOutput(
    "my-host-pass-output", cl::Hidden, cl::init(""));

static bool isCudaLaunchKernelCall(CallBase &CB) {
  Function *Callee = CB.getCalledFunction();
  if (!Callee)
    return false;
  StringRef Name = Callee->getName();
  if (Name != "cudaLaunchKernel")
    return false;
  return true;
}

static const Function *resolveKernelFunction(Value *V) {
  SmallPtrSet<const Value *, 32> Visited;
  SmallVector<Value *, 16> Worklist;
  Worklist.push_back(V);

  unsigned Steps = 0;
  while (!Worklist.empty() && Steps++ < 128) {
    Value *Cur = Worklist.pop_back_val();
    if (!Cur)
      continue;
    Cur = Cur->stripPointerCasts();
    if (!Visited.insert(Cur).second)
      continue; 

    if (auto *F = dyn_cast<Function>(Cur))
      return F;

    if (auto *GA = dyn_cast<GlobalAlias>(Cur)) {
      if (const GlobalObject *GO = GA->getAliaseeObject())
        if (const auto *F = dyn_cast<Function>(GO))
          return F;
      continue;
    }

    if (auto *CE = dyn_cast<ConstantExpr>(Cur)) {
      if (CE->getNumOperands() > 0)
        Worklist.push_back(CE->getOperand(0));
      continue;
    }
  }

  return nullptr;
}

struct CudaKernelArgInfo {
  unsigned Index = 0;
  std::string SlotName;
  std::string SlotTypeStr;
  bool IsCudaMallocDerived = false;
  std::string CudaMallocOutParamName;
  uint64_t KnownAlignment = 0;
};

struct CudaLaunchKernelInfo {
  std::string HostFunctionName;
  std::string ResolvedKernelMangled;
  std::string GuessedOriginalKernelName;
  std::string KernelArgsAllocaName;
  SmallVector<CudaKernelArgInfo, 16> Args;
};

static std::optional<CudaLaunchKernelInfo>
dumpCudaLaunchKernelArgs(CallBase &CB, AssumptionCache &AC, DominatorTree &DT) {
  constexpr unsigned KernelArgsIndex = 5;
  Function *Caller = CB.getFunction();
  const DataLayout &DL = Caller->getParent()->getDataLayout();
    
  errs() << "cudaLaunchKernel in function: " << Caller->getName() << "\n";

  if (CB.arg_size() <= KernelArgsIndex) {
    errs() << "  unexpected arg_size=" << CB.arg_size() << " (need >= 6)\n";
    return std::nullopt;
  }

  Value *KernelFn = CB.getArgOperand(0);
  CudaLaunchKernelInfo Info;
  Info.HostFunctionName = Caller->getName().str();
  const Function *KF = resolveKernelFunction(KernelFn);
  if (!KF) {
    errs() << "    resolved kernel: <unknown>\n";
    return std::nullopt;
  }

  errs() << "    resolved kernel: " << KF->getName() << "\n";
  if (KF->getName() == Caller->getName()) {
    errs() << "    caller is stub kernel function\n";
    return std::nullopt;
  }

  Info.ResolvedKernelMangled = KF->getName().str();

  std::string Demangled = demangle(KF->getName().str());
  StringRef DemangledRef(Demangled);
  constexpr StringRef StubMarker = "__device_stub__";
  size_t Pos = DemangledRef.find(StubMarker);
  if (Pos != StringRef::npos) {
    StringRef After = DemangledRef.drop_front(Pos + StubMarker.size());
    StringRef GuessedKernel = After.split('(').first.trim();
    if (!GuessedKernel.empty()) {
      errs() << "    guessed original kernel (demangled): " << GuessedKernel
             << "\n";
      Info.GuessedOriginalKernelName = GuessedKernel.str();
    }
  }

  Value *KernelArgs = CB.getArgOperand(KernelArgsIndex);
  errs() << "  kernel args operand (idx 5):\n";
  errs() << "    name: "
         << (KernelArgs->hasName() ? KernelArgs->getName() : "<unnamed>")
         << "\n";
  errs() << "    type: ";
  KernelArgs->getType()->print(errs());
  errs() << "\n";
  Value *Base = KernelArgs->stripPointerCasts();
  auto *AI = dyn_cast<AllocaInst>(Base);
  if (!AI) {
    errs() << "  kernel args base is not an alloca after stripping casts\n";
    return std::nullopt;
  }

  if (AI->hasName())
    Info.KernelArgsAllocaName = AI->getName().str();

  auto *AT = dyn_cast<ArrayType>(AI->getAllocatedType());
  if (!AT) {
    errs() << "  kernel args alloca is not an array: ";
    AI->getAllocatedType()->print(errs());
    errs() << "\n";
    return std::nullopt;
  }

  errs() << "  kernel args array elements (from stores):\n";
  SmallVector<std::pair<unsigned, Value *>, 16> IndexToValue;

  unsigned TotalStores = 0;
  unsigned MatchingStores = 0;

  SmallPtrSet<const Value *, 16> CudaMallocOutParams;
  for (Instruction &I : instructions(*Caller)) {
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

  auto IsDerivedFromCudaMallocOut = [&](Value *V) -> const Value * {
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
  };

  auto RecordStore = [&](unsigned Index, Value *Stored) {
    IndexToValue.push_back({Index, Stored});
  };

  const uint64_t PtrSize = DL.getPointerTypeSize(KernelArgs->getType());

  for (Instruction &I : instructions(*Caller)) {
    auto *SI = dyn_cast<StoreInst>(&I);
    if (!SI)
      continue;

    ++TotalStores;

    int64_t ByteOffset = 0;
    const Value *BasePtr = GetPointerBaseWithConstantOffset(
        SI->getPointerOperand(), ByteOffset, DL);
    if (BasePtr != AI)
      continue;

    if (ByteOffset < 0)
      continue;

    if (PtrSize == 0 || (static_cast<uint64_t>(ByteOffset) % PtrSize) != 0)
      continue;

    unsigned Index = static_cast<unsigned>(
        static_cast<uint64_t>(ByteOffset) / PtrSize);
    RecordStore(Index, SI->getValueOperand());
    ++MatchingStores;
  }

  errs() << "  store scan: total=" << TotalStores
         << " matching=" << MatchingStores << "\n";

  for (const auto &IV : IndexToValue) {
    CudaKernelArgInfo Arg;
    Arg.Index = IV.first;
    if (IV.second->hasName())
      Arg.SlotName = IV.second->getName().str();
    {
      std::string TyStr;
      raw_string_ostream RSO(TyStr);
      IV.second->getType()->print(RSO);
      Arg.SlotTypeStr = RSO.str();
    }

    errs() << "    [" << IV.first << "] stored value: ";
    IV.second->print(errs());
    errs() << "\n";
    errs() << "         type: ";
    IV.second->getType()->print(errs());
    errs() << "\n";

    if (auto *SlotAlloca = dyn_cast<AllocaInst>(IV.second->stripPointerCasts())) {
      bool FoundStoreIntoSlot = false;
      bool AnyCudaMallocDerived = false;
      uint64_t BestKnownAlignment = 0;
      std::string BestCudaMallocOutName;
      for (Instruction &I : instructions(*Caller)) {
        auto *SI = dyn_cast<StoreInst>(&I);
        if (!SI)
          continue;
        if (SI->getPointerOperand()->stripPointerCasts() != SlotAlloca)
          continue;
        FoundStoreIntoSlot = true;

        Value *RHS = SI->getValueOperand();
        const Value *CudaMallocOut = IsDerivedFromCudaMallocOut(RHS);
        if (CudaMallocOut) {
          AnyCudaMallocDerived = true;
          errs() << "         rhs derives from cudaMalloc out param: ";
          CudaMallocOut->print(errs());
          errs() << "\n";

          Align KnownAlign = getKnownAlignment(RHS, DL, SI, &AC, &DT);
          errs() << "         rhs known alignment: " << KnownAlign.value()
                 << "\n";

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
        errs() << "         slot: not cudaMalloc-derived (based on stores in this function)\n";
      }

      Arg.IsCudaMallocDerived = AnyCudaMallocDerived;
      Arg.CudaMallocOutParamName = BestCudaMallocOutName;
      Arg.KnownAlignment = BestKnownAlignment;
    }

    Info.Args.push_back(std::move(Arg));
  }

  return Info;
}

PreservedAnalyses MyHostPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAMProxy = AM.getResult<FunctionAnalysisManagerModuleProxy>(M);
  FunctionAnalysisManager &FAM = FAMProxy.getManager();

  SmallVector<CallBase *, 16> CudaLaunchKernels;
  SmallVector<CudaLaunchKernelInfo, 16> LaunchInfos;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    auto &AC = FAM.getResult<AssumptionAnalysis>(F);
    auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
    for (auto &BB: F) {
      for (auto &I: BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (isCudaLaunchKernelCall(*CI)) {
            CudaLaunchKernels.push_back(CI);
            if (auto Info = dumpCudaLaunchKernelArgs(*CI, AC, DT))
              LaunchInfos.push_back(std::move(*Info));
          }
          continue;
        }

        if (auto *II = dyn_cast<InvokeInst>(&I)) {
          if (isCudaLaunchKernelCall(*II)) {
            CudaLaunchKernels.push_back(II);
            if (auto Info = dumpCudaLaunchKernelArgs(*II, AC, DT))
              LaunchInfos.push_back(std::move(*Info));
          }
          continue;
        }
      }
    }
  }

  json::Array Launches;
  for (const auto &LI : LaunchInfos) {
    json::Object Launch;
    Launch["host_function"] = LI.HostFunctionName;
    Launch["resolved_kernel_mangled"] = LI.ResolvedKernelMangled;
    if (!LI.GuessedOriginalKernelName.empty())
      Launch["guessed_original_kernel"] = LI.GuessedOriginalKernelName;
    if (!LI.KernelArgsAllocaName.empty())
      Launch["kernel_args_alloca"] = LI.KernelArgsAllocaName;

    json::Array Args;
    for (const auto &AI : LI.Args) {
      json::Object Arg;
      Arg["index"] = static_cast<int64_t>(AI.Index);
      if (!AI.SlotName.empty())
        Arg["slot"] = AI.SlotName;
      if (!AI.SlotTypeStr.empty())
        Arg["type"] = AI.SlotTypeStr;
      Arg["cudaMallocDerived"] = AI.IsCudaMallocDerived;
      if (!AI.CudaMallocOutParamName.empty())
        Arg["cudaMallocOutParam"] = AI.CudaMallocOutParamName;
      Arg["knownAlignment"] = static_cast<int64_t>(AI.KnownAlignment);
      Args.push_back(std::move(Arg));
    }
    Launch["args"] = std::move(Args);
    Launches.push_back(std::move(Launch));
  }

  json::Object Root;
  Root["launches"] = std::move(Launches);

  if (!MyHostPassOutput.empty()) {
    std::error_code EC;
    raw_fd_ostream OS(MyHostPassOutput, EC, sys::fs::OF_Text);
    if (EC) {
      errs() << "MyHostPass: failed to open output file '" << MyHostPassOutput
             << "': " << EC.message() << "\n";
    } else {
      OS << formatv("{0:2}\n", json::Value(std::move(Root)));
    }
  } else {
    errs() << formatv("{0:2}\n", json::Value(std::move(Root)));
  }
  return PreservedAnalyses::all();
}
