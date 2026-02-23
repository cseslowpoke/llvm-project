//===- OffloadHostAnalysisPass.cpp - Host-side kernel launch analysis ----===//
//
// This pass analyzes cudaLaunchKernel calls on the host side to extract
// information about kernel arguments, particularly their alignment properties.
// The analysis results are output as JSON for consumption by device-side passes
// (OffloadAssumeInjectionPass, OffloadParamAttributePass).
//
// Architecture:
// - This pass runs on HOST code (not device code)
// - It finds cudaLaunchKernel calls and analyzes the kernel args array
// - For each argument, it checks if it derives from cudaMalloc
// - Alignment analysis is delegated to OffloadAlignmentAnalyzer (internal module)
// - Results are written to JSON file specified by -offload-host-analysis-output
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/OffloadHostAnalysisPass.h"
#include "OffloadAliasAnalysis.h"     // Internal alias analysis module
#include "OffloadAlignmentAnalysis.h" // Internal alignment analysis module

#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "offload-host-analysis"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"

#include <optional>
#include <string>

using namespace llvm;

/// Command-line option to specify the output JSON file path.
/// This file will contain kernel launch information for device-side passes.
/// Usage: -mllvm -offload-host-analysis-output=/path/to/output.json
static cl::opt<std::string> OffloadHostAnalysisOutput(
    "offload-host-analysis-output", cl::Hidden, cl::init(""));

/// Check if a call instruction is a cudaLaunchKernel call.
/// cudaLaunchKernel is the runtime API used to launch CUDA kernels.
static bool isCudaLaunchKernelCall(CallBase &CB) {
  Function *Callee = CB.getCalledFunction();
  if (!Callee)
    return false;
  StringRef Name = Callee->getName();
  if (Name != "cudaLaunchKernel")
    return false;
  return true;
}

/// Resolve the kernel function from the first argument of cudaLaunchKernel.
///
/// cudaLaunchKernel signature:
///   cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
///                                 void **args, size_t sharedMem, cudaStream_t stream)
///
/// The first argument (func) is a pointer to the kernel function, but it may
/// be wrapped in casts, aliases, or constant expressions. This function
/// traverses through these to find the actual Function.
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

    // Found the kernel function directly
    if (auto *F = dyn_cast<Function>(Cur))
      return F;

    // Handle global aliases (kernel may be aliased)
    if (auto *GA = dyn_cast<GlobalAlias>(Cur)) {
      if (const GlobalObject *GO = GA->getAliaseeObject())
        if (const auto *F = dyn_cast<Function>(GO))
          return F;
      continue;
    }

    // Handle constant expressions (e.g., bitcast)
    if (auto *CE = dyn_cast<ConstantExpr>(Cur)) {
      if (CE->getNumOperands() > 0)
        Worklist.push_back(CE->getOperand(0));
      continue;
    }
  }

  return nullptr;
}

/// Information about a single kernel argument extracted from cudaLaunchKernel.
struct CudaKernelArgInfo {
  unsigned Index = 0;                    ///< Argument index (0-based)
  std::string SlotName;                  ///< Name of the slot alloca (if any)
  std::string SlotTypeStr;               ///< LLVM type string of the slot value
  bool IsCudaMallocDerived = false;      ///< True if derived from cudaMalloc
  std::string CudaMallocOutParamName;    ///< Name of cudaMalloc output param
  uint64_t KnownAlignment = 0;           ///< Best known alignment in bytes
  unsigned AllocationID = 0;             ///< Unique ID for noalias analysis
};

/// Information about a cudaLaunchKernel call site.
struct CudaLaunchKernelInfo {
  std::string HostFunctionName;          ///< Function containing the launch
  std::string ResolvedKernelMangled;     ///< Mangled name of the kernel
  std::string GuessedOriginalKernelName; ///< Demangled kernel name (guessed)
  std::string KernelArgsAllocaName;      ///< Name of the args array alloca
  SmallVector<CudaKernelArgInfo, 16> Args; ///< Per-argument information
};

/// Analyze a cudaLaunchKernel call and extract argument information.
///
/// This function:
/// 1. Resolves the kernel function from argument 0
/// 2. Extracts the kernel args array from argument 5 (index 5 = args parameter)
/// 3. Scans stores into the args array to find individual argument values
/// 4. For each argument, analyzes alignment using OffloadAlignmentAnalyzer
///
/// @param CB  The cudaLaunchKernel call instruction
/// @param AC  AssumptionCache for alignment analysis
/// @param DT  DominatorTree for alignment analysis
/// @return    Optional launch info, or nullopt if analysis fails
static std::optional<CudaLaunchKernelInfo>
dumpCudaLaunchKernelArgs(CallBase &CB, AssumptionCache &AC, DominatorTree &DT) {
  // cudaLaunchKernel args: func, gridDim, blockDim, args, sharedMem, stream
  // Index 5 is the 'args' parameter (void** pointing to argument array)
  constexpr unsigned KernelArgsIndex = 5;
  Function *Caller = CB.getFunction();
  const DataLayout &DL = Caller->getParent()->getDataLayout();
    
  LLVM_DEBUG(dbgs() << "cudaLaunchKernel in function: " << Caller->getName() << "\n");

  if (CB.arg_size() <= KernelArgsIndex) {
    LLVM_DEBUG(dbgs() << "  unexpected arg_size=" << CB.arg_size() << " (need >= 6)\n");
    return std::nullopt;
  }

  Value *KernelFn = CB.getArgOperand(0);
  CudaLaunchKernelInfo Info;
  Info.HostFunctionName = Caller->getName().str();
  const Function *KF = resolveKernelFunction(KernelFn);
  if (!KF) {
    LLVM_DEBUG(dbgs() << "    resolved kernel: <unknown>\n");
    return std::nullopt;
  }

  LLVM_DEBUG(dbgs() << "    resolved kernel: " << KF->getName() << "\n");
  if (KF->getName() == Caller->getName()) {
    LLVM_DEBUG(dbgs() << "    caller is stub kernel function\n");
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
      LLVM_DEBUG(dbgs() << "    guessed original kernel (demangled): " << GuessedKernel
                        << "\n");
      Info.GuessedOriginalKernelName = GuessedKernel.str();
    }
  }

  Value *KernelArgs = CB.getArgOperand(KernelArgsIndex);
  LLVM_DEBUG(dbgs() << "  kernel args operand (idx 5):\n");
  LLVM_DEBUG(dbgs() << "    name: "
                    << (KernelArgs->hasName() ? KernelArgs->getName() : "<unnamed>")
                    << "\n");
  LLVM_DEBUG(dbgs() << "    type: "; KernelArgs->getType()->print(dbgs()); dbgs() << "\n");
  Value *Base = KernelArgs->stripPointerCasts();
  auto *AI = dyn_cast<AllocaInst>(Base);
  if (!AI) {
    LLVM_DEBUG(dbgs() << "  kernel args base is not an alloca after stripping casts\n");
    return std::nullopt;
  }

  if (AI->hasName())
    Info.KernelArgsAllocaName = AI->getName().str();

  auto *AT = dyn_cast<ArrayType>(AI->getAllocatedType());
  if (!AT) {
    LLVM_DEBUG(dbgs() << "  kernel args alloca is not an array: ";
               AI->getAllocatedType()->print(dbgs()); dbgs() << "\n");
    return std::nullopt;
  }

  LLVM_DEBUG(dbgs() << "  kernel args array elements (from stores):\n");
  SmallVector<std::pair<unsigned, Value *>, 16> IndexToValue;

  unsigned TotalStores = 0;
  unsigned MatchingStores = 0;

  // Create alignment analyzer for this function.
  // This is an internal module that handles cudaMalloc tracking and
  // alignment computation. See OffloadAlignmentAnalysis.h for details.
  OffloadAlignmentAnalyzer AlignAnalyzer(*Caller, AC, DT);

  // Create alias analyzer for this function.
  // This tracks which cudaMalloc allocation each argument comes from.
  // Arguments from different allocations are guaranteed to not alias.
  OffloadAliasAnalyzer AliasAnalyzer(*Caller);

  auto RecordStore = [&](unsigned Index, Value *Stored) {
    IndexToValue.push_back({Index, Stored});
  };

  const uint64_t PtrSize = DL.getPointerTypeSize(KernelArgs->getType());

  // Scan all stores in the function to find stores into the kernel args array.
  // The args array is set up like:
  //   %args = alloca [N x ptr]
  //   store ptr %arg0_slot, ptr getelementptr([N x ptr], ptr %args, 0, 0)
  //   store ptr %arg1_slot, ptr getelementptr([N x ptr], ptr %args, 0, 1)
  //   ...
  // We use GetPointerBaseWithConstantOffset to match stores to array elements.
  for (Instruction &I : instructions(*Caller)) {
    auto *SI = dyn_cast<StoreInst>(&I);
    if (!SI)
      continue;

    ++TotalStores;

    // Check if this store targets the kernel args array
    int64_t ByteOffset = 0;
    const Value *BasePtr = GetPointerBaseWithConstantOffset(
        SI->getPointerOperand(), ByteOffset, DL);
    if (BasePtr != AI)
      continue;

    if (ByteOffset < 0)
      continue;

    // Convert byte offset to array index
    if (PtrSize == 0 || (static_cast<uint64_t>(ByteOffset) % PtrSize) != 0)
      continue;

    unsigned Index = static_cast<unsigned>(
        static_cast<uint64_t>(ByteOffset) / PtrSize);
    RecordStore(Index, SI->getValueOperand());
    ++MatchingStores;
  }

  LLVM_DEBUG(dbgs() << "  store scan: total=" << TotalStores
                    << " matching=" << MatchingStores << "\n");

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

    LLVM_DEBUG(dbgs() << "    [" << IV.first << "] stored value: ";
               IV.second->print(dbgs()); dbgs() << "\n");
    LLVM_DEBUG(dbgs() << "         type: ";
               IV.second->getType()->print(dbgs()); dbgs() << "\n");

    // Delegate alignment analysis to the internal OffloadAlignmentAnalyzer.
    // This checks if the argument derives from cudaMalloc and computes
    // the known alignment using LLVM's getKnownAlignment.
    KernelArgAlignmentInfo AlignInfo = AlignAnalyzer.analyzeArgumentAlignment(IV.second);
    Arg.IsCudaMallocDerived = AlignInfo.IsCudaMallocDerived;
    Arg.CudaMallocOutParamName = AlignInfo.CudaMallocOutParamName;
    Arg.KnownAlignment = AlignInfo.KnownAlignment;

    // Delegate alias analysis to the internal OffloadAliasAnalyzer.
    // This assigns a unique allocation ID to arguments from cudaMalloc.
    // Arguments with different IDs are guaranteed to not alias.
    KernelArgAliasInfo AliasInfo =
        AliasAnalyzer.analyzeArgumentAlias(IV.second);
    Arg.AllocationID = AliasInfo.AllocationID;

    Info.Args.push_back(std::move(Arg));
  }

  return Info;
}

/// Main entry point for the OffloadHostAnalysisPass.
///
/// This pass scans the module for cudaLaunchKernel calls, analyzes each one
/// to extract kernel argument information (especially alignment), and outputs
/// the results as JSON for consumption by device-side passes.
///
/// The JSON output format:
/// {
///   "launches": [
///     {
///       "host_function": "main",
///       "resolved_kernel_mangled": "_Z6kernelPfS_i",
///       "guessed_original_kernel": "kernel",
///       "args": [
///         { "index": 0, "cudaMallocDerived": true, "knownAlignment": 256 },
///         ...
///       ]
///     }
///   ]
/// }
PreservedAnalyses OffloadHostAnalysisPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAMProxy = AM.getResult<FunctionAnalysisManagerModuleProxy>(M);
  FunctionAnalysisManager &FAM = FAMProxy.getManager();

  SmallVector<CallBase *, 16> CudaLaunchKernels;
  SmallVector<CudaLaunchKernelInfo, 16> LaunchInfos;

  // Scan all functions in the module for cudaLaunchKernel calls.
  // We handle both CallInst and InvokeInst (for exception-safe code).
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    // Get analysis results needed for alignment computation
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

  // Build JSON output from collected launch information.
  // This JSON will be consumed by device-side passes to apply optimizations.
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
      if (AI.AllocationID > 0)
        Arg["allocationID"] = static_cast<int64_t>(AI.AllocationID);
      Args.push_back(std::move(Arg));
    }
    Launch["args"] = std::move(Args);
    Launches.push_back(std::move(Launch));
  }

  json::Object Root;
  Root["launches"] = std::move(Launches);

  // Write JSON output to file or stderr.
  // The file path is specified via -offload-host-analysis-output option.
  if (!OffloadHostAnalysisOutput.empty()) {
    std::error_code EC;
    raw_fd_ostream OS(OffloadHostAnalysisOutput, EC, sys::fs::OF_Text);
    if (EC) {
      errs() << "OffloadHostAnalysisPass: failed to open output file '" << OffloadHostAnalysisOutput
             << "': " << EC.message() << "\n";
    } else {
      OS << formatv("{0:2}\n", json::Value(std::move(Root)));
    }
  } else {
    // If no output file specified, print to debug stream
    LLVM_DEBUG(dbgs() << formatv("{0:2}\n", json::Value(std::move(Root))));
  }

  // This is an analysis pass; it doesn't modify the IR.
  return PreservedAnalyses::all();
}
