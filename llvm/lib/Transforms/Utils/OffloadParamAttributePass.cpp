#include "llvm/Transforms/Utils/OffloadParamAttributePass.h"

#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "offload-param-attribute"
#include "llvm/ADT/StringMap.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> OffloadParamAttributeInput(
    "offload-param-attribute-input", cl::Hidden, cl::init(""),
    cl::desc("JSON file with kernel launch info for OffloadParamAttributePass"));

static cl::opt<bool> OffloadParamAttributeAlign(
    "offload-param-attribute-align", cl::Hidden, cl::init(true),
    cl::desc("Add alignment attributes to kernel parameters (default: true)"));

static cl::opt<bool> OffloadParamAttributeNoAlias(
    "offload-param-attribute-noalias", cl::Hidden, cl::init(true),
    cl::desc("Add noalias attributes to kernel parameters (default: true)"));

/// Extract the base function name from a demangled name.
static StringRef extractKernelBaseName(StringRef Demangled) {
  if (Demangled.starts_with("__device_stub__"))
    Demangled = Demangled.drop_front(strlen("__device_stub__"));
  size_t ParenPos = Demangled.find('(');
  if (ParenPos != StringRef::npos)
    Demangled = Demangled.substr(0, ParenPos);
  return Demangled;
}

PreservedAnalyses OffloadParamAttributePass::run(Module &M, ModuleAnalysisManager &AM) {
  (void)AM;

  if (OffloadParamAttributeInput.empty())
    return PreservedAnalyses::all();

  // Read JSON file
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(OffloadParamAttributeInput);
  if (!BufOrErr) {
    LLVM_DEBUG(dbgs() << "OffloadParamAttributePass: failed to read '" << OffloadParamAttributeInput << "'\n");
    return PreservedAnalyses::all();
  }

  Expected<json::Value> Parsed = json::parse(BufOrErr.get()->getBuffer());
  if (!Parsed) {
    consumeError(Parsed.takeError());
    return PreservedAnalyses::all();
  }

  json::Object *Root = Parsed->getAsObject();
  if (!Root)
    return PreservedAnalyses::all();

  json::Array *Launches = Root->getArray("launches");
  if (!Launches)
    return PreservedAnalyses::all();

  LLVM_DEBUG(dbgs() << "OffloadParamAttributePass: received valid JSON (launches=" << Launches->size()
                    << ") from '" << OffloadParamAttributeInput << "'\n");

  // Build map: kernel base name -> (arg index -> (alignment, allocationID))
  struct ArgInfo {
    unsigned Alignment = 0;
    unsigned AllocationID = 0;
  };
  StringMap<SmallVector<std::pair<unsigned, ArgInfo>, 8>> KernelArgInfos;

  for (const json::Value &LV : *Launches) {
    const json::Object *LO = LV.getAsObject();
    if (!LO)
      continue;

    std::optional<StringRef> KernelName = LO->getString("guessed_original_kernel");
    if (!KernelName || KernelName->empty())
      continue;

    const json::Array *Args = LO->getArray("args");
    if (!Args)
      continue;

    auto &ArgInfoVec = KernelArgInfos[*KernelName];
    for (const json::Value &AV : *Args) {
      const json::Object *AO = AV.getAsObject();
      if (!AO)
        continue;

      std::optional<int64_t> Idx = AO->getInteger("index");
      std::optional<int64_t> Align = AO->getInteger("knownAlignment");
      std::optional<bool> IsCudaMalloc = AO->getBoolean("cudaMallocDerived");
      std::optional<int64_t> AllocID = AO->getInteger("allocationID");

      // Only process cudaMalloc-derived pointers
      if (Idx && IsCudaMalloc && *IsCudaMalloc) {
        ArgInfo Info;
        if (Align && *Align > 1)
          Info.Alignment = static_cast<unsigned>(*Align);
        if (AllocID && *AllocID > 0)
          Info.AllocationID = static_cast<unsigned>(*AllocID);
        ArgInfoVec.push_back({static_cast<unsigned>(*Idx), Info});
      }
    }
  }

  if (KernelArgInfos.empty())
    return PreservedAnalyses::all();

  bool Changed = false;
  LLVMContext &Ctx = M.getContext();

  // Find kernel functions and add alignment/noalias attributes
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    std::string Demangled = demangle(F.getName().str());
    StringRef BaseName = extractKernelBaseName(Demangled);

    auto It = KernelArgInfos.find(BaseName);
    if (It == KernelArgInfos.end())
      continue;

    // Collect allocation IDs to determine noalias relationships
    // Arguments with different non-zero allocation IDs cannot alias
    SmallVector<std::pair<unsigned, unsigned>, 8>
        ArgAllocIDs; // (argIdx, allocID)
    for (const auto &[ArgIdx, Info] : It->second) {
      if (Info.AllocationID > 0)
        ArgAllocIDs.push_back({ArgIdx, Info.AllocationID});
    }

    for (const auto &[ArgIdx, Info] : It->second) {
      if (ArgIdx >= F.arg_size())
        continue;

      Argument *Arg = F.getArg(ArgIdx);
      if (!Arg->getType()->isPointerTy())
        continue;

      // Add alignment attribute if available and enabled
      if (OffloadParamAttributeAlign && Info.Alignment > 1) {
        F.addParamAttr(ArgIdx,
                       Attribute::getWithAlignment(Ctx, Align(Info.Alignment)));
        Changed = true;
        LLVM_DEBUG(dbgs() << "OffloadParamAttributePass: added align("
                          << Info.Alignment << ") to arg " << ArgIdx << " of "
                          << F.getName() << "\n");
      }

      // Add noalias attribute if enabled and this argument has a unique
      // allocation ID that differs from all other arguments' allocation IDs
      if (OffloadParamAttributeNoAlias && Info.AllocationID > 0) {
        bool CanBeNoAlias = true;
        for (const auto &[OtherIdx, OtherAllocID] : ArgAllocIDs) {
          if (OtherIdx != ArgIdx && OtherAllocID == Info.AllocationID) {
            // Same allocation ID means they might alias
            CanBeNoAlias = false;
            break;
          }
        }
        if (CanBeNoAlias) {
          F.addParamAttr(ArgIdx, Attribute::NoAlias);
          Changed = true;
          LLVM_DEBUG(dbgs()
                     << "OffloadParamAttributePass: added noalias to arg "
                     << ArgIdx << " of " << F.getName()
                     << " (allocationID=" << Info.AllocationID << ")\n");
        }
      }
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
