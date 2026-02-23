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

  // Build map: kernel base name -> (arg index -> alignment)
  StringMap<SmallVector<std::pair<unsigned, unsigned>, 8>> KernelArgAligns;

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

    auto &ArgAligns = KernelArgAligns[*KernelName];
    for (const json::Value &AV : *Args) {
      const json::Object *AO = AV.getAsObject();
      if (!AO)
        continue;

      std::optional<int64_t> Idx = AO->getInteger("index");
      std::optional<int64_t> Align = AO->getInteger("knownAlignment");
      std::optional<bool> IsCudaMalloc = AO->getBoolean("cudaMallocDerived");

      // Only add alignment for cudaMalloc-derived pointers with known alignment > 1
      if (Idx && Align && IsCudaMalloc && *IsCudaMalloc && *Align > 1) {
        ArgAligns.push_back({static_cast<unsigned>(*Idx),
                             static_cast<unsigned>(*Align)});
      }
    }
  }

  if (KernelArgAligns.empty())
    return PreservedAnalyses::all();

  bool Changed = false;
  LLVMContext &Ctx = M.getContext();

  // Find kernel functions and add alignment attributes
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    std::string Demangled = demangle(F.getName().str());
    StringRef BaseName = extractKernelBaseName(Demangled);

    auto It = KernelArgAligns.find(BaseName);
    if (It == KernelArgAligns.end())
      continue;

    for (const auto &[ArgIdx, Alignment] : It->second) {
      if (ArgIdx >= F.arg_size())
        continue;

      Argument *Arg = F.getArg(ArgIdx);
      if (!Arg->getType()->isPointerTy())
        continue;

      // Add alignment attribute directly to the parameter
      F.addParamAttr(ArgIdx, Attribute::getWithAlignment(Ctx, Align(Alignment)));
      Changed = true;

      LLVM_DEBUG(dbgs() << "OffloadParamAttributePass: added align(" << Alignment << ") to arg "
                        << ArgIdx << " of " << F.getName() << "\n");
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
