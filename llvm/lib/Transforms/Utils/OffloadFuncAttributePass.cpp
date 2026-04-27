#include "llvm/Transforms/Utils/OffloadFuncAttributePass.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"

#define DEBUG_TYPE "offload-func-attribute"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> OffloadFuncAttributeInput(
    "offload-func-attribute-input", cl::Hidden, cl::init(""),
    cl::desc("JSON file with kernel launch info for OffloadFuncAttributePass"));

static cl::opt<bool> OffloadFuncAttributeMaxNTID(
    "offload-func-attribute-maxntid", cl::Hidden, cl::init(true),
    cl::desc("Add nvvm.maxntid function attribute (default: true)"));

/// Extract the base function name from a demangled name.
static StringRef extractKernelBaseName(StringRef Demangled) {
  if (Demangled.starts_with("__device_stub__"))
    Demangled = Demangled.drop_front(strlen("__device_stub__"));
  size_t ParenPos = Demangled.find('(');
  if (ParenPos != StringRef::npos)
    Demangled = Demangled.substr(0, ParenPos);
  return Demangled;
}

/// Read an unsigned dim component from a JSON object field. Returns nullopt if
/// the field is missing or null (encoded by OffloadHostAnalysisPass for
/// statically-unknown components).
static std::optional<uint64_t> readDimComponent(const json::Object &O,
                                                StringRef Key) {
  const json::Value *V = O.get(Key);
  if (!V || V->kind() == json::Value::Null)
    return std::nullopt;
  if (auto I = V->getAsInteger())
    return static_cast<uint64_t>(*I);
  return std::nullopt;
}

PreservedAnalyses OffloadFuncAttributePass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  (void)AM;

  if (OffloadFuncAttributeInput.empty())
    return PreservedAnalyses::all();

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(OffloadFuncAttributeInput);
  if (!BufOrErr) {
    LLVM_DEBUG(dbgs() << "OffloadFuncAttributePass: failed to read '"
                      << OffloadFuncAttributeInput << "'\n");
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

  LLVM_DEBUG(dbgs() << "OffloadFuncAttributePass: received valid JSON (launches="
                    << Launches->size() << ") from '"
                    << OffloadFuncAttributeInput << "'\n");

  // Build map: kernel base name -> max known ntid product across launches.
  // Only count launches whose block_dim is fully known (x, y, z all present).
  StringMap<uint64_t> KernelMaxNTID;

  for (const json::Value &LV : *Launches) {
    const json::Object *LO = LV.getAsObject();
    if (!LO)
      continue;

    std::optional<StringRef> KernelName =
        LO->getString("guessed_original_kernel");
    if (!KernelName || KernelName->empty())
      continue;

    const json::Object *BlockDim = LO->getObject("block_dim");
    if (!BlockDim)
      continue;

    auto X = readDimComponent(*BlockDim, "x");
    auto Y = readDimComponent(*BlockDim, "y");
    auto Z = readDimComponent(*BlockDim, "z");
    if (!X || !Y || !Z)
      continue;

    uint64_t Product = (*X) * (*Y) * (*Z);
    if (Product == 0)
      continue;

    auto &Cur = KernelMaxNTID[*KernelName];
    Cur = std::max(Cur, Product);
  }

  if (KernelMaxNTID.empty())
    return PreservedAnalyses::all();

  bool Changed = false;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    // Only apply to NVPTX kernels.
    if (F.getCallingConv() != CallingConv::PTX_Kernel)
      continue;

    std::string Demangled = demangle(F.getName().str());
    StringRef BaseName = extractKernelBaseName(Demangled);

    auto It = KernelMaxNTID.find(BaseName);
    if (It == KernelMaxNTID.end())
      continue;

    if (OffloadFuncAttributeMaxNTID && !F.hasFnAttribute("nvvm.maxntid")) {
      F.addFnAttr("nvvm.maxntid", utostr(It->second));
      Changed = true;
      LLVM_DEBUG(dbgs() << "OffloadFuncAttributePass: added nvvm.maxntid="
                        << It->second << " to " << F.getName() << "\n");
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
