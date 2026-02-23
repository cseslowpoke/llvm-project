#include "llvm/Transforms/Utils/OffloadAssumeInjectionPass.h"

#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "offload-assume-injection"
#include "llvm/ADT/StringMap.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void validateOffloadAssumeInjectionInput(StringRef Path) {
  if (Path.empty())
    return;

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr = MemoryBuffer::getFile(Path);
  if (!BufOrErr) {
    LLVM_DEBUG(dbgs() << "OffloadAssumeInjectionPass: failed to read input '" << Path
                      << "': " << BufOrErr.getError().message() << "\n");
    return;
  }

  Expected<json::Value> Parsed = json::parse(BufOrErr.get()->getBuffer());
  if (!Parsed) {
    LLVM_DEBUG(dbgs() << "OffloadAssumeInjectionPass: invalid JSON in '" << Path << "': ";
               logAllUnhandledErrors(Parsed.takeError(), dbgs()); dbgs() << "\n");
    return;
  }

  json::Object *Root = Parsed->getAsObject();
  if (!Root) {
    LLVM_DEBUG(dbgs() << "OffloadAssumeInjectionPass: JSON root is not an object in '" << Path << "'\n");
    return;
  }

  json::Array *Launches = Root->getArray("launches");
  if (!Launches) {
    LLVM_DEBUG(dbgs() << "OffloadAssumeInjectionPass: JSON missing key 'launches' (or not an array) in '"
                      << Path << "'\n");
    return;
  }

  LLVM_DEBUG(dbgs() << "OffloadAssumeInjectionPass: received valid JSON (launches=" << Launches->size()
                    << ") from '" << Path << "'\n");
  LLVM_DEBUG(
    for (const json::Value &LV : *Launches) {
      const json::Object *LO = LV.getAsObject();
      if (!LO)
        continue;

      StringRef HostFn = LO->getString("host_function").value_or("<unknown>");
      StringRef Kernel =
          LO->getString("guessed_original_kernel").value_or("<unknown>");
      const json::Array *Args = LO->getArray("args");
      size_t ArgCount = Args ? Args->size() : 0;

      dbgs() << "  launch: host_function=" << HostFn << " kernel=" << Kernel
             << " args=" << ArgCount << "\n";
    }
  );
}

static cl::opt<std::string> OffloadAssumeInjectionInput(
    "offload-assume-injection-input", cl::Hidden, cl::init(""),
    cl::callback([](const std::string &V) { validateOffloadAssumeInjectionInput(V); }));

/// Extract the base function name from a demangled name.
/// e.g., "Fan1(float*, float*, int, int)" -> "Fan1"
///       "__device_stub__Fan1" -> "Fan1"
static StringRef extractKernelBaseName(StringRef Demangled) {
  // Remove __device_stub__ prefix if present
  if (Demangled.starts_with("__device_stub__"))
    Demangled = Demangled.drop_front(strlen("__device_stub__"));
  // Strip function signature (everything from '(' onwards)
  size_t ParenPos = Demangled.find('(');
  if (ParenPos != StringRef::npos)
    Demangled = Demangled.substr(0, ParenPos);
  return Demangled;
}

PreservedAnalyses OffloadAssumeInjectionPass::run(Module &M, ModuleAnalysisManager &AM) {
  (void)AM;

  if (OffloadAssumeInjectionInput.empty())
    return PreservedAnalyses::all();

  // Read JSON file
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(OffloadAssumeInjectionInput);
  if (!BufOrErr) {
    LLVM_DEBUG(dbgs() << "OffloadAssumeInjectionPass: failed to read '" << OffloadAssumeInjectionInput << "'\n");
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

  // Build map: kernel base name -> (arg index -> alignment)
  // Using StringMap with std::string keys from JSON
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

  // Find kernel functions and insert llvm.assume calls for alignment
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    // Get the demangled name to match against JSON kernel names
    std::string Demangled = demangle(F.getName().str());
    StringRef BaseName = extractKernelBaseName(Demangled);

    auto It = KernelArgAligns.find(BaseName);
    if (It == KernelArgAligns.end())
      continue;

    // Insert llvm.assume calls at the start of the function
    BasicBlock &EntryBB = F.getEntryBlock();
    IRBuilder<> Builder(&*EntryBB.getFirstInsertionPt());
    LLVMContext &Ctx = M.getContext();
    Function *AssumeFn = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::assume);

    for (const auto &[ArgIdx, Alignment] : It->second) {
      if (ArgIdx >= F.arg_size())
        continue;

      Argument *Arg = F.getArg(ArgIdx);
      if (!Arg->getType()->isPointerTy())
        continue;

      // Create operand bundle: [ "align"(ptr %p, i64 N) ]
      Value *AlignVal = ConstantInt::get(Type::getInt64Ty(Ctx), Alignment);
      SmallVector<Value *, 2> BundleArgs;
      BundleArgs.push_back(Arg);
      BundleArgs.push_back(AlignVal);
      OperandBundleDef AlignBundle("align", BundleArgs);

      Builder.CreateCall(AssumeFn, {Builder.getTrue()}, {AlignBundle});
      Changed = true;
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
