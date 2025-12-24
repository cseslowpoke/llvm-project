#include "llvm/Transforms/Utils/MyDevicePass.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> MyDevicePassInput(
    "my-device-pass-input", cl::Hidden, cl::init(""));

PreservedAnalyses MyDevicePass::run(Module &M, ModuleAnalysisManager &AM) {
  (void)M;
  (void)AM;

  if (MyDevicePassInput.empty())
    return PreservedAnalyses::all();

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(MyDevicePassInput);
  if (!BufOrErr) {
    errs() << "MyDevicePass: failed to read input '" << MyDevicePassInput
           << "': " << BufOrErr.getError().message() << "\n";
    return PreservedAnalyses::all();
  }

  return PreservedAnalyses::all();
}
