#include "llvm/Transforms/Utils/MyHostPass.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> MyHostPassOutput(
    "my-host-pass-output", cl::Hidden, cl::init(""));

PreservedAnalyses MyHostPass::run(Module &M, ModuleAnalysisManager &AM) {
  (void)AM;

  if (MyHostPassOutput.empty())
    return PreservedAnalyses::all();

  std::error_code EC;
  raw_fd_ostream OS(MyHostPassOutput, EC, sys::fs::OF_Text);
  if (EC) {
    errs() << "MyHostPass: failed to open output '" << MyHostPassOutput
           << "': " << EC.message() << "\n";
    return PreservedAnalyses::all();
  }

  OS << "{\n";
  OS << "  \"module\": \"" << M.getName() << "\"\n";
  OS << "}\n";

  return PreservedAnalyses::all();
}
