//===- ReplaceCudaFatbinPass.cpp - Replace dummy fatbin in host IR --------===//
//
// This pass replaces the dummy fatbin data in host LLVM IR with the real
// fatbin file contents. It finds the global variable in the .nv_fatbin
// section (created by Clang's CodeGen with a dummy fatbin) and replaces
// its initializer with the actual fatbin data.
//
// This enables a two-phase host compilation:
//   Phase 1: .cu -> cc1 (with dummy fatbin) -> .bc (has full registration code)
//   Phase 2: .bc -> replace-cuda-fatbin -> .bc' -> llc -> .o
// Avoiding the need to re-parse the .cu source in Phase 2.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ReplaceCudaFatbinPass.h"

#define DEBUG_TYPE "replace-cuda-fatbin"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<std::string> ReplaceCudaFatbinFile(
    "replace-cuda-fatbin-file", cl::Hidden, cl::init(""),
    cl::desc("Path to the real fatbin file to replace the dummy fatbin data"));

PreservedAnalyses ReplaceCudaFatbinPass::run(Module &M,
                                             ModuleAnalysisManager &AM) {
  if (ReplaceCudaFatbinFile.empty())
    return PreservedAnalyses::all();

  // Read the real fatbin file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(ReplaceCudaFatbinFile);
  if (!BufOrErr) {
    errs() << "ReplaceCudaFatbinPass: failed to read fatbin file '"
           << ReplaceCudaFatbinFile
           << "': " << BufOrErr.getError().message() << "\n";
    return PreservedAnalyses::all();
  }
  StringRef FatbinData = BufOrErr.get()->getBuffer();
  LLVM_DEBUG(dbgs() << "ReplaceCudaFatbinPass: read fatbin file '"
                    << ReplaceCudaFatbinFile << "' (" << FatbinData.size()
                    << " bytes)\n");

  // Find the fatbin global variable in the .nv_fatbin section.
  // Clang's CGCUDANV.cpp creates this via makeConstantArray() with section
  // ".nv_fatbin" (Linux) or "__NV_CUDA,__nv_fatbin" (macOS).
  GlobalVariable *FatbinGV = nullptr;
  for (GlobalVariable &GV : M.globals()) {
    StringRef Section = GV.getSection();
    if (Section == ".nv_fatbin" || Section == "__NV_CUDA,__nv_fatbin" ||
        Section == "__nv_relfatbin" || Section == "__NV_CUDA,__nv_relfatbin") {
      FatbinGV = &GV;
      break;
    }
  }

  if (!FatbinGV) {
    LLVM_DEBUG(
        dbgs()
        << "ReplaceCudaFatbinPass: no .nv_fatbin global found in module\n");
    return PreservedAnalyses::all();
  }

  LLVM_DEBUG(dbgs() << "ReplaceCudaFatbinPass: found fatbin global '"
                    << FatbinGV->getName() << "' in section '"
                    << FatbinGV->getSection() << "'\n");

  // Create a new constant with the real fatbin data.
  // makeConstantArray in CGCUDANV.cpp uses a null-terminated string constant,
  // so we do the same here.
  Constant *NewInit =
      ConstantDataArray::getString(M.getContext(), FatbinData,
                                   /*AddNull=*/true);

  // Create a new global variable with the correct type (size may differ).
  auto *NewGV = new GlobalVariable(
      M, NewInit->getType(), FatbinGV->isConstant(), FatbinGV->getLinkage(),
      NewInit, "", FatbinGV, FatbinGV->getThreadLocalMode(),
      FatbinGV->getAddressSpace());
  NewGV->setAlignment(FatbinGV->getAlign());
  NewGV->setSection(FatbinGV->getSection());
  NewGV->setUnnamedAddr(FatbinGV->getUnnamedAddr());
  NewGV->copyMetadata(FatbinGV, /*Offset=*/0);

  // Transfer the name and replace all uses.
  // With opaque pointers, all users reference this as ptr, so RAUW works
  // directly regardless of the type/size change.
  NewGV->takeName(FatbinGV);
  FatbinGV->replaceAllUsesWith(NewGV);
  FatbinGV->eraseFromParent();

  LLVM_DEBUG(dbgs() << "ReplaceCudaFatbinPass: replaced fatbin data ("
                    << FatbinData.size() << " bytes)\n");

  return PreservedAnalyses::none();
}
