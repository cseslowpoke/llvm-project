#include "llvm/Transforms/Utils/CudaAssumeAlignPass.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<unsigned> CudaAssumeAlign(
    "cuda-assume-align", cl::Hidden, cl::init(256));

static cl::opt<bool> CudaAssumeAlignVerbose(
    "cuda-assume-align-verbose", cl::Hidden, cl::init(false));

PreservedAnalyses CudaAssumeAlignPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  (void)AM;

  if (F.isDeclaration())
    return PreservedAnalyses::all();

  Module *M = F.getParent();
  Function *AssumeFn = Intrinsic::getOrInsertDeclaration(M, Intrinsic::assume);

  bool Changed = false;

  SmallPtrSet<const Value *, 16> CudaMallocOutParams;
  for (Instruction &I : instructions(F)) {
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB)
      continue;
    Function *Callee = CB->getCalledFunction();
    if (!Callee)
      continue;
    if (Callee->getName() != "cudaMalloc")
      continue;
    if (CB->arg_size() < 1)
      continue;
    // Because CudaMalloc(&ptr, size), so the device pointer is the first arg
    const Value *OutParam = CB->getArgOperand(0)->stripPointerCasts();
    CudaMallocOutParams.insert(OutParam);
  }

  if (CudaMallocOutParams.empty())
    return PreservedAnalyses::all();

  for (Instruction &I : instructions(F)) {
    auto *LI = dyn_cast<LoadInst>(&I);
    if (!LI)
      continue;

    const Value *PtrOp = LI->getPointerOperand()->stripPointerCasts();
    if (!CudaMallocOutParams.contains(PtrOp))
      continue;

    Value *LoadedPtr = LI;

    IRBuilder<> B(LI->getNextNode());
    LLVMContext &Ctx = M->getContext();

    // Create operand bundle: [ "align"(ptr %p, i64 N) ]
    Value *AlignVal = ConstantInt::get(Type::getInt64Ty(Ctx), CudaAssumeAlign);
    SmallVector<Value *, 2> BundleArgs;
    BundleArgs.push_back(LoadedPtr);
    BundleArgs.push_back(AlignVal);
    OperandBundleDef AlignBundle("align", BundleArgs);

    CallInst *NewAssume = B.CreateCall(AssumeFn, {B.getTrue()}, {AlignBundle});
    (void)NewAssume;

    Changed = true;
    if (CudaAssumeAlignVerbose) {
      errs() << "CudaAssumeAlignPass: inserted assume align=" << CudaAssumeAlign
             << " for loaded ptr: ";
      LoadedPtr->print(errs());
      errs() << "\n";
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
