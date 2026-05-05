#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {

class ForLoopFusionPass
    : public PassWrapper<ForLoopFusionPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "dolov-loop-fusion"; }
  StringRef getDescription() const final {
    return "Fuses adjacent scf.for loops with identical bounds and no "
           "dependencies";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool changed;

    do {
      changed = false;
      module.walk([&](Block *block) {
        for (auto it = block->begin(); it != block->end(); ++it) {
          auto loop1 = dyn_cast<scf::ForOp>(&*it);
          if (!loop1)
            continue;

          auto nextIt = std::next(it);
          if (nextIt == block->end())
            continue;

          auto loop2 = dyn_cast<scf::ForOp>(&*nextIt);
          if (!loop2)
            continue;

          if (canFuse(loop1, loop2)) {
            fuse(loop1, loop2);
            changed = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
    } while (changed);
  }

private:
  bool canFuse(scf::ForOp loop1, scf::ForOp loop2) {
    if (loop1.getLowerBound() != loop2.getLowerBound() ||
        loop1.getUpperBound() != loop2.getUpperBound() ||
        loop1.getStep() != loop2.getStep()) {
      return false;
    }

    for (Value result : loop1.getResults()) {
      for (Operation *user : result.getUsers()) {
        if (loop2->isAncestor(user)) {
          return false;
        }
      }
    }
    return true;
  }

  void fuse(scf::ForOp loop1, scf::ForOp loop2) {
    OpBuilder builder(loop1);

    SmallVector<Value> combinedInitArgs;
    combinedInitArgs.append(loop1.getInitArgs().begin(),
                            loop1.getInitArgs().end());
    combinedInitArgs.append(loop2.getInitArgs().begin(),
                            loop2.getInitArgs().end());

    auto fusedLoop = builder.create<scf::ForOp>(
        loop1.getLoc(), loop1.getLowerBound(), loop1.getUpperBound(),
        loop1.getStep(), combinedInitArgs);

    fusedLoop.getBody()->clear();

    IRMapping mapper;
    mapper.map(loop1.getInductionVar(), fusedLoop.getInductionVar());
    mapper.map(loop2.getInductionVar(), fusedLoop.getInductionVar());

    unsigned n1 = loop1.getNumRegionIterArgs();
    for (unsigned i = 0; i < n1; ++i)
      mapper.map(loop1.getRegionIterArgs()[i],
                 fusedLoop.getRegionIterArgs()[i]);

    for (unsigned i = 0; i < loop2.getNumRegionIterArgs(); ++i)
      mapper.map(loop2.getRegionIterArgs()[i],
                 fusedLoop.getRegionIterArgs()[n1 + i]);

    builder.setInsertionPointToStart(fusedLoop.getBody());

    for (auto &op : loop1.getBody()->without_terminator()) {
      builder.clone(op, mapper);
    }

    for (auto &op : loop2.getBody()->without_terminator()) {
      builder.clone(op, mapper);
    }

    SmallVector<Value> combinedYieldValues;

    auto yield1 = cast<scf::YieldOp>(loop1.getBody()->getTerminator());
    for (Value v : yield1.getOperands())
      combinedYieldValues.push_back(mapper.lookupOrDefault(v));

    auto yield2 = cast<scf::YieldOp>(loop2.getBody()->getTerminator());
    for (Value v : yield2.getOperands())
      combinedYieldValues.push_back(mapper.lookupOrDefault(v));

    builder.create<scf::YieldOp>(loop1.getLoc(), combinedYieldValues);

    for (unsigned i = 0; i < loop1.getNumResults(); ++i)
      loop1.getResult(i).replaceAllUsesWith(fusedLoop.getResult(i));

    for (unsigned i = 0; i < loop2.getNumResults(); ++i)
      loop2.getResult(i).replaceAllUsesWith(fusedLoop.getResult(n1 + i));

    loop2.erase();
    loop1.erase();
  }
};

} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(ForLoopFusionPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(ForLoopFusionPass)

mlir::PassPluginLibraryInfo getForLoopFusionPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "ForLoopFusionPass", "1.0",
          []() { mlir::PassRegistration<ForLoopFusionPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return getForLoopFusionPassPluginInfo();
}