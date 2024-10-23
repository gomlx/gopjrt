#include <string>
#include <stdlib.h>

#include "xlabuilder.h"
#include "gomlx/xlabuilder/utils.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"



#ifdef USE_STABLEHLO

// StableHLO will be linked in.
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Version.h"        // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"        // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace std;

const bool HasStableHLO = true;

// LoadHloDialects to the context: apparently not needed -- earlier I was having an error in Dialect.
static void LoadHloDialects(mlir::MLIRContext& context) {
    mlir::DialectRegistry registry;
    mlir::stablehlo::registerAllDialects(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
}

static absl::Status ConvertHloToMhlo(const xla::HloModuleProto* proto,
                                     mlir::ModuleOp* mlir_module) {
  auto status = xla::ConvertHloToMlirHlo(*mlir_module, proto,
                                         /*import_all_computations=*/false);
  if (!status.ok()) {
    return status;
  }
  if (!mlir::verify(*mlir_module).succeeded()) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "MHLO Module from HLO -> MHLO conversion is not legal.");
  }
  return absl::OkStatus();
}

static absl::StatusOr<unique_ptr<mlir::ModuleOp>> ConvertToStableHLO(mlir::MLIRContext &context, xla::HloModuleProto *proto) {
    auto mlir_module = std::make_unique<mlir::ModuleOp>(
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
    auto status = ConvertHloToMhlo(proto, mlir_module.get());
    if (!status.ok()) {
        return status;
    }
    return mlir_module;
}

absl::StatusOr<std::string> SerializeMLIRUsingBytecode(const mlir::ModuleOp &mlir_module) {
    std::string bytecode;
    llvm::raw_string_ostream os(bytecode);
    const std::string stablehlo_version = mlir::vhlo::Version::getCurrentVersion().toString();
    auto result = mlir::stablehlo::serializePortableArtifact(
          mlir_module, /* target_version = */ stablehlo_version, os);
    if (!result.succeeded()) {
        return absl::InvalidArgumentError("mlir::stablehlo::serializePortableArtifact() failed");
    }
    return bytecode;
}

// "Legalizing" (whatever that means) the MLIR: copied from github.com/pytorch/xla, and commented out PyTorch stuff.
static absl::Status mhloToStablehloHelper(mlir::ModuleOp* mlir_module,
                                          mlir::MLIRContext* context) {
  mlir::PassManager pm(context);

  // pm.addPass(torch_xla::runtime::CreatePrepareXlaMlirDebuginfoPass());

  // legalize `mhlo.dot` to `mhlo.dot_general` to workaround the shape
  // refinement issue in `stablehlo.dot`.
  // TODO(lsy323): Remove this pass when mhlo.dot will can be leagalized to
  // stablehlo.dot_general in MHLO->StableHLO converter. Or shape refinement
  // logic is fixed for stablehlo.dot.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeDotToDotGeneralPass());
  // Apply pass to remove HLO tuple output, as MHLO/StableHLO supports multiple
  // outputs.
  pm.addPass(mlir::mhlo::createExpandHloTuplesPass());
  // Canonicalization after tuple flatten, to remove unused tuple op.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());

  // Group patterns into StableHLO composites.
    //  pm.addPass(torch_xla::runtime::CreateBuildStableHLOCompositePass());
    //  pm.addNestedPass<mlir::func::FuncOp>(torch_xla::runtime::CreateRemoveXlaMarkTensorOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  if (!mlir::succeeded(pm.run(*mlir_module))) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "StableHLO Module from MHLO -> StableHLO conversion is not legal.");
  }
  return absl::OkStatus();
}

// XlaComputation -> StableHLO -> Serialized bytes
StatusOr XlaComputationSerializedStableHLO(XlaComputation *xla_comp) {
    mlir::MLIRContext context;
    // LoadHloDialects(context);

    StatusOr r{0, 0};
    absl::StatusOr<unique_ptr<mlir::ModuleOp>> mlir_or = ConvertToStableHLO(context, xla_comp->mutable_proto());
    if (!mlir_or.ok()) {
        r.status = FromStatus(mlir_or.status());
        return r;
    }

    auto status = mhloToStablehloHelper(mlir_or.value().get(), &context);
    if (!status.ok()) {
        r.status = FromStatus(status);
        return r;
    }

    absl::StatusOr<string> mlir_str_or = SerializeMLIRUsingBytecode(*mlir_or.value());
    if (!mlir_str_or.ok()) {
        r.status = FromStatus(mlir_str_or.status());
        return r;
    }
    r.value = str_to_bytes(mlir_str_or.value());
    return r;
}


static std::string PrintMLIRModule(mlir::ModuleOp *mlir_module) {
    std::string str;
    llvm::raw_string_ostream os(str);
    mlir::OpPrintingFlags flags;
    // flags.enableDebugInfo();
    mlir_module->print(os, flags);
    return str;
}

// XlaComputation -> StableHLO -> Text
StatusOr XlaComputationStableHLOText(XlaComputation *xla_comp) {
    mlir::MLIRContext context;
    // LoadHloDialects(context);

    StatusOr r{0, 0};
    absl::StatusOr<unique_ptr<mlir::ModuleOp>> mlir_or = ConvertToStableHLO(context, xla_comp->mutable_proto());
    if (!mlir_or.ok()) {
        r.status = FromStatus(mlir_or.status());
        return r;
    }

    r.value = c_str(PrintMLIRModule(mlir_or.value().get()));
    return r;
}

#else   // USE_STABLEHLO

static const char *_NOT_INCLUDED = "StableHLO support was not included in this build";

const bool HasStableHLO = false;

// XlaComputation -> StableHLO -> Text
// Not implemented version.
StatusOr XlaComputationStableHLOText(XlaComputation *xla_comp) {
    StatusOr r{0, 0};
    r.status = FromStatus(absl::Status(absl::StatusCode::kUnimplemented, _NOT_INCLUDED));
    return r;
}

// XlaComputation -> StableHLO -> Serialized bytes
StatusOr XlaComputationSerializedStableHLO(XlaComputation *xla_comp) {
    StatusOr r{0, 0};
    r.status = FromStatus(absl::Status(absl::StatusCode::kUnimplemented, _NOT_INCLUDED));
    return r;
}
#endif  // USE_STABLEHLO