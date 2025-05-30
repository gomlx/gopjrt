package(default_visibility = ["//visibility:public"])

# Dependencies needed for XlaBuilder and XlaComputation
# A stripped-down version of what is defined in the `xla_cc_binary` rule, defined in //@xla/xla/xla.bzl.
gomlx_xlabuilder_deps = [
    "@com_google_absl//absl/types:span",
    "@com_google_absl//absl/types:optional",
    "@com_google_absl//absl/base:log_severity",
    "@com_google_absl//absl/log:initialize",
    "@com_google_absl//absl/status",
    "@com_google_absl//absl/status:statusor",
    "@xla//xla:comparison_util",
    "@xla//xla:literal",
    "@xla//xla:shape_util",
    "@xla//xla:types",
    "@xla//xla:util",
    "@xla//xla:xla_data_proto_cc",
    "@xla//xla:xla_proto_cc",
    "@xla//xla/hlo/builder:xla_builder",
    "@xla//xla/hlo/builder:xla_computation",
    "@xla//xla/hlo/builder/lib:math",
]

# StableHLO dependencies
stablehlo_deps = [
    # StableHLO/MLIR conversion
    "@llvm-project//mlir:BytecodeWriter",
    "@llvm-project//mlir:IR",
    "@stablehlo//:register",
    "@stablehlo//:stablehlo_serialization",
    "@xla//xla/hlo/translate/hlo_to_mhlo:hlo_to_mlir_hlo",
]

config_setting(
    name = "use_stablehlo",
    values = {"define": "use_stablehlo=true"},
)

cc_library(
    name = "xlabuilder",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    defines = select({
        ":use_stablehlo": ["USE_STABLEHLO"],
        "//conditions:default": [],
    }),
    # SKIP_ABSL_INITIALIZE_LOG: this will trigger skipping of the call to `absl::InitializeLog()`
    # to prevent double-calling, in case whoever is linking GoMLX also calls it. Setting it may
    # lead to spurious logging by XLA library at startup up.
    # defines = ["SKIP_ABSL_INITIALIZE_LOG"],
    linkopts = ["-shared"],
    deps = gomlx_xlabuilder_deps + select({
        ":use_stablehlo": stablehlo_deps,
        "//conditions:default": [],
    }),
    alwayslink = True,
)

# Static version of the gomlx_xlabuilder library.
#
# Note: cc_static_library only available in bazel >= 7.4.0:
cc_static_library(
    name = "gomlx_xlabuilder",
    deps = gomlx_xlabuilder_deps + [
        ":xlabuilder",
    ] + select({
        ":use_stablehlo": stablehlo_deps,
        "//conditions:default": [],
    }),
)

filegroup(
    name = "headers",
    srcs = glob(["*.h"]),
)
