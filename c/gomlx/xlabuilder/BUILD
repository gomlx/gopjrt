package(default_visibility = ["//visibility:public"])

# Dependencies needed for XlaBuilder and XlaComputation
# A stripped-down version of what is defined in the `xla_cc_binary` rule, defined in //@xla/xla/xla.bzl.
gomlx_xlabuilder_deps = [
    "@com_google_absl//absl/types:span",
    "@com_google_absl//absl/types:optional",
    "@com_google_absl//absl/base:log_severity",
    "@com_google_absl//absl/log:initialize",
    "@xla//xla:comparison_util",
    "@xla//xla:literal",
    "@xla//xla:shape_util",
    "@xla//xla:status",
    "@xla//xla:statusor",
    "@xla//xla:types",
    "@xla//xla:util",
    "@xla//xla:xla_data_proto_cc",
    "@xla//xla:xla_proto_cc",
    "@xla//xla/hlo/builder:xla_builder",
    "@xla//xla/hlo/builder:xla_computation",
    "@xla//xla/hlo/builder/lib:math",
]

cc_library(
    name = "xlabuilder",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    # SKIP_ABSL_INITIALIZE_LOG: this will trigger skipping of the call to `absl::InitializeLog()`
    # to prevent double-calling, in case whoever is linking GoMLX also calls it. Setting it may
    # lead to spurious logging by XLA library at startup up.
    # defines = ["SKIP_ABSL_INITIALIZE_LOG"],
    linkopts = ["-shared"],
    deps = gomlx_xlabuilder_deps,
    alwayslink = True,
)

cc_binary(
    name = "libgomlx_xlabuilder.so",
    linkopts = [
        "-shared",
    ],
    # Setting linkshared=1 and linkstatic=1 should try to link everything (except system libraries) statically
    # but generate a dynamically linked library -- needed to link it to other languages, Go in this case.
    # At least that's what I understood from reading the docs in
    # https://bazel.build/reference/be/c-cpp#cc_binary.linkshared
    linkshared = 1,
    linkstatic = 1,
    deps = gomlx_xlabuilder_deps + [
        ":xlabuilder",
    ],
)

# Waiting for Bazel 7.4.0:
#cc_static_library(
#    name = "libgomlx_xlabuilder.a",
#    deps = gomlx_xlabuilder_deps + [
#        ":xlabuilder",
#    ],
#)

filegroup(
    name = "headers",
    srcs = glob(["*.h"]),
)
