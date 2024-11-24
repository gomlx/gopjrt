load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("gomlx_xlabuilder.bzl", "gomlx_xlabuilder_genrule")

package(default_visibility = ["//visibility:private"])

# This is the genrule used by TF install headers to correctly
# map headers into a directory structure
genrule(
    name = "gomlx_xlabuilder_headers_include",
    srcs = [
        "//gomlx/xlabuilder:headers",
    ],
    outs = ["include/gomlx/xlabuilder/xlabuilder.h"],  # We list only one, because it's hard to keep all the list in sync.
    cmd = """
    mkdir -p "$(@D)"
    echo "out=$(@D)"
    for f in $(locations //gomlx/xlabuilder:headers); do
      echo "$${f} -> $(@D)"
      cp "$${f}" "$(@D)/"
    done
    """,
)

# XlaBuilder static library.
genrule(
    name = "gomlx_xlabuilder_static_lib",
    srcs = [
        "//gomlx/xlabuilder:gomlx_xlabuilder",
    ],
    outs = ["lib/libgomlx_xlabuilder.a"],
    cmd = """
    mkdir -p "$(@D)"
    ln $(location //gomlx/xlabuilder:gomlx_xlabuilder) "$@"
    """,
)

# PJRT static library.
genrule(
    name = "pjrt_cpu_static_lib",
    srcs = [
        "//gomlx/pjrt:pjrt_c_api_cpu_static",
    ],
    outs = ["lib/libpjrt_c_api_cpu_static.a"],
    cmd = """
    mkdir -p "$(@D)"
    ln $(location //gomlx/pjrt:pjrt_c_api_cpu_static) "$@"
    """,
)

# PJRT static library.
genrule(
    name = "pjrt_cpu_dynamic_lib",
    srcs = [
        "@xla//xla/pjrt/c:pjrt_c_api_cpu_plugin.so",
    ],
    outs = ["lib/libpjrt_c_api_cpu_dynamic.so"],
    cmd = """
    mkdir -p "$(@D)/gomlx/pjrt/" ;
    ln -f "$(location @xla//xla/pjrt/c:pjrt_c_api_cpu_plugin.so)" "$(@D)/gomlx/pjrt/pjrt_c_api_cpu_plugin.so" ;
    pushd "$(@D)" ;
    ln -sf "gomlx/pjrt/pjrt_c_api_cpu_plugin.so" "libpjrt_c_api_cpu_dynamic.so" ;
    ln -sf "gomlx/pjrt/pjrt_c_api_cpu_plugin.so" "pjrt_c_api_cpu_plugin.so" ;
    popd
    """,
)

# Generate genrules for each supported platform.
gomlx_xlabuilder_genrule("linux_amd64")

gomlx_xlabuilder_genrule("darwin_amd64")

gomlx_xlabuilder_genrule("darwin_arm64")
