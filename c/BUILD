load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

package(default_visibility = ["//visibility:private"])

# This is the genrule used by TF install headers to correctly
# map headers into a directory structure
genrule(
    name = "gomlx_xlabuilder_headers_include",
    srcs = ["//gomlx/xlabuilder:headers"],
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

genrule(
    name = "gomlx_xlabuilder_dynamic_lib",
    srcs = [
        "//gomlx/xlabuilder:libgomlx_xlabuilder.so",
    ],
    outs = ["lib/libgomlx_xlabuilder.so"],
    cmd = """
    mkdir -p "$(@D)"
    ln $(location //gomlx/xlabuilder:libgomlx_xlabuilder.so) "$@"
    """,
)

# Waiting for cc_static_library.
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

# Tar packaging: notice we force date/time and ordering to be deterministic, to generate the
# always same archive, if the contents are the same. See https://reproducible-builds.org/docs/archives/.
genrule(
    name = "gomlx_xlabuilder",
    srcs = [
        ":gomlx_xlabuilder_headers_include",
        ":gomlx_xlabuilder_dynamic_lib",
        ":gomlx_xlabuilder_static_lib",
    ],
    outs = ["gomlx_xlabuilder.tar.gz"],
    cmd_bash = """
    tt="$$(mktemp -d)"
    echo "tt=$${tt}"
    build_dir="$$(pwd)/$(@D)"
    echo "build_dir=$${build_dir}"
    files="lib include"
    for ii in $${files} ; do
        ln -sf "$${build_dir}/$${ii}" "$${tt}"
    done
    echo "files=$${files}"
    TAR=tar
    if [[ "$$OSTYPE" == "darwin"* ]]; then
        if ! command -v gtar &> /dev/null; then
            echo -e "\\033[1;31mgtar (gnu-tar) cannot not be found. gnu-tar can be installed using:\nbrew install gnu-tar"
        exit
        fi
        TAR=gtar
    fi
    $$TAR --create --dereference --directory="$${tt}" \
        --sort=name --owner=0 --group=0 --numeric-owner --format=gnu \
        --gzip --file "$@" $${files}
    rm -rf "$${tt}"
    """,
)

# Dummy meta-rule that simply compile gomlx_xlabuilder but doesnt' package it.
genrule(
    name = "gomlx_xlabuilder_quick",
    srcs = [
        ":gomlx_xlabuilder_headers_include",
        ":gomlx_xlabuilder_lib",
    ],
    outs = ["gomlx_xlabuilder_quick.txt"],
    cmd = """
    touch "$@"
    """,
)
