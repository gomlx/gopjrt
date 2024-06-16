load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

package(default_visibility = ["//visibility:private"])

# This is the genrule used by TF install headers to correctly
# map headers into a directory structure
genrule(
    name = "gomlx_xlabuilder_headers_include",
    srcs = ["//gomlx/xlabuilder:headers"],
    outs = ["include"],
    cmd = """
    mkdir -p "$@"
    mkdir -p "$@/gomlx"
    echo "out=$@"
    for f in $(locations //gomlx/xlabuilder:headers); do
      echo "$${f}"
      cp "$${f}" "$@/$${f}"
    done
    """,
)

# This genrule remaps libxla_extension.so and libtcmalloc.a to lib/libxla_extension.so
genrule(
    name = "gomlx_xlabuilder_lib",
    srcs = [
        "//gomlx/xlabuilder:libgomlx_xlabuilder.so",
        "@gperftools//:libtcmalloc.a",
    ],
    outs = ["lib"],
    cmd = """
    mkdir "$@"
    ln $(location //gomlx/xlabuilder:libgomlx_xlabuilder.so) "$@"

    mkdir "$@/gomlx"
    GPERFTOOLS_LIB="$$(pwd)/$$(dirname $(location @gperftools//:libtcmalloc.a))"
    if [ -d "$$GPERFTOOLS_LIB"/lib ]; then
      GPERFTOOLS_LIB="$$GPERFTOOLS_LIB"/lib
    fi
    ln "$$GPERFTOOLS_LIB"/lib* "$@/gomlx"
    """,
)

# Tar packaging: notice we force date/time and ordering to be deterministic, to generate the
# always same archive, if the contents are the same. See https://reproducible-builds.org/docs/archives/.
genrule(
    name = "gomlx_xlabuilder",
    srcs = [
        ":gomlx_xlabuilder_headers_include",
        ":gomlx_xlabuilder_lib",
    ],
    outs = ["gomlx_xlabuilder.tar.gz"],
    cmd = """
    tt="$$(mktemp -d)"
    files=""
    for ii in $(SRCS) ; do
        ln -sf "$$(pwd)/$${ii}" "$${tt}"
        files="$${files} $$(basename $${ii})"
    done
    echo "$${files}"
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
