def gomlx_xlabuilder_genrule(target_platform):
    """
    Generates a genrule for building the gomlx_xlabuilder tarball for the given target platform.

    Args:
    target_platform: The target platform (e.g., "linux_amd64", "darwin_arm64").
    """
    native.genrule(
        name = "gomlx_xlabuilder_" + target_platform,
        srcs = [
            ":gomlx_xlabuilder_headers_include",
            ":gomlx_xlabuilder_static_lib",
        ],
        outs = ["gomlx_xlabuilder_" + target_platform + ".tar.gz"],
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
