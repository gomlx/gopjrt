package(default_visibility = ["//visibility:public"])

# Statically linked PJRT CPU plugin.
#
# Note: cc_static_library only available in bazel >= 7.4.0:
cc_static_library(
    name = "pjrt_c_api_cpu_static",
    deps = [
        "@xla//xla/pjrt/c:pjrt_c_api_cpu",
    ],
)
