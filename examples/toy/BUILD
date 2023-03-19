load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

td_library(
    name = "ToyOpsTdFiles",
    srcs = [
        "include/toy/Ops.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "ToyOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "include/toy/Ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "include/toy/Ops.cpp.inc",
        ),
        (
            ["-gen-dialect-decls"],
            "include/toy/Dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "include/toy/Dialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/toy/Ops.td",
    deps = [":ToyOpsTdFiles"],
)

cc_binary(
    name = "toyc-ch2",
    srcs = [
        "toyc.cpp",
        "parser/AST.cpp",
        "mlir/MLIRGen.cpp",
        "mlir/Dialect.cpp",
    ] + glob(["include/toy/*.h"]),
    includes = ["include/"],
    deps = [
        ":ToyOpsIncGen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "toy_dialect_lib",
    srcs = ["mlir/Dialect.cpp"],
    includes = ["include/"],
    hdrs = glob(["include/toy/*.h"]),
    deps = [
        ":ToyOpsIncGen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "dummy_pass",
    srcs = ["dummy_pass.cc"],
    hdrs = ["dummy_pass.h"],
    deps = [
        ":toy_dialect_lib",
        "@llvm-project//mlir:Pass",
    ],
)

cc_test(
    name = "ops_test",
    size = "small",
    srcs = ["ops_test.cc"],
    includes = ["include/"],
    deps = [
        ":dummy_pass",
        ":toy_dialect_lib",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)
