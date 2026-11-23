#!/usr/bin/env bash
# Build step run by the evaluator before anything else (cwd: the workspace). Must be repeatable.
# On a Mac there is no offline `metal` compiler without Xcode: compile Metal source at run time (mx.fast.metal_kernel,
# or MTLDevice newLibraryWithSource from Objective-C++ built here with clang).
set -euo pipefail
