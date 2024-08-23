#!/bin/bash

# Run this from the root of gopjrt repository to generate docs/coverage.out with the coverage data.

PACKAGE_COVERAGE="./pjrt ./xlabuilder"
go test -v -cover -coverprofile docs/coverage.out -coverpkg ${PACKAGE_COVERAGE}
go tool cover -func docs/coverage.out -o docs/coverage.out