package main

import (
	"fmt"
	"github.com/janpfeifer/must"
	"os"
	"os/exec"
	"text/template"
)

const (
	enumGoFileName = "gen_op_types.go"
	enumCFileName  = "../c/gomlx/xlabuilder/gen_op_types.h"
)

var (
	enumGoTemplate = template.Must(template.New(enumGoFileName).Parse(`
package xlabuilder

/***** File generated by gopjrt/internal/cmd/xlabuilder_codegen, based on op_types.txt. Don't edit it directly. *****/

// OpType enumerates the various operation types supported by XLA.
type OpType int32

const (
{{range .}}	{{.}}
{{end}})
`))

	enumCTemplate = template.Must(template.New(enumCFileName).Parse(`
/***** File generated by gopjrt/internal/cmd/xlabuilder_codegen, based on xlabuilder/op_types.txt. Don't edit it directly. *****/

#ifndef _GOMLX_XLABUILDER_GEN_OP_TYPES_H
#define _GOMLX_XLABUILDER_GEN_OP_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

enum OpType {
{{range .}}  {{.Name}}Op,
{{end}}};

#ifdef __cplusplus
}
#endif

#endif 
`))
)

func generateOpsEnums(opsInfo []OpInfo) {
	declarations := make([]string, len(opsInfo))
	for ii, info := range opsInfo {
		if ii == 0 {
			declarations[ii] = fmt.Sprintf("%sOp OpType = iota", info.Name)
		} else {
			declarations[ii] = fmt.Sprintf("%sOp", info.Name)
		}
	}

	// Go Enums
	f := must.M1(os.Create(enumGoFileName))
	must.M(enumGoTemplate.Execute(f, declarations))
	must.M(exec.Command("gofmt", "-w", enumGoFileName).Run())
	fmt.Printf("Generated %q based on %q\n", enumGoFileName, OpTypesFileName)
	// Generate enum names with Go stringer.
	must.M(exec.Command("stringer", "-type=OpType", enumGoFileName).Run())

	// C Enums
	f = must.M1(os.Create(enumCFileName))
	must.M(enumCTemplate.Execute(f, opsInfo))
	fmt.Printf("Generated %q based on %q\n", enumCFileName, OpTypesFileName)

}
