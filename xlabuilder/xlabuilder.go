package xlabuilder

// #cgo LDFLAGS: -lgomlx_xlabuilder -lstdc++ -lm
/*
#include <gomlx/xlabuilder/xlabuilder.h>
*/
import "C"
import (
	"fmt"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"runtime"
	"unsafe"
)

// CVersion returns the version of the XlaBuilder C/C++ wrapper.
//
// This uses the same "semantic versioning" numbers (e.g. "v0.6.0") used by the Go,
// and follows (with a lag) the Gopjrt version.
//
// This often lags behind Gopjrt version, if/when the C/C++ wrapper doesn't change --
// we don't bump the version of the C/C++ code if it doesn't change.
// But when it changes, it matches the Gopjrt version it's being released with.
//
// At initialization this value is checked if it matches the version required by
// Gopjrt, and will print a warning if it doesn't match.
func CVersion() string {
	return C.GoString(C.GopjrtXlaBuilderVersion)
}

// MatchingCVersion is the Gopjrt XlaBuilder C/C++ wrapper version that matches the
// Go library.
//
// This is needed because they can go out-of-sync in developers machines -- if one updates
// the Go library, but not the corresponding C/C++ libgomlx_xlabuilder.so library.
var MatchingCVersion = "v0.6.3"

func init() {
	if CVersion() != MatchingCVersion {
		klog.Errorf(
			"Gopjrt C/C++ library libgomlx_xlabuilder.so version is %q, but this Gopjrt "+
				"version requires libgomlx_xlabuilder.so version %q. "+
				"See https://github.com/gomlx/gopjrt on how to install the newest version.",
			CVersion(), MatchingCVersion)
	}
}

//go:generate go run ../cmd/xlabuilder_codegen

// Since CGO C types cannot cross boundaries of a package (see issue https://github.com/golang/go/issues/13467)
// We make a copy of chelper.go for every sub-directory that needs it.
//go:generate go run ../cmd/copy_go_code --original=chelper.go

// panicf panics with formatted description.
//
// It is only used for "bugs in the code" -- when parameters don't follow the specifications.
// In principle, it should never happen -- the same way nil-pointer panics should never happen.
func panicf(format string, args ...any) {
	panic(errors.Errorf(format, args...))
}

// XlaBuilder is used to create "computations" (XlaComputation), that are like "StableHLO" functions.
//
// In turn XlaComputation can be exported to a serialized `HloModuleProto` (a binary blob) and used by a PJRT plugin
// (see github.com/gomlx/gopjrt/pjrt package) to compile and execute on accelerators.
//
// Once created (New), one can issue "operations" ("ops" for short), like "Add", "Mul", etc, which are recorded.
// When the computation definition is finalized, call "XlaBuilder.Build" to get the XlaComputation representing
// the function built.
// The XlaComputation can then be used with PJRT (see XlaComputation.SerializedHLO), or pretty (+/-, relatively speaking)
// print (text, HTML, graphviz, etc). See XlaComputation documentation.
//
// Once done (usually, just after StableHLO is called) deallocate the underlying C++ resources by calling Destroy.
//
// Some observations:
//
//   - The XlaBuilder is used by all ops creating functions (like "Add", "Mul", etc.). But since the input of most ops,
//     are other created ops, and they hold a link to the XlaBuilder, there is no need to explicitly pass the XlaBuilder to
//     every op function.
type XlaBuilder struct {
	// cXlaBuilder is registered to be destroyed in the finalizer -> only use it protected by a runtime.KeepAlive(XlaBuilder).
	cXlaBuilder *C.XlaBuilder
	name        string
	parent      *XlaBuilder // parent builder, if created with CreateSubBuilder.

	// cacheStandardComputations are sub-computations used on standard operations, like Reduce{Max,Min,Mul,Sum}.
	cachedStandardComputations map[string]*XlaComputation

	// cacheStandardComputations for things like the initial values for reductions, for each dtype.
	cachedStandardConstants map[string]*Op
}

// New create a new XlaBuilder with the given name, that can be used to create a new StableHLO program.
// See details on how to use it on XlaBuilder.
func New(name string) *XlaBuilder {
	var cBuilder *C.XlaBuilder
	cName := C.CString(name)
	defer cFree(cName)
	cBuilder = (*C.XlaBuilder)(C.NewXlaBuilder(cName))
	return newXlaBuilder(cBuilder)
}

func newXlaBuilder(cXlaBuilder *C.XlaBuilder) *XlaBuilder {
	b := &XlaBuilder{
		cXlaBuilder:                cXlaBuilder,
		name:                       cStrFree(C.XlaBuilderName(unsafe.Pointer(cXlaBuilder))),
		cachedStandardComputations: map[string]*XlaComputation{},
		cachedStandardConstants:    map[string]*Op{},
	}
	runtime.SetFinalizer(b, func(b *XlaBuilder) { b.Destroy() })
	return b
}

// IsNil returns true if either b is nil or the contained C++ XlaBuilder. Usually true for destroyed XlaBuilder objects.
func (b *XlaBuilder) IsNil() bool { return b == nil || b.cXlaBuilder == nil }

// Destroy and free the underlying C++ object.
// It can be called more than once -- once finalized the first time, it becomes a no-op.
//
// It is called at garbage-collection automatically.
func (b *XlaBuilder) Destroy() {
	if b.IsNil() {
		return
	}
	b.parent = nil // Help the GC just in case.
	for _, subComp := range b.cachedStandardComputations {
		subComp.Destroy()
	}
	b.cachedStandardComputations = nil
	C.XlaBuilderDestroy(unsafe.Pointer(b.cXlaBuilder))
	b.cXlaBuilder = nil
}

// Name returns the name after it was canonicalized by the XlaBuilder library -- so it may be different from the
// one given.
func (b *XlaBuilder) Name() string {
	if b == nil {
		return "<nil>"
	}
	if b.cXlaBuilder == nil {
		return fmt.Sprintf("%s (destroyed)", b.name)
	}
	return b.name
}

// addOp will add the operation described by op.
// If it succeeds it fills the fields Op.index and Op.op, with the C++ references.
func (b *XlaBuilder) addOp(op *Op) error {
	if b.IsNil() {
		return errors.Errorf("trying to add op %s to a nil XlaBuilder", op.Type)
	}
	if op.builder != nil {
		return errors.Errorf("XlaBuilder.Op %s being added seems to have been already added to some cXlaBuilder", op.Type)
	}
	for ii, input := range op.OpInputs {
		if input.builder != b {
			return errors.Errorf("XlaBuilder.Op %s being added to a different builder than its input #%d (%s)", op.Type, ii, input.Type)
		}
	}

	op.builder = b
	if op.Type == IdentityOp {
		op.cOp = op.OpInputs[0].cOp
		op.Shape = op.OpInputs[0].Shape
		return nil
	}
	serializedOp := serializeToC(op)

	var pinner runtime.Pinner
	defer pinner.Unpin()
	pinner.Pin(b)
	pinner.Pin(serializedOp)

	err := errorFromStatus(C.XlaBuilderAddOp(unsafe.Pointer(b.cXlaBuilder), serializedOp))
	if err != nil {
		return errors.Wrapf(err, "while trying to add op %s (%d) to XlaBuilder", op.Type, int(op.Type))
	}
	op.cOp = (*C.XlaOp)(serializedOp.new_op)
	serializedOp.new_op = nil // Ownership transferred.
	op.Shape = shapeFromCShape(serializedOp.new_shape)
	destroyCSerializedOp(serializedOp)
	return nil
}

// Build builds the computation (*XlaComputation) with the requested operations (the outputOp and all its dependencies)
// or returns a non-ok status.
//
// Note that all ops that have been enqueued will be moved to the computation being returned and will no longer be valid.
func (b *XlaBuilder) Build(outputOp *Op) (*XlaComputation, error) {
	if b.IsNil() {
		return nil, errors.New("trying to access XlaBuilder that is nil or already destroyed")
	}
	statusOr := C.XlaBuilderBuildComp(unsafe.Pointer(b.cXlaBuilder), unsafe.Pointer(outputOp.cOp))
	var err error
	var cComp *C.XlaComputation
	cComp, err = pointerOrError[C.XlaComputation](statusOr)
	if err != nil {
		return nil, errors.Wrapf(err, "while building the computation with the XlaBuilder (outputOp=%s)", outputOp.Type)
	}

	return newXlaComputation(cComp), nil
}

// CreateSubBuilder returns a new XlaBuilder whose resultant Computation is used only by this
// XlaBuilder.
//
// Some operations, like Call and Reduce, take as input a sub-computation (the reduction function), that can be created
// with a sub-builder.
//
// It takes as input the computationName that is going to be built with it.
func (b *XlaBuilder) CreateSubBuilder(computationName string) *XlaBuilder {
	if b.IsNil() {
		panicf("trying to access XlaBuilder that is nil or already destroyed")
	}
	cName := C.CString(computationName)
	defer cFree(cName)
	var cNewBuilder *C.XlaBuilder
	cNewBuilder = (*C.XlaBuilder)(C.XlaBuilderCreateSubBuilder(unsafe.Pointer(b.cXlaBuilder), cName))
	newB := newXlaBuilder(cNewBuilder)
	newB.parent = b
	return newB
}
