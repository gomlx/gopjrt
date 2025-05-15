// protoc_xla_protos compiles the .proto from the OpenXLA/XLA sources to subpackages of
// "github.com/gomlx/gopjrt/internal/protos".
//
// It should be executed under the gopjrt/internal/protos directory -- suggested as a go:generate --
// and it requires XLA_SRC to be set to a cloned github.com/openxla/xla clone.
//
// It first removes from the proto any lines with `option go_package = "...";`, since they
// wouldn't allow the compilation. See https://github.com/golang/protobuf/issues/1621 on this issue.
// The **files are modified** in place.
package main

import (
	"fmt"
	"github.com/pkg/errors"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	xlaSrcEnvVar = "XLA_SRC"
	basePackage  = "github.com/gomlx/gopjrt/internal/protos"
)

var protos = []string{
	"xla/tsl/protobuf/dnn.proto",
	"xla/autotune_results.proto",
	"xla/autotuning.proto",
	"xla/pjrt/compile_options.proto",
	"xla/service/hlo.proto",
	"xla/service/metrics.proto",
	"xla/stream_executor/cuda/cuda_compute_capability.proto",
	"xla/stream_executor/device_description.proto",
	"xla/xla.proto",
	"xla/xla_data.proto",
}

func main() {
	xlaSrc := os.Getenv(xlaSrcEnvVar)
	if xlaSrc == "" {
		log.Fatalf("Please set %s to the directory containing the cloned github.com/openxla/xla repository.\n", xlaSrcEnvVar)
	}

	// Generate the --go_opt=M... flags
	goOpts := make([]string, len(protos))
	for ii, proto := range protos {
		goPackage := filepath.Join(basePackage, protoPackage(proto))
		goOpts[ii] = fmt.Sprintf("--go_opt=M%s=%s", proto, goPackage)
	}
	goOpts = append(goOpts, "--go_opt=Mtsl/protobuf/dnn.proto="+filepath.Join(basePackage, "dnn"))

	for _, proto := range protos {
		packageName := protoPackage(proto)
		err := os.Mkdir(packageName, 0755)
		if err != nil && !os.IsExist(err) {
			log.Fatalf("Failed to create sub-directory %q: %+v", packageName, err)
		}
		// Remove go_package options from the proto file
		protoPath := filepath.Join(xlaSrc, proto)
		if err := removeGoPackageOption(protoPath); err != nil {
			log.Fatalf("Error removing go_package option from %s: %v\n", proto, err)
		}

		// Construct the protoc command
		args := []string{
			"--go_out=./" + protoPackage(proto),
			"-I=" + xlaSrc,
			"-I=" + filepath.Join(xlaSrc, "third_party/tsl"),
			fmt.Sprintf("--go_opt=module=%s", filepath.Join(basePackage, packageName)),
		}
		args = append(args, goOpts...)
		args = append(args, protoPath)

		cmd := exec.Command("protoc", args...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr

		if err := cmd.Run(); err != nil {
			log.Printf("Command:\n%s\n", cmd)
			log.Fatalf("Error executing protoc for %s: %v\n", proto, err)
		}

		currentDir, err := os.Getwd()
		if err != nil {
			log.Fatalf("Failed to get current directory: %v", err)
		}
		localCopyPath := path.Join(currentDir, path.Base(protoPath))
		if err := copyFile(localCopyPath, protoPath); err != nil {
			log.Fatalf("Failed to copy file: %v", err)
		}
	}
}

func protoPackage(protoPath string) string {
	pkg := path.Base(protoPath)
	pkg = strings.TrimSuffix(pkg, ".proto")
	return pkg
}

func removeGoPackageOption(protoPath string) error {
	content, err := os.ReadFile(protoPath)
	if err != nil {
		return err
	}

	re := regexp.MustCompile(`option\s+go_package\s*=\s*"[^"]*?";`)
	newContent := re.ReplaceAll(content, []byte{})

	return os.WriteFile(protoPath, newContent, 0644)
}

func copyFile(dst, src string) error {
	// Read all content of src to data, may cause OOM for a large file.
	data, err := os.ReadFile(src)
	if err != nil {
		return errors.Wrapf(err, "failed to read %q", src)
	}

	// Write data to dst
	err = os.WriteFile(dst, data, 0644)
	if err != nil {
		return errors.Wrapf(err, "failed to read %q", src)
	}
	return nil
}
