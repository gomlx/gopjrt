// protoc_xla_protos compiles the .proto from the OpenXLA/XLA sources to subpackages of
// "github.com/gomlx/gopjrt/protos".
//
// It should be executed under the gopjrt/protos directory -- suggested as a go:generate --
// and it requires XLA_SRC to be set to a cloned github.com/openxla/xla clone.
//
// It first removes from the proto any lines with `option go_package = "...";`, since they
// wouldn't allow the compilation. See https://github.com/golang/protobuf/issues/1621 on this issue.
// The **files are modified** in place.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	xlaSrcEnvVar = "XLA_SRC"
	basePackage  = "github.com/gomlx/gopjrt/protos"
)

var protos = []string{
	"third_party/tsl/tsl/protobuf/dnn.proto",
	"xla/autotune_results.proto",
	"xla/autotuning.proto",
	"xla/pjrt/compile_options.proto",
	"xla/service/hlo.proto",
	"xla/stream_executor/device_description.proto",
	"xla/xla.proto",
	"xla/xla_data.proto",
}

func main() {
	xlaSrc := os.Getenv(xlaSrcEnvVar)
	if xlaSrc == "" {
		fmt.Fprintf(os.Stderr, "Please set %s to the directory containing the cloned github.com/openxla/xla repository.\n", xlaSrcEnvVar)
		os.Exit(1)
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
			fmt.Fprintf(os.Stderr, "Failed to create sub-directory %q: %+v", packageName, err)
			os.Exit(1)
		}
		// Remove go_package options from the proto file
		protoPath := filepath.Join(xlaSrc, proto)
		if err := removeGoPackageOption(protoPath); err != nil {
			fmt.Fprintf(os.Stderr, "Error removing go_package option from %s: %v\n", proto, err)
			os.Exit(1)
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
			fmt.Fprintf(os.Stderr, "Error executing protoc for %s: %v\n", proto, err)
			fmt.Fprintf(os.Stderr, "Command:\n%s\n", cmd)
			os.Exit(1)
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
