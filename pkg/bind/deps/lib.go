package deps

import (
	"fmt"
	"os"
)

func DumpSDLibrary(gpu bool) (filename string, err error) {
	file, err := os.CreateTemp("", libName)
	if err != nil {
		err = fmt.Errorf("error creating temp file: %w", err)
		return
	}

	err = os.WriteFile(file.Name(), getLib(gpu), 0400)
	if err != nil {
		err = fmt.Errorf("error writing file: %w", err)
		return
	}

	err = file.Close()
	if err != nil {
		err = fmt.Errorf("error closing file: %w", err)
		return
	}

	filename = file.Name()
	return
}
