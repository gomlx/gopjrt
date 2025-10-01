package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/charmbracelet/bubbles/key"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/huh"
	"github.com/charmbracelet/huh/spinner"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/term"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

var (
	// ErrUserAborted is returned when the user cancels the interaction (e.g., via Ctrl+C or Esc on the first question).
	ErrUserAborted = errors.New("interaction aborted by user")

	// Special value to trigger the custom input prompt.
	customValueOption = "--- Type a custom value ---"
)

type Question struct {
	Title string

	// Flag of type string with default value (if any) and usage description.
	Flag *flag.Flag

	// Valid values.
	Values []string

	// ValuesDescriptions is a list of descriptions for each value. Optional.
	ValuesDescriptions []string

	// CustomValues indicates whether this flag accepts arbitrary custom values
	CustomValues bool

	// ValidateFn is called to validate the value selected by the user.
	// If it returns an error, the error is printed and the user is prompted again.
	ValidateFn func() error
}

const ReservedCustomValue = "#custom"

// Interact with the user to get values for each of the flags.
func Interact(command string, questions []Question) error {
	// Theme: no border around focused option.
	theme := huh.ThemeCharm()
	theme.Focused.Base.Border(lipgloss.HiddenBorder())
	fmt.Println()

	// Header:
	fmt.Println(theme.Focused.Title.Render("Welcome to the Gopjrt installer!"))
	fmt.Println(theme.Focused.Description.Render("This tool will guide you through the installation choices " +
		"for Gopjrt, and will display you the flags you need to use. In the end you can choose to run the command " +
		"according to your selection, or just exit -- you can copy&paste your flags selection to execute later."))
	fmt.Println()

	// Key map: use arrow keys to navigate, enter to confirm, escape to go back.
	keyMap := huh.NewDefaultKeyMap()
	keyMap.Quit = key.NewBinding(key.WithKeys("ctrl+c", "q"))
	keyMap.Select.SetFilter = key.NewBinding(key.WithKeys("ctrl+f"))
	keyMap.Select.ClearFilter = key.NewBinding(key.WithKeys("ctrl+f"))
	keyMap.Select.Submit = key.NewBinding(key.WithKeys("enter", "esc"))

	// displayCommandFn displays the command being edited through the questions.
	firstDisplay := true
	var numLinesHeader int
	displayCommandFn := func() {
		if !firstDisplay {
			// Move the cursor up numLinesHeader lines and erase them
			fmt.Printf("\033[%dA", numLinesHeader)
			// Erases from cursor to end of screen
			fmt.Printf("\033[0J")
		} else {
			firstDisplay = false
		}
		fmt.Println("Selected flags to run:")
		var sb strings.Builder
		sb.WriteString(command)
		for _, question := range questions {
			_, _ = fmt.Fprintf(&sb, " -%s=%s", question.Flag.Name, question.Flag.Value.String())
		}
		coloredCommand := theme.Focused.SelectedOption.Render(sb.String())
		commandLines, err := getRenderedHeight(coloredCommand)
		if err != nil {
			commandLines = 1
			klog.Errorf("Failed to calculate number of lines for terminal: %v", err)
		}
		fmt.Printf("\t%s\n\n", coloredCommand)
		numLinesHeader = 2 + commandLines
	}

	// Loop over questions: notice the user may mover forward and backward, revisiting decisions,
	// so the for does not range over the questions.
	questionIdx := 0
	for questionIdx < len(questions) {
		if questionIdx < 0 {
			// User backed to the beginning
			return ErrUserAborted
		}
		displayCommandFn()
		question := questions[questionIdx]
		options := make([]huh.Option[string], 0, len(question.Values)+1)
		for i, value := range question.Values {
			optionKey := value
			if len(question.ValuesDescriptions) > i {
				optionKey = question.ValuesDescriptions[i]
			}
			options = append(options, huh.NewOption(optionKey, value))
		}
		if question.CustomValues {
			options = append(options, huh.NewOption(lipgloss.NewStyle().Italic(true).Render("…  other …"), ReservedCustomValue))
		}
		value := question.Flag.Value.String()
		selection := huh.NewSelect[string]().
			Title(fmt.Sprintf("(%d of %d) - %s", questionIdx+1, len(questions), question.Title)).
			Description(question.Flag.Usage + "\n").
			Options(options...).
			Value(&value)
		form := huh.NewForm(huh.NewGroup(selection)).
			WithTheme(theme).
			WithKeyMap(keyMap)
		//err := form.Run()
		model := &formModel{form: form}
		prog := tea.NewProgram(model)
		_, err := prog.Run()
		if err != nil {
			return err
		}
		if model.IsEscExit {
			questionIdx--
			continue
		}
		if model.form.State == huh.StateAborted {
			return ErrUserAborted
		}
		if question.CustomValues && value == ReservedCustomValue {
			value = ""
			err = huh.NewInput().
				Title(question.Title).
				Description(question.Flag.Usage).
				Value(&value).
				WithTheme(theme).
				WithKeyMap(keyMap).
				Run()
			if err != nil {
				return err
			}
		}
		err = question.Flag.Value.Set(value)
		if err != nil {
			return errors.Wrapf(err, "failed to set -%s to %q", question.Flag.Name, value)
		}

		// Validate selections:
		if question.ValidateFn != nil {
			var validationErr error
			err := spinner.New().
				Title(fmt.Sprintf("Validating %q ….", question.Flag.Name)).
				Action(func() { validationErr = question.ValidateFn() }).
				Run()
			if err != nil {
				return err
			}
			if validationErr != nil {
				err := huh.NewConfirm().
					Title(question.Title).
					Description(validationErr.Error()).
					Affirmative("Ok").
					Negative("").Run()
				if err != nil {
					return err
				}
				// Don't increment the questionIdx: the question will be asked again.
				continue
			}
		}
		questionIdx++
	}

	// Ask if the user wants to run the command immediately.
	displayCommandFn()
	var confirm bool
	err := huh.NewConfirm().
		Title("Run it immediately ?").
		Affirmative("Yes!").
		Negative("No.").
		Value(&confirm).Run()
	if err != nil {
		return err
	}

	if !confirm {
		return errors.New("Execution not confirmed.")
	}
	return nil
}

// Our custom bubbletea model that wraps the huh.Form
type formModel struct {
	form      *huh.Form
	IsEscExit bool
}

// Init is the first command that is run when the program starts.
func (m *formModel) Init() tea.Cmd {
	return m.form.Init()
}

var escKey = key.NewBinding(key.WithKeys("esc"))

// Update is called when a message is received.
func (m *formModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msgT := msg.(type) {
	// 1. Intercept specific key presses BEFORE they are sent to the form.
	case tea.KeyMsg:
		if key.Matches(msgT, escKey) {
			m.IsEscExit = true
		}
	}

	// Pass all other messages to the form for processing.
	form, cmd := m.form.Update(msg)
	if f, ok := form.(*huh.Form); ok {
		m.form = f
	}

	// If the form is done, we need to quit.
	if m.form.State == huh.StateCompleted || m.form.State == huh.StateAborted {
		return m, tea.Quit
	}
	return m, cmd
}

// View renders the UI.
func (m *formModel) View() string {
	// Delegate the view rendering to the huh.Form.
	return m.form.View()
}

// getRenderedHeight calculates how many terminal lines a string will occupy,
// accounting for both explicit newlines ('\n') and automatic wrapping.
func getRenderedHeight(content string) (int, error) {
	// 1. Get the current terminal width.
	// os.Stdout.Fd() gets the file descriptor for standard output.
	termWidth, _, err := term.GetSize(os.Stdout.Fd())
	if err != nil {
		// Return a sensible default if we can't get the terminal size
		// (e.g., when not running in an interactive terminal).
		return 0, fmt.Errorf("failed to get terminal size: %w", err)
	}

	totalHeight := 0
	// 2. Split the content by explicit newlines first.
	lines := strings.Split(content, "\n")

	for _, line := range lines {
		// 3. Use lipgloss.Width() for a Unicode-aware width calculation.
		lineWidth := lipgloss.Width(line)
		if lineWidth == 0 {
			// An empty line still occupies one line of height.
			totalHeight++
			continue
		}

		// 4. Calculate how many times the line wraps.
		// For example, a 100-char line in an 80-char terminal takes 2 lines.
		// (100 + 80 - 1) / 80 = 199 / 80 = 2
		wrappedLines := (lineWidth + termWidth - 1) / termWidth
		totalHeight += wrappedLines
	}

	return totalHeight, nil
}
