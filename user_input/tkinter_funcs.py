import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

from .tkinter_classes import (IntInputElement, FloatInputElement, StringInputElement, BoolInputElement,
                              ListInputElement, SingleFileInputElement, MultipleFilesInputElement, SaveFileInputElement,
                              DirectoryInputElement, DictModifier)


def generate_function_launcher_window(func_dict, title="Function Launcher", desc_dict=None):
    """
    Generates a Tkinter window with a dropdown menu and a launch button.
    The dropdown menu contains the keys of the func_dict, and the launch button launches the function
    corresponding to the selected key.

    :param func_dict: A dictionary of functions to be launched. The keys are the names of the functions.
    :type func_dict: dict
    :param title: The title of the window. Defaults to "Function Launcher".
    :type title: str
    :param desc_dict: An optional dictionary of descriptions for each function. The keys are the names of the functions.
    :type desc_dict: dict or None
    :rtype: None
    """

    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)

    # get max length of keys for width of dropdown
    max_len = max([len(k) for k in func_dict.keys()])

    # create dropdown selector
    dropdown_label = tk.Label(root, text="Select a function:")
    dropdown_label.grid(row=0, column=0)

    selected_func = tk.StringVar(root)
    selected_func.set(list(func_dict.keys())[0])

    dropdown = tk.OptionMenu(root, selected_func, *func_dict.keys())
    dropdown.config(width=max_len)
    dropdown.grid(row=0, column=1)

    def on_dropdown_change(*_):
        if desc_dict is None:
            return

        selected_option = selected_func.get()
        if selected_option in desc_dict:
            desc_label.configure(text=desc_dict[selected_option])
            desc_label.grid(row=1, column=0, columnspan=3, sticky="w")
        else:
            desc_label.grid_remove()

    selected_func.trace("w", on_dropdown_change)

    def launch_func(func):
        text_output.configure(text=f"Running {func.__name__}...")
        root.update()
        func()
        text_output.configure(text="Ready")

    # create launch button
    launch_button = tk.Button(root, text="Launch",
                              command=lambda: launch_func(func_dict[selected_func.get()]))
    launch_button.config(width=6)
    launch_button.grid(row=0, column=2)

    root.update()  # Allow the window's size to be calculated,

    # create text output
    text_output = tk.Label(root, justify="left", text="Ready")
    text_output.grid(row=2, column=0, columnspan=3, sticky="w")
    text_output.configure(wraplength=root.winfo_width() * 0.95)

    # create description label
    if desc_dict is not None:
        desc_label = tk.Label(root, justify="left", wraplength=root.winfo_width())
        on_dropdown_change()

    root.mainloop()


def select_file_dialog(multiple_files=True, **options):
    """

    :param multiple_files:
    :type multiple_files:
    :param options:
    :type options:
    :return:
    :rtype:
    """
    tk.Tk().withdraw()
    if multiple_files:
        file_path = filedialog.askopenfilenames(**options)
    else:
        file_path = filedialog.askopenfilename(**options)
    return file_path


def select_dir_dialog(**options):
    """

    :param options:
    :return:
    :rtype: str
    """
    tk.Tk().withdraw()
    dir_path = filedialog.askdirectory(**options)
    return dir_path


def ask_multiple_choice_question(prompt, options, title="Select an option", initial_value=0):
    """
    Display a dialog box to allow the user to select one option from a list.

    :param prompt: A prompt to display to the user.
    :type prompt: str
    :param options: A list of options to display to the user.
    :type options: list of str
    :param title: The title of the dialog box.
    :type title: str
    :param initial_value: The index of the option to select by default.
    :type initial_value: int
    :return: The selected option.
    :rtype: str
    """
    root = tk.Tk()
    root.title(title)

    v = tk.IntVar(root, initial_value)
    if prompt:
        tk.Label(root, text=prompt, padx=5).pack()
    for i, option in enumerate(options):
        tk.Radiobutton(root, text=option, variable=v, value=i, padx=5).pack(anchor="w")

    submit_button = tk.Button(root, text="Submit", command=root.quit)
    submit_button.pack()

    root.mainloop()
    root.destroy()

    return options[v.get()]


type_input_element_map = {
    "str": StringInputElement,
    "int": IntInputElement,
    "float": FloatInputElement,
    "bool": BoolInputElement,
    "list": ListInputElement,
    "single_file": SingleFileInputElement,
    "multiple_files": MultipleFilesInputElement,
    "save_file": SaveFileInputElement,
    "directory": DirectoryInputElement
}


def create_input_element(name, input_type, parent=None, **kwargs):
    """
    Creates an input element of the specified type.

    :param name: The name of the input element.
    :type name: str
    :param input_type: The type of input element to create.
    :type input_type: str
    :param parent: The parent of the input element. Defaults to None.
    :type parent: tk.Widget or None
    :param kwargs: Additional keyword arguments to pass to the input element initialization method.
    :type kwargs: dict
    :return: The created input element.
    :rtype: utility.tkinter_classes.InputElementBase
    """
    if input_type not in type_input_element_map:
        raise ValueError(f"Invalid input type {input_type}")
    return type_input_element_map[input_type](name, parent=parent, **kwargs)


def create_input_frame(element_type_dict, element_kwargs=None, **kwargs):
    """
    Creates a tk.LabelFrame containing InputElements.

    :param element_type_dict: A dictionary mapping input element names to input types.
    :type element_type_dict: dict of str to str
    :param element_kwargs: A dictionary mapping input element names to dictionaries of keyword arguments to pass to the
        appropriate input element initialization method.
    :type element_kwargs: dict[str, dict] or None
    :return: The created tk.Frame and a dictionary mapping input element names to input elements.
    :rtype: (tk.LabelFrame, dict[str, InputElement])
    """
    element_kwargs = element_kwargs if element_kwargs is not None else {}
    label_frame = tk.LabelFrame(**kwargs)
    input_elements = {}
    for name, input_type in element_type_dict.items():
        input_elements[name] = create_input_element(name, input_type, parent=label_frame,
                                                    **element_kwargs.get(name, {}))
        input_elements[name].pack(fill="x")
    return label_frame, input_elements


def create_input_mask(required=None, optional=None, element_kwargs=None):
    """
    Creates a tk window containing two tk.LabelFrames with InputElements and a Submit button.

    :param required: A dictionary mapping required input element names to input types.
    :type required: dict of str to str
    :param optional: A dictionary mapping optional input element names to input types.
    :type optional: dict of str to str
    :param element_kwargs: A dictionary mapping input element names to dictionaries of keyword arguments to pass to the
        appropriate input element initialization method.
    :type element_kwargs: dict[str, dict] or None
    :return: A dictionary mapping input element names to input element values on successful submission.
    :rtype: dict of str to any
    """
    # Check if at least one input dictionary is given
    if required is None and optional is None:
        raise ValueError("At least one of Required or Optional input element dictionary should be provided")

    # Create a top-level window for the input mask
    mask_window = tk.Tk()

    # Create the Required LabelFrame
    if required is not None:
        required_frame, required_inputs = create_input_frame(required, element_kwargs=element_kwargs,
                                                             master=mask_window, text="Required", fg="red")
        required_frame.pack(side="top", fill="x", padx=5, pady=(5, 0))

    # Create the Optional LabelFrame
    if optional is not None:
        optional_frame, optional_inputs = create_input_frame(optional, element_kwargs=element_kwargs,
                                                             master=mask_window, text="Optional")
        optional_frame.pack(side="top", fill="x", padx=5, pady=(5, 0))

    # Create the Submit button
    def submit():
        # Check if all required inputs are filled
        if required is not None:
            required_filled = all(required_inputs[name].get() != '' for name in required_inputs)
            if not required_filled:
                messagebox.showerror("Unfilled required fields", "Please fill in all required elements.")
                return

        mask_window.quit()

    submit_button = tk.Button(mask_window, text="Submit", command=submit)
    submit_button.pack(side="top", fill="x", padx=5, pady=5)

    mask_window.mainloop()

    # Get the values of all inputs and return as a dictionary
    input_values = {}
    if required is not None:
        input_values.update({name: required_inputs[name].get() for name in required_inputs})
    if optional is not None:
        input_values.update({name: optional_inputs[name].get() for name in optional_inputs})

    mask_window.destroy()

    # Return a dictionary of input values on successful submission
    return input_values


def create_dict_modifier(input_dict, key_name="Key", value_name="Value", key_tooltip=None, value_tooltip=None,
                         **kwargs):
    """
    Creates a tk window containing a DictModificator.

    :param input_dict: The dictionary to modify.
    :type input_dict: dict
    :param key_name: The name of the key input element.
    :type key_name: str
    :param value_name: The name of the value input element.
    :type value_name: str
    :param key_tooltip: The tooltip of the key input element.
    :type key_tooltip: str or None
    :param value_tooltip: The tooltip of the value input element.
    :type value_tooltip: str or None
    :param kwargs: Additional keyword arguments to pass to the DictModificator initialization method.
    :type kwargs: dict
    :return: The modified dictionary.
    :rtype: dict
    """

    window = tk.Tk()
    window.title("Modify Dictionary")
    dict_modifier = DictModifier(window, initial_dict=input_dict, key_name=key_name, value_name=value_name,
                                 key_tooltip=key_tooltip, value_tooltip=value_tooltip, **kwargs)
    dict_modifier.pack(fill=tk.BOTH, expand=True)
    window.mainloop()

    return dict_modifier.dict
