import tkinter as tk
from tkinter import filedialog


class InputElementBase(tk.LabelFrame):
    """
    A base class to create a label frame with an input element.
    """

    def __init__(self, name, parent=None, frame_kwargs=None, verification_func=None, description=None, **kwargs):
        """
        Initializes an instance of InputElement.

        :param name: The name of the input element.
        :type name: str
        :param input_type: The type of input element, such as "int", "float", "string", "boolean", or "list".
        :type input_type: str
        :param parent: The parent widget.
        :type parent: tk.Widget or None
        :param frame_kwargs: Keyword arguments to pass to the tk.LabelFrame.
        :type frame_kwargs: dict or None
        :param verification_func: A function that takes a single argument and returns a boolean indicating whether the
            argument is valid. If a string, it will be interpreted as a function name of a known verification function.
            If None, no verification will be performed.
        :type verification_func: function or str or None
        :param description: A description of the input element. Will be shown as a tooltip when the mouse hovers over
            the input element.
        :type description: str or None
        :param kwargs: additional arguments that will be passed to the appropriate input element initialization method.
        """
        frame_kwargs = frame_kwargs if frame_kwargs is not None else {}
        default_frame_kwargs = dict(text=name, labelanchor='w')
        super().__init__(master=parent, **{**default_frame_kwargs, **frame_kwargs})

        if description is not None:
            create_tooltip(self, description)

        self._input_var = None
        self._input_widgets = None
        self._verification_func = verification_func

        self.build_input_element(**kwargs)

    def build_input_element(self, **kwargs):
        """
        Builds the input element, creating the input variable and input widgets.

        :raises NotImplementedError: If this method is not overridden.
        :rtype: None
        """
        raise NotImplementedError

    def get(self):
        """
        Gets the current value of the input element.

        :return: The current value of the input element.
        :rtype: str or bool or list or int or float
        """
        return self._input_var.get()

    def set(self, value):
        """
        Sets the value of the input element.

        :param value: The value to set the input element to.
        :type value: Any
        :rtype: None
        """
        self._input_var.set(value)

    def verify(self):
        """
        Verifies the current value of the input element.

        :return: Whether the current value of the input element is valid. Defaults to true if no verification function
        is specified.
        :rtype: bool
        """
        if self._verification_func is None:
            return True
        else:
            return self._verification_func(self.get())

    @property
    def input_var(self):
        """
        Gets the tk.Variable associated with the input element.

        :return: The tk.Variable associated with the input element.
        :rtype: tk.Variable
        """
        return self._input_var

    @property
    def input_widgets(self):
        """
        Gets the input widgets associated with the input element.

        :return: The input widgets associated with the input element.
        :rtype: dict[str, tk.Widget]
        """
        return self._input_widgets

    @property
    def verification_func(self):
        """
        Gets the verification function associated with the input element.

        :return: The verification function associated with the input element.
        :rtype: function or None
        """
        return self._verification_func


class NumberInputElementBase(InputElementBase):
    """
    A class to create a label frame with a numeric input element.
    """

    def build_input_element(self, default_value=0, from_=0, to=10000, **kwargs):
        """
        Initializes the input element as a numeric input.
        See https://docs.python.org/3/library/tkinter.ttk.html#ttk.Spinbox for more information.

        :param default_value: The default value for the input. Defaults to 0.
        :type default_value: int or float
        :param from_: The minimum value for the input. Defaults to 0.
        :type from_: int or float
        :param to: The maximum value for the input. Defaults to 10000.
        :type to: int or float
        :param kwargs: Additional keyword arguments to pass to the tk.Spinbox.
        :type kwargs: dict
        :rtype: None
        """

        self._input_var = self._get_input_var(default_value)
        self._input_widgets = {"SpinBox": tk.Spinbox(self, from_=from_, to=to, textvariable=self._input_var, **kwargs)}
        self._input_widgets["SpinBox"].pack(fill='x', expand=1)

    @staticmethod
    def _get_input_var(default_value):
        """
        Gets the appropriate tk.Variable for the input element.

        :param default_value: The default value for the input.
        :type default_value: int or float, depending on the actual implementation
        :return: A tk.IntVar or tk.DoubleVar, depending on the actual implementation.
        :rtype: tk.Variable
        :raises NotImplementedError: If this method is not overridden.
        """

        raise NotImplementedError


class IntInputElement(NumberInputElementBase):
    @staticmethod
    def _get_input_var(default_value):
        """
        See NumberInputElementBase._get_input_var.
        """
        return tk.IntVar(value=default_value)


class FloatInputElement(NumberInputElementBase):
    @staticmethod
    def _get_input_var(default_value):
        """
        See NumberInputElementBase._get_input_var.
        """
        return tk.DoubleVar(value=default_value)


class StringInputElement(InputElementBase):
    """
    A class to create a label frame with a string input element.
    """

    def build_input_element(self, default_value=None, **kwargs):
        """
        Initializes the input element as a string input.

        :param default_value: The default value for the input. Defaults to None.
        :type default_value: str or None
        :param kwargs: Additional keyword arguments to pass to the tk.Entry.
        :type kwargs: dict
        :rtype: None
        """
        self._input_var = tk.StringVar(value=default_value if default_value is not None else "")
        self._input_widgets = {"Entry": tk.Entry(self, textvariable=self._input_var, **kwargs)}
        self._input_widgets["Entry"].pack(fill='x', expand=1)


class BoolInputElement(InputElementBase):
    """
    A class to create a label frame with a boolean input element.
    """

    def build_input_element(self, default_value=False, **kwargs):
        """
        Initializes the input element as a boolean input.

        :param default_value: The default value for the input. Defaults to False.
        :type default_value: bool
        :param kwargs: Additional keyword arguments to pass to the two tk.Radiobutton.
        :type kwargs: dict
        :rtype: None
        """
        self._input_var = tk.BooleanVar(value=default_value)
        self._input_widgets = {
            "True": tk.Radiobutton(self, text='True', variable=self._input_var, value=True, **kwargs),
            "False": tk.Radiobutton(self, text='False', variable=self._input_var, value=False, **kwargs)
        }
        self._input_widgets["True"].pack(fill='x', expand=1)
        self._input_widgets["False"].pack(fill='x', expand=1)


class ListInputElement(InputElementBase):
    """
    A class to create a label frame with a list input element.
    """

    def build_input_element(self, options, default_index=None, multi_select=True, height=2, **kwargs):
        """
        Initializes the input element as a list input.

        :param options: A list of options to display to the user.
        :type options: list of str or tuple of str
        :param default_index: The index of the option to select by default. Defaults to None.
        :type default_index: int or list of int or tuple of int or None
        :param multi_select: Whether to allow multiple selections. Defaults to True.
        :type multi_select: bool
        :param height: The height of the listbox. Defaults to 2.
        :type height: int
        :param kwargs: Additional keyword arguments to pass to the tk.Listbox.
        :type kwargs: dict
        :rtype: None
        """
        if default_index is None:
            default_value = None if not multi_select else []
        elif isinstance(default_index, (list, tuple)):
            if multi_select:
                default_value = [options[i] for i in default_index]
            else:
                raise ValueError("Cannot set multiple default indices for a non-multi-select list")
        elif isinstance(default_index, int):
            default_value = options[default_index]
        else:
            raise ValueError(f"Invalid default index type {type(default_index)}")

        self._input_var = tk.Variable(value=default_value)

        if multi_select:
            select_mode = tk.MULTIPLE
        else:
            select_mode = tk.BROWSE

        self._input_widgets = {"Listbox": tk.Listbox(self, listvariable=tk.Variable(value=options), height=height,
                                                     selectmode=select_mode, **kwargs)}
        self._input_widgets["Listbox"].pack(fill='x', expand=1)

        if default_index is not None:
            if isinstance(default_index, (list, tuple)):
                for i in default_index:
                    self._input_widgets["Listbox"].selection_set(i)
            else:
                self._input_widgets["Listbox"].selection_set(default_index)

        def set_input_var(*_):
            currently_selected = [options[i] for i in self._input_widgets["Listbox"].curselection()]
            self._input_var.set(currently_selected if multi_select else currently_selected[0])
        self._input_widgets["Listbox"].bind("<<ListboxSelect>>", set_input_var)

        self._input_widgets["Scrollbar"] = tk.Scrollbar(self._input_widgets["Listbox"], orient="vertical")
        self._input_widgets["Scrollbar"].config(command=self._input_widgets["Listbox"].yview)
        self._input_widgets["Scrollbar"].pack(side="right", fill="y")


class PathInputElementBase(InputElementBase):
    """
    A class to create a label frame with a path input element.
    """

    def build_input_element(self, default_path=None, path_type="single_file", **kwargs):
        """
        Initializes the input element as a path selection input.

        :param default_path: The default path to display to the user. Defaults to None.
        :type default_path: str or None
        :param path_type: The type of path to select, either "single_file", "multiple_files", or "directory".
        :type path_type: str
        :param kwargs: Additional keyword arguments to pass to the tk.Button.
        :type kwargs: dict
        :rtype: None
        """
        self._input_var = tk.Variable(value=default_path if default_path is not None else "")

        def path_select_dialog():
            path = self._browse(**kwargs)
            self._input_var.set(path)

        self._input_widgets = {
            "Entry": tk.Entry(self, textvariable=self._input_var),
            "Button": tk.Button(self, text="Search", command=path_select_dialog)
        }
        self._input_widgets["Entry"].pack(fill='x', expand=1, side='left')
        self._input_widgets["Button"].pack(side='right')

    @staticmethod
    def _browse(**kwargs):
        """
        Opens a file dialog to select a path or multiple paths.

        :return: The selected path or list of paths, depending on the final implementation.
        :rtype: str or list of str
        :raises NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError("This method must be implemented by a subclass")


class SingleFileInputElement(PathInputElementBase):
    """
    A class to create a label frame with a single file path input element.
    See PathInputElementBase for additional information.
    """
    def _browse(self, **kwargs):
        """
        Opens a file dialog to select a file path. See the documentation for tkinter.filedialog.askopenfilename for
        additional keyword arguments. See PathInputElementBase._browse for additional information.
        """
        return filedialog.askopenfilename(**kwargs)


class MultipleFilesInputElement(PathInputElementBase):
    """
    A class to create a label frame with a multiple file path input element.
    See PathInputElementBase for additional information.
    """
    def _browse(self, **kwargs):
        """
        Opens a file dialog to select multiple file paths. See the documentation for tkinter.filedialog.askopenfilenames
        for additional keyword arguments. See PathInputElementBase._browse for additional information.
        """
        return filedialog.askopenfilenames(**kwargs)


class SaveFileInputElement(PathInputElementBase):
    """
    A class to create a label frame with a save file path input element.
    See PathInputElementBase for additional information.
    """
    def _browse(self, **kwargs):
        """
        Opens a file dialog to select a file path to save to. See the documentation for
        tkinter.filedialog.asksaveasfilename for additional keyword arguments. See PathInputElementBase._browse for
        additional information.
        """
        return filedialog.asksaveasfilename(**kwargs)


class DirectoryInputElement(PathInputElementBase):
    """
    A class to create a label frame with a directory path input element.
    See PathInputElementBase for additional information.
    """
    def _browse(self, **kwargs):
        """
        Opens a file dialog to select a directory path. See the documentation for tkinter.filedialog.askdirectory for
        additional keyword arguments. See PathInputElementBase._browse for additional information.
        """
        return filedialog.askdirectory(**kwargs)


class ToolTip(object):
    """
    A Tkinter tooltip class.
    Based on http://www.voidspace.org.uk/python/weblog/arch_d7_2006_07_01.shtml#e387, found on the Wayback Machine
    https://web.archive.org/web/20210226190130/http://www.voidspace.org.uk:80/python/weblog/arch_d7_2006_07_01.shtml
    """
    def __init__(self, widget):
        self.widget = widget
        self.text = ""
        self.tip_window = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        self.text = text
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


def create_tooltip(widget, text):
    """
    Creates a tooltip for a given widget.

    :param widget: The widget to create a tooltip for.
    :type widget: tk.Widget
    :param text: The text to display in the tooltip.
    :type text: str
    :rtype: None
    """
    tool_tip = ToolTip(widget)

    def enter(_):
        tool_tip.showtip(text)

    def leave(_):
        tool_tip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


class SyncedMultiListbox(tk.Frame):
    def __init__(self, parent, n_listboxes=2, listbox_kwargs=None, scrollbar_kwargs=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # the listboxes
        self._listboxes = []
        listbox_kwargs = listbox_kwargs if listbox_kwargs is not None else {}
        for i in range(n_listboxes):
            self._listboxes.append(self._build_listbox(**listbox_kwargs))

        # the shared scrollbar
        scrollbar_kwargs = scrollbar_kwargs if scrollbar_kwargs is not None else {}
        self._scrollbar = self._build_scrollbar(**scrollbar_kwargs)
        self.grid_rowconfigure(0, weight=1)

    def _build_listbox(self, **kwargs):
        default_lb_kwargs = dict(yscrollcommand=self._listbox_callback_on_yscroll, exportselection=False)
        listbox_init_kwargs = {**default_lb_kwargs, **kwargs}

        listbox_index = len(self._listboxes)

        # exportselection must be disabled, as otherwise there will be a callback loop.
        # No matter if you unbind the callback or not.
        new_listbox = tk.Listbox(self, **listbox_init_kwargs)
        new_listbox.bind('<<ListboxSelect>>', self._listbox_callback_on_select)
        new_listbox.grid(row=0, column=listbox_index, sticky='nsew')
        self.grid_columnconfigure(listbox_index, weight=1)
        return new_listbox

    def _build_scrollbar(self, **kwargs):
        default_sb_kwargs = dict(orient="vertical", command=self._scrollbar_callback_yview)
        scrollbar_init_kwargs = {**default_sb_kwargs, **kwargs}

        scrollbar = tk.Scrollbar(self, **scrollbar_init_kwargs)
        column, row = self.grid_size()
        scrollbar.grid(row=0, column=column+1, sticky='ns')
        self.grid_columnconfigure(column+1, weight=0)
        return scrollbar

    def _listbox_callback_on_select(self, event):
        callback_selection = event.widget.curselection()

        if len(callback_selection) == 0:
            return

        self.selection_clear(0, tk.END)
        self.selection_set(callback_selection)
        for listbox in self._listboxes:
            # Set the selection
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(callback_selection)

    def _listbox_callback_on_yscroll(self, *args):
        for listbox in self._listboxes:
            listbox.yview_moveto(args[0])
        self._scrollbar.set(*args)

    def _scrollbar_callback_yview(self, *args):
        for lb in self._listboxes:
            lb.yview(*args)

    def curselection(self):
        return [listbox.curselection() for listbox in self._listboxes]

    def selection_clear(self, first, last=None):
        for lb in self._listboxes:
            lb.selection_clear(first, last)

    def selection_set(self, first, last=None):
        for lb in self._listboxes:
            lb.selection_set(first, last)

    def insert(self, index, elements):
        assert len(elements) == len(self._listboxes), "Number of elements must match number of listboxes"
        for lb, element in zip(self._listboxes, elements):
            lb.insert(index, element)

    def delete(self, first, last=None):
        for lb in self._listboxes:
            lb.delete(first, last)

    def get(self, first, last=None):
        return [lb.get(first, last) for lb in self._listboxes]


class SyncedNamedListboxes(SyncedMultiListbox):
    def __init__(self, parent, names, name_tooltip_dict=None, *args, **kwargs):
        self._names = names
        self._named_listboxes = {}
        self._name_tooltip_dict = name_tooltip_dict if name_tooltip_dict is not None else {}
        super().__init__(parent, n_listboxes=len(names), *args, **kwargs)

    def _build_listbox(self, **kwargs):
        default_lb_kwargs = dict(yscrollcommand=self._listbox_callback_on_yscroll, exportselection=False)
        listbox_init_kwargs = {**default_lb_kwargs, **kwargs}

        listbox_index = len(self._listboxes)
        listbox_name = self._names[listbox_index]
        new_labelframe = tk.LabelFrame(self, text=listbox_name, labelanchor="n")

        # exportselection must be disabled, as otherwise there will be a callback loop.
        # No matter if you unbind the callback or not.
        new_listbox = tk.Listbox(new_labelframe, **listbox_init_kwargs)
        new_listbox.bind('<<ListboxSelect>>', self._listbox_callback_on_select)
        new_listbox.pack(fill='both', expand=True)
        self._named_listboxes[listbox_name] = new_listbox

        new_labelframe.grid(row=0, column=listbox_index, sticky='nsew')
        self.grid_columnconfigure(listbox_index, weight=1)

        create_tooltip(new_labelframe, self._name_tooltip_dict.get(listbox_name, ""))

        return new_listbox

    def curselection(self):
        return {name: listbox.curselection() for name, listbox in self._named_listboxes.items()}

    def selection_clear(self, first, last=None):
        for lb in self._listboxes:
            lb.selection_clear(first, last)

    def selection_set(self, first, last=None):
        for lb in self._listboxes:
            lb.selection_set(first, last)

    def insert(self, index, elements_dict):
        assert len(elements_dict) == len(self._listboxes), "Number of elements must match number of listboxes"
        assert set(elements_dict.keys()) == set(self._names), "Keys of elements must match names of listboxes"
        for lb_name, lb in self._named_listboxes.items():
            lb.insert(index, elements_dict[lb_name])

    def get(self, first, last=None):
        return {name: lb.get(first, last) for name, lb in self._named_listboxes.items()}


class TableModificator(SyncedNamedListboxes):
    button_add_text = "Add Row"
    button_remove_text = "Remove Row"

    def __init__(self, parent, column_names, column_tooltip_dict=None, *args, **kwargs):
        super().__init__(parent, column_names, column_tooltip_dict, *args, **kwargs)
        self._named_entries = {}
        self._create_entry_fields()
        self._create_buttons()

    def _listbox_callback_on_select(self, event):
        super()._listbox_callback_on_select(event)
        self._update_entry_fields()

    def _create_entry_fields(self):
        for i, name in enumerate(self._names):
            entry_field = tk.Entry(self)
            entry_field.grid(row=1, column=i, sticky='nsew')
            self._named_entries[name] = entry_field

    def _update_entry_fields(self):
        selected_values = self.get(self.curselection()[self._names[0]])
        for name, value in selected_values.items():
            self._named_entries[name].delete(0, tk.END)
            self._named_entries[name].insert(0, value)

    def _create_buttons(self):
        button_frame = tk.Frame(self)

        add_row_button = tk.Button(button_frame, text=self.button_add_text, command=self._add_row)
        add_row_button.pack(side="left", fill="both", expand=True)

        delete_row_button = tk.Button(button_frame, text=self.button_remove_text, command=self._delete_row)
        delete_row_button.pack(side="left", fill="both", expand=True)

        button_frame.grid(row=2, column=0, columnspan=len(self._names), sticky='ew')

    def _add_row(self):
        entry_values = {name: entry.get() for name, entry in self._named_entries.items()}
        self.insert(tk.END, entry_values)

    def _delete_row(self):
        selected_values = self.curselection()
        if not any(selected_values.values()):
            return

        index_to_delete = next(iter(selected_values.values()))[0]
        self.delete(index_to_delete)


class DictModifier(TableModificator):
    button_add_text = "Add / Update Entry"
    button_remove_text = "Remove Entry"

    def __init__(self, parent, initial_dict=None, key_name="Key", value_name="Value",
                 key_tooltip=None, value_tooltip=None, *args, **kwargs):
        self._key_name, self._value_name = key_name, value_name
        column_tooltip_dict = {}
        if key_tooltip is not None:
            column_tooltip_dict[key_name] = key_tooltip
        if value_tooltip is not None:
            column_tooltip_dict[value_name] = value_tooltip
        super().__init__(parent, [self._key_name, self._value_name],
                         column_tooltip_dict=column_tooltip_dict, *args, **kwargs)

        initial_dict = initial_dict if initial_dict is not None else {}
        for key, value in initial_dict.items():
            self.insert(tk.END, {key_name: key, value_name: value})

        self._dict = initial_dict

    def _add_row(self):
        entry_values = {name: entry.get() for name, entry in self._named_entries.items()}

        if entry_values[self._key_name] in self.get(0, tk.END)[self._key_name]:
            row_index = self.get(0, tk.END)[self._key_name].index(entry_values[self._key_name])
            self.delete(row_index)
        else:
            row_index = tk.END

        self.insert(row_index, entry_values)
        self._update_dict()

    def _delete_row(self):
        super()._delete_row()
        self._update_dict()

    def _update_dict(self):
        self._dict = {key: value for key, value in zip(self.get(0, tk.END)[self._key_name],
                                                       self.get(0, tk.END)[self._value_name])}

    @property
    def dict(self):
        return self._dict
