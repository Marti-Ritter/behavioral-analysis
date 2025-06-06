import types


class GeneratorWithOptionalLength(object):
    """
    Adapted from https://stackoverflow.com/a/6416585.
    """
    def __init__(self, generator_source, length=None):
        if not isinstance(generator_source, types.GeneratorType):
            if hasattr(generator_source, "__len__") and length is None:
                length = len(generator_source)
            generator_source = (i for i in generator_source)

        self.generator_source = generator_source
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.generator_source


class GeneratorWithAdditionalProperties(object):
    """
    A small wrapper to add further properties to an existing generator object.

    """
    def __init__(self, generator_source, **kwargs):
        self.generator_source = generator_source
        self.__dict__.update(kwargs)

    def __iter__(self):
        return self.generator_source

    def __next__(self):
        return next(self.generator_source)



class GeneratorWrapperNoEdits(object):
    """
    Just an example to build a class that has an internal generator and can be used as such, but with possible
    additional properties added based on unchanging generator properties, e.g. the length here.
    """
    def __init__(self, start=0, end=10):
        self._internal_generator = self._internal_generator_func(start, end)
        self._length = end - start

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._internal_generator)

    def _internal_generator_func(self, start, end):
        for i in range(start, end):
            yield i

    def __len__(self):
        return self._length


class GeneratorWrapperWithEdits(object):
    """
    A slightly more complicated implementation that allows access to the internal generators index, so that one can edit
    this during the loop or between iterations.
    """
    def __init__(self, start=0, end=10):
        self._range = range(start, end)
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index > len(self._range) - 1:
            raise StopIteration
        return_value = self._range[self.index]
        self.index += 1
        return return_value

    def __len__(self):
        return len(self._range)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        if value > len(self._range) or value < 0:
            raise ValueError("Index out of range")
        self._index = value


class SlicableGenerator(object):
    """
    A generator wrapper that can be sliced. The indices will be based on order in the generator. Optionally the input
    generator can also return its own index along with an element. The next() function of this generator runs
    independently of the slicing and getting of elements. Elements are only evaluated when needed, that is either when
    the generator is iterated over or when the element is requested. After an element is evaluated it is stored in a
    dictionary with the key being the index of the element. This allows for fast access to elements without having to
    evaluate them again. But this also increases memory requirements. This might be useful during development.

    The main use case might be here some iterator that produces large output, but you don't want to evaluate all of it
    for slicing, e.g. a video frame reader, because you are only interested in a small subset or single value at the
    beginning for evaluation purposes. The approach shown here limits the memory requirements to the minimum in that
    case, but is limited by the nature of generators (depletable and unpredictable).

    Frankly, the motivating question here was more whether it was possible, rather than whether it is needed...
    The answer is yes, but it requires loads of code. I'm not sure whether it is worth the effort.
    """

    def __init__(self, source_generator, returns_index=False):
        self.evaluated_generator_output = {}
        self.wrapped_generator = source_generator
        self.returns_index = returns_index
        self._current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        eval_gen_out = self.evaluated_generator_output
        wrapped_gen = self.wrapped_generator

        if self._current_index < len(eval_gen_out):
            existing_indices = list(eval_gen_out.keys())
            index_to_yield = existing_indices[self._current_index]
            element_to_yield = eval_gen_out[index_to_yield]
            self._current_index += 1
            return (index_to_yield, element_to_yield) if self.returns_index else element_to_yield

        if self.returns_index:
            index_to_yield, element_to_yield = next(wrapped_gen)
        else:
            index_to_yield = self._current_index
            element_to_yield = next(wrapped_gen)
        eval_gen_out[index_to_yield] = element_to_yield
        self._current_index += 1
        return (index_to_yield, element_to_yield) if self.returns_index else element_to_yield

    def _evaluate_generator(self, stop_index):
        eval_gen_out = self.evaluated_generator_output
        wrapped_gen = self.wrapped_generator

        generator_iter = wrapped_gen if self.returns_index else enumerate(wrapped_gen, start=len(eval_gen_out))
        for element_index, element in generator_iter:
            eval_gen_out[element_index] = element
            if element_index == stop_index:
                return True
        return False

    def get_item_by_index(self, index):
        eval_gen_out = self.evaluated_generator_output

        if index in eval_gen_out:
            return (index, eval_gen_out[index]) if self.returns_index else eval_gen_out[index]

        found_index = self._evaluate_generator(index)
        if not found_index:
            raise IndexError("Index out of range")
        return (index, eval_gen_out[index]) if self.returns_index else eval_gen_out[index]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get_item_by_index(item)
        elif isinstance(item, slice):
            start = item.start or 0
            stop = item.stop or len(self.evaluated_generator_output)
            step = item.step or 1

            stop_within_range = self._evaluate_generator(stop)
            if not stop_within_range:
                stop = len(self.evaluated_generator_output)
            return [self.get_item_by_index(i) for i in range(start, stop, step)]
        else:
            raise TypeError(f"Cannot index generator with {type(item)}")