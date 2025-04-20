from pyparsing import Token, ParseException


# following from pyparsing.wikispaces.com Examples page
class CloseMatch(Token):
    # https://stackoverflow.com/a/19776529
    """A special subclass of Token that does *close* matches. For each
       close match of the given string, a tuple is returned giving the
       found close match, and a list of mismatch positions."""

    def __init__(self, seq, maxMismatches=1):
        super(CloseMatch, self).__init__()
        self.sequence = seq
        self.maxMismatches = maxMismatches
        if isinstance(self.sequence, str):
            self.errmsg = "Expected " + self.sequence
        else:
            self.errmsg = "Expected " + repr(self.sequence)
        self.mayIndexError = False
        self.mayReturnEmpty = False

    def parseImpl(self, instring, loc, doActions=True):
        start = loc
        instrlen = len(instring)
        maxloc = start + len(self.sequence)

        if maxloc <= instrlen:
            seq = self.sequence
            seqloc = 0
            mismatches = []
            throwException = False
            done = False
            while loc < maxloc and not done:
                if instring[loc] != seq[seqloc]:
                    mismatches.append(seqloc)
                    if len(mismatches) > self.maxMismatches:
                        throwException = True
                        done = True
                loc += 1
                seqloc += 1
        else:
            throwException = True

        if throwException:
            # ~ exc = self.myException
            # ~ exc.loc = loc
            # ~ exc.pstr = instring
            # ~ raise exc
            raise ParseException(instring, loc, self.errmsg)

        return loc, (instring[start:loc], mismatches)


def int_sequence_to_string_synonym(input_tuple, zero_char_offset=97):
    """
    Convert a tuple of integers to a string synonym. This is done by mapping each integer to a character using the
    zero_char_offset as the starting point for the character mapping. The default zero_char_offset is 97, which maps
    0 to 'a', 1 to 'b', and so on. If your int sequence contains negative numbers greater than the negative of the
    zero_char_offset, you will get a ValueError, so adjust the zero_char_offset accordingly.
    The maximum span of available characters is 0 to 1114111, which is the maximum unicode code point.

    :param input_tuple: A tuple of integers
    :type input_tuple: tuple of int
    :param zero_char_offset: The offset for the character mapping. The default is 97, which maps 0 to 'a'
    :type zero_char_offset: int
    :return: A string synonym of the input tuple
    :rtype: str
    """
    return "".join(map(lambda x: chr(x + zero_char_offset), input_tuple))


def string_synonym_to_int_sequence(input_string, zero_char_offset=97):
    """
    Convert a string synonym to a tuple of integers. The inverse operation of int_sequence_to_string_synonym. See the
    documentation for int_sequence_to_string_synonym for more information.

    :param input_string: A string synonym
    :type input_string: str
    :param zero_char_offset: The offset for the character mapping. The default is 97, which maps 'a' to 0
    :type zero_char_offset: int
    :return: A tuple of integers
    :rtype: tuple of int
    """
    return tuple(map(lambda x: ord(x) - zero_char_offset, input_string))
