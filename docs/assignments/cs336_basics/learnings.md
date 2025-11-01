# Learnings

My learnings from the assignment.

## High-level learnings

## Assignment Questions

### Unicode

<!-- prettier-ignore-start -->
??? question "What Unicode character does `chr(0)` return?"
    `'\x00'`

??? question "How does this character’s string representation (`repr()`) differ from its printed representation?"
    The printed representation is non-visible, whereas its string representation is `'\x00'`.

??? question "What happens when this character occurs in text?"
    When added to a string, its string representation is stored as is. When printed, this `NULL` character is non-visible.

??? question "What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32?"
    - The longer sequence of byte values (resp. 2x and 4x) lead to worse compression ratio.
    - All default ascii characters, which are likely the most common characters in standard text, can be represented using a single byte.

??? question "Why is the decoding function incorrect?"
    This function decodes the bytestring byte-by-byte. It will thus fail for characters that are encoded on multiple bytes.
    An example input bytestring that yields incorrect results is the encoding for é `b'\xc3\xa9'`

??? question "Give a two byte sequence that does not decode to any Unicode character(s)."
    `b'\xc3\x41'` is such a sequence. While the first byte is valid, the second byte is invalid as its binary form highest order bits don't match `10`.

<!-- prettier-ignore-end -->
