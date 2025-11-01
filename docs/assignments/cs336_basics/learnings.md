# Learnings

My learnings from the assignment.

## High-level learnings

## Assignment Questions

### Understanding unicode

<!-- prettier-ignore-start -->
??? question "What Unicode character does `chr(0)` return?"
    `'\x00'`

??? question "How does this character’s string representation (`repr()`) differ from its printed representation?"
    The printed representation is non-visible, whereas its string representation is `'\x00'`.

??? question "What happens when this character occurs in text?"
    When added to a string, its string representation is stored as is. When printed, this `NULL` character is non-visible.
<!-- prettier-ignore-end -->
