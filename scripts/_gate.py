"""The four primitives every `check_*.py` gate shares.

Extracted when Module E made this the twelfth gate script and each one was
still carrying its own copy of `FAILURES`, `rule`, `verdict`, `indent`, the
summary block and the exit-code convention.

**Only the primitives move here.** Nothing question-specific, nothing that
knows about a module — each gate stays readable top to bottom, which is the
property that made eleven copies tolerable in the first place.

The exit-code convention, which callers and CI depend on:

    0   every check passed
    1   at least one check failed
    2   setup error — a missing key, an unreachable database, a bad argument.
        Deliberately distinct, because "the thing under test is broken" and
        "you have not finished configuring your machine" want different
        reactions from whoever ran it.

A check is one `verdict(bool, label)` call, and the label is a full sentence
describing what held — on failure it is what the summary prints back, so it
has to stand alone without the code around it.
"""

FAILURES: list[str] = []


def rule(title: str) -> None:
    """A section heading."""
    print(f"\n{'━' * 78}\n{title}\n{'━' * 78}")


def verdict(ok: bool, label: str) -> bool:
    """Record one check. Returns its own result, so it can gate follow-on output."""
    print(f"  [{'ok ' if ok else 'BAD'}] {label}")
    if not ok:
        FAILURES.append(label)
    return ok


def note(label: str) -> None:
    """A reading that is worth printing but is not a pass/fail claim.

    The `[-- ]` marker, for things a gate observes and deliberately does not
    judge — an inert cache breakpoint on a model whose floor the prefix does
    not clear, a cap the run never reached. Reported by name so it cannot be
    mistaken for either a tick or a silence.
    """
    print(f"  [-- ] {label}")


def indent(text: str, prefix: str = "    │ ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def summary(on_failure: str = "", on_success: str = "") -> None:
    """The closing block. `on_failure` should point at the layer that owns the fix."""
    rule("summary")
    if FAILURES:
        print(f"  {len(FAILURES)} check(s) FAILED:")
        for failure in FAILURES:
            print(f"    - {failure}")
        if on_failure:
            print(f"\n  {on_failure}")
    else:
        print(f"  all checks passed{' — ' + on_success if on_success else ''}")


def exit_code() -> int:
    return 1 if FAILURES else 0
