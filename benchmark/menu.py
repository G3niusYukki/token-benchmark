"""Interactive multi-select menu for picking models to benchmark.

Keybindings (prompt_toolkit path):

  Space       toggle the highlighted item
  a           select all
  i           invert selection
  n / Down    move down
  p / Up      move up
  /           start a fuzzy filter
  Enter       confirm current selection
  Esc / q     cancel (returns empty list)

If prompt_toolkit isn't available or stdin isn't a TTY, we fall back to a
plain numbered checklist driven by ``input()``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

from rich.console import Console

from benchmark.model_fetcher import ModelInfo

console = Console()


@dataclass
class SelectResult:
    """What the menu returns."""

    selected: List[ModelInfo]
    cancelled: bool = False


# ---------------------------------------------------------------------------
# prompt_toolkit implementation
# ---------------------------------------------------------------------------

def _run_prompt_toolkit(items: Sequence[ModelInfo], title: str) -> SelectResult:
    from prompt_toolkit import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import D
    from prompt_toolkit.styles import Style

    state: dict = {
        "cursor": 0,
        "selected": {m.id: True for m in items},   # all pre-selected
        "filter": "",
        "cancelled": False,
        "done": False,
    }

    def visible_indices() -> list[int]:
        if not state["filter"]:
            return list(range(len(items)))
        needle = state["filter"].lower()
        return [i for i, m in enumerate(items) if needle in m.id.lower()]

    def move(delta: int) -> None:
        vis = visible_indices()
        if not vis:
            return
        try:
            cur = vis.index(state["cursor"])
        except ValueError:
            cur = 0
        new = max(0, min(len(vis) - 1, cur + delta))
        state["cursor"] = vis[new]

    def toggle_current() -> None:
        mid = items[state["cursor"]].id
        state["selected"][mid] = not state["selected"][mid]

    def select_all(value: bool) -> None:
        for i in visible_indices():
            state["selected"][items[i].id] = value

    def invert() -> None:
        for i in visible_indices():
            mid = items[i].id
            state["selected"][mid] = not state["selected"][mid]

    def confirm() -> None:
        state["done"] = True

    def cancel() -> None:
        state["cancelled"] = True
        state["done"] = True

    def render_header() -> FormattedText:
        n_total = len(items)
        n_vis = len(visible_indices())
        n_sel = sum(1 for v in state["selected"].values() if v)
        return FormattedText([
            ("class:title", f"  {title}\n"),
            ("class:hint",
             f"  {n_sel}/{n_total} selected · {n_vis} visible · "
             "Space=toggle · a=all · i=invert · /=filter · Enter=confirm · q=cancel\n\n"),
        ])

    def render_list() -> FormattedText:
        ft: list[tuple[str, str]] = []
        vis = visible_indices()
        if not vis:
            ft.append(("class:empty", "  (no models match the filter)\n"))
        for idx in vis:
            m = items[idx]
            is_cur = idx == state["cursor"]
            is_sel = state["selected"][m.id]
            pointer = "> " if is_cur else "  "
            check = "[x]" if is_sel else "[ ]"
            style = "class:item.current" if is_cur else "class:item"
            line = f"{pointer}{check}  {m.display_name}"
            if m.owned_by and m.owned_by not in m.display_name:
                line += f"  ({m.owned_by})"
            ft.append((style, line + "\n"))
        return FormattedText(ft)

    def render_footer() -> FormattedText:
        if state["filter"]:
            ftext = f"  filter: /{state['filter']}"
        else:
            ftext = "  press / to filter, Enter to confirm"
        return FormattedText([("class:filter", ftext + "\n")])

    body_ctrl = FormattedTextControl(render_list, focusable=True, key_bindings=None)
    body_win = Window(content=body_ctrl, height=D(min(15, len(items) or 1)))

    header_ctrl = FormattedTextControl(render_header, focusable=False)
    footer_ctrl = FormattedTextControl(render_footer, focusable=False)

    root = HSplit([
        Window(content=header_ctrl, height=2, dont_extend_height=True),
        body_win,
        Window(content=footer_ctrl, height=1, dont_extend_height=True),
    ])

    style = Style.from_dict({
        "title":      "bold magenta",
        "hint":       "ansibrightblack",
        "item":       "",
        "item.current": "reverse bold cyan",
        "filter":     "ansiyellow",
        "empty":      "ansibrightblack italic",
    })

    kb = KeyBindings()

    @kb.add(" ")
    def _(event):
        toggle_current()

    @kb.add("a")
    def _(event):
        select_all(True)

    @kb.add("i")
    def _(event):
        invert()

    @kb.add("n")
    @kb.add("down")
    def _(event):
        move(1)

    @kb.add("p")
    @kb.add("up")
    def _(event):
        move(-1)

    @kb.add("/")
    def _(event):
        state["filter"] = ""

    @kb.add("enter")
    def _(event):
        confirm()

    @kb.add("q")
    @kb.add("escape")
    @kb.add("c-c")
    def _(event):
        cancel()

    app = Application(
        layout=Layout(root, focused=body_win),
        key_bindings=kb,
        style=style,
        mouse_support=True,
        full_screen=False,
    )

    app.run()

    if state["cancelled"]:
        return SelectResult(selected=[], cancelled=True)

    chosen = [m for m in items if state["selected"].get(m.id, False)]
    return SelectResult(selected=chosen, cancelled=False)


# ---------------------------------------------------------------------------
# Plain input() fallback
# ---------------------------------------------------------------------------

def _run_plain(items: Sequence[ModelInfo], title: str) -> SelectResult:
    console.print(f"\n[bold magenta]{title}[/bold magenta]")
    console.print(
        "[dim]Enter a comma-separated list of numbers to test "
        "(empty = all, 'q' to cancel).[/dim]\n"
    )
    for i, m in enumerate(items, 1):
        owner = f"  [dim]({m.owned_by})[/dim]" if m.owned_by else ""
        console.print(f"  [cyan]{i:>3}[/cyan]. {m.display_name}{owner}")
    console.print()
    raw = input("  Select (e.g. 1,3,5): ").strip()
    if raw.lower() in ("q", "quit", "exit"):
        return SelectResult(selected=[], cancelled=True)
    if not raw:
        return SelectResult(selected=list(items), cancelled=False)
    try:
        idxs = sorted({int(x) - 1 for x in raw.split(",") if x.strip()})
    except ValueError:
        console.print("  [red]Invalid input — defaulting to all models.[/red]")
        return SelectResult(selected=list(items), cancelled=False)
    chosen = [items[i] for i in idxs if 0 <= i < len(items)]
    return SelectResult(selected=chosen, cancelled=False)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def select_models(
    items: Sequence[ModelInfo],
    title: str = "Select models to benchmark",
    *,
    force_plain: bool = False,
) -> SelectResult:
    """Show an interactive multi-select and return the chosen models."""
    if not items:
        return SelectResult(selected=[], cancelled=True)

    if force_plain or not sys.stdin.isatty() or not sys.stdout.isatty():
        return _run_plain(items, title)

    try:
        return _run_prompt_toolkit(items, title)
    except Exception as e:
        console.print(f"[yellow]Interactive picker unavailable ({e}); "
                      "falling back to plain input.[/yellow]")
        return _run_plain(items, title)


def select_endpoint_style(default: Optional[str] = None) -> Optional[str]:
    """Lightweight single-select for endpoint style."""
    from benchmark.endpoint_detector import EndpointStyle as ES

    if not sys.stdin.isatty():
        raw = input("\n  Endpoint style [openai/anthropic] (default=openai): ").strip()
        if not raw:
            return ES.OPENAI.value
        return raw.lower()

    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import WordCompleter
        completer = WordCompleter(["openai", "anthropic"], ignore_case=True)
        raw = prompt(
            "  Endpoint style (openai/anthropic): ",
            completer=completer,
            default=default or "",
        ).strip()
        return raw.lower() or (default or ES.OPENAI.value)
    except Exception:
        raw = input("  Endpoint style (openai/anthropic): ").strip()
        return raw.lower() or (default or ES.OPENAI.value)


def ask_text(
    question: str,
    default: str = "",
    password: bool = False,
) -> str:
    """One-line free-form text input with prompt_toolkit (falls back to input)."""
    suffix = " " if not question.endswith(" ") else ""
    if password and sys.stdin.isatty():
        try:
            from prompt_toolkit import prompt
            return prompt(
                question + suffix,
                is_password=True,
                default=default,
            ).strip()
        except Exception:
            pass
    if sys.stdin.isatty():
        try:
            from prompt_toolkit import prompt
            return prompt(question + suffix, default=default).strip()
        except Exception:
            pass
    shown_default = f" [{default}]" if default else ""
    raw = input(question + shown_default + ": ").strip()
    return raw or default


def ask_confirm(question: str, default: bool = True) -> bool:
    if not sys.stdin.isatty():
        suffix = " [Y/n]" if default else " [y/N]"
        raw = input(question + suffix + ": ").strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes")
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.key_binding import KeyBindings
        kb = KeyBindings()

        @kb.add("y")
        @kb.add("Y")
        def _(event):
            event.app.exit(result=True)

        @kb.add("n")
        @kb.add("N")
        def _(event):
            event.app.exit(result=False)

        @kb.add("enter")
        def _(event):
            event.app.exit(result=default)

        result = prompt(question + " (y/n): ", key_bindings=kb, default="Y" if default else "N")
        return result.strip().lower() in ("y", "yes")
    except Exception:
        suffix = " [Y/n]" if default else " [y/N]"
        raw = input(question + suffix + ": ").strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes")
