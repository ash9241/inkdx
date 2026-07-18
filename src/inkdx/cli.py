"""inkdx command-line interface."""

from __future__ import annotations

import typer

from inkdx import __version__

app = typer.Typer(
    name="inkdx",
    help="Ink-failure diagnostics for the Vesuvius Challenge: "
    "attribute missing ink to scan, surface, or model.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """inkdx: scan / surface / model failure attribution for scroll segments."""


@app.command()
def version() -> None:
    """Print the inkdx version."""
    typer.echo(__version__)


# `inkdx run`, `inkdx ablate`, `inkdx calibrate`, `inkdx compare` land with their stages.
