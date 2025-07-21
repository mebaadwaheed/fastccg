import json
import os
from pathlib import Path
from typing_extensions import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="A CLI tool for inspecting, validating, and converting .fcvs (FastCCG Vector Store) files."
)
console = Console()


def _load_fcvs_file(filepath: Path) -> dict:
    """Loads and performs basic validation on an .fcvs file."""
    if not filepath.exists():
        console.print(f"[bold red]Error:[/] File not found at '{filepath}'")
        raise typer.Exit(code=1)
    if filepath.suffix != ".fcvs":
        console.print(f"[bold yellow]Warning:[/] File does not have the '.fcvs' extension.")

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("FCVS file content must be a JSON object (dictionary).")
        return data
    except json.JSONDecodeError:
        console.print(f"[bold red]Error:[/] Invalid JSON. Could not decode the file at '{filepath}'.")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/] An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def inspect(
    filepath: Annotated[Path, typer.Argument(help="The path to the .fcvs file to inspect.")]
):
    """
    Inspects an .fcvs file and displays metadata about its contents.
    """
    console.print(f"--- Inspecting [cyan]{filepath.name}[/cyan] ---")
    data = _load_fcvs_file(filepath)

    num_vectors = len(data)
    file_size_bytes = os.path.getsize(filepath)
    file_size_kb = file_size_bytes / 1024

    table = Table(title="File Metadata")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="green")

    table.add_row("File Size", f"{file_size_bytes:,} bytes (~{file_size_kb:.2f} KB)")
    table.add_row("Number of Vectors", str(num_vectors))

    if num_vectors > 0:
        first_key = next(iter(data))
        first_item = data[first_key]
        vector_dim = 0
        if isinstance(first_item, (list, tuple)) and len(first_item) > 0 and isinstance(first_item[0], list):
            vector_dim = len(first_item[0])
        table.add_row("Vector Dimensions", str(vector_dim))
        table.add_row("Metadata Present", str(isinstance(first_item[1], dict)))

    console.print(table)


@app.command()
def validate(
    filepath: Annotated[Path, typer.Argument(help="The path to the .fcvs file to validate.")]
):
    """
    Validates the structure and format of an .fcvs file.
    """
    console.print(f"--- Validating [cyan]{filepath.name}[/cyan] ---")
    data = _load_fcvs_file(filepath)

    is_valid = True
    for key, value in data.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            console.print(f"[bold red]Error:[/] Invalid structure for key '{key}'. Expected a list/tuple of length 2.")
            is_valid = False
            break
        if not isinstance(value[0], list):
            console.print(f"[bold red]Error:[/] Invalid vector for key '{key}'. Expected a list of floats.")
            is_valid = False
            break
        if not isinstance(value[1], dict):
            console.print(f"[bold red]Error:[/] Invalid metadata for key '{key}'. Expected a dictionary.")
            is_valid = False
            break

    if is_valid:
        console.print("[bold green]Success:[/] File is a valid .fcvs file.")
    else:
        raise typer.Exit(code=1)


@app.command()
def convert(
    filepath: Annotated[Path, typer.Argument(help="The path to the .fcvs file to convert.")],
    to: Annotated[str, typer.Option("--to", help="The format to convert to. Currently only 'json' is supported.")] = "json",
):
    """
    Converts an .fcvs file to another format (e.g., json).
    """
    if to.lower() != 'json':
        console.print(f"[bold red]Error:[/] Conversion to '{to}' is not supported. Only 'json' is available.")
        raise typer.Exit(code=1)

    console.print(f"--- Converting [cyan]{filepath.name}[/cyan] to JSON ---")
    # Since .fcvs is already a JSON file, this is mainly a copy/rename operation.
    data = _load_fcvs_file(filepath)
    
    output_path = filepath.with_suffix('.json')

    if output_path.exists():
        overwrite = typer.confirm(f"File '{output_path}' already exists. Overwrite?")
        if not overwrite:
            console.print("Conversion cancelled.")
            raise typer.Exit()

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    console.print(f"[bold green]Success:[/] File converted and saved to '{output_path}'.")


if __name__ == "__main__":
    app()
