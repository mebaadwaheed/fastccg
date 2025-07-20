import typer
import os
import asyncio
from rich.console import Console
from rich.table import Table
import inspect

import fastccg
from fastccg.models import gpt, gemini, claude, mistral

# Initialize Typer app and Rich console
app = typer.Typer(
    name="fastccg",
    help="A minimalist CLI for rapid LLM prompt engineering.",
    add_completion=False,
)
console = Console()

def get_all_model_classes():
    """Dynamically get all model classes from the modules."""
    all_models = []
    for module in [gpt, gemini, claude, mistral]:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, fastccg.ModelBase) and not name.startswith('_'):
                all_models.append({
                    "alias": name,
                    "provider": obj.provider,
                    "class": obj
                })
    return all_models

@app.command()
def models():
    """
    List all available model presets.
    """
    table = Table(title="FastCCG Supported Models")
    table.add_column("Alias", style="cyan", no_wrap=True)
    table.add_column("Provider", style="magenta")

    model_classes = get_all_model_classes()
    
    for model_info in sorted(model_classes, key=lambda x: (x['provider'], x['alias'])):
        table.add_row(model_info['alias'], model_info['provider'].capitalize())

    console.print(table)


def get_model_class_by_alias(alias: str):
    """Get a model class by its alias."""
    for model_info in get_all_model_classes():
        if model_info['alias'].lower() == alias.lower():
            return model_info['class']
    return None

@app.command()
def ask(
    prompt: str = typer.Argument(..., help="The prompt to send to the model."),
    model: str = typer.Option("gpt-3.5-turbo", "--model", "-m", help="Model alias to use."),
    key: str = typer.Option(None, "--key", "-k", help="API key. If not provided, checks environment variables."),
    temperature: float = typer.Option(None, "--temperature", "-t", help="Temperature for sampling."),
    max_tokens: int = typer.Option(None, "--max-tokens", "-mt", help="Maximum tokens in the response."),
):
    """
    One-shot prompt with a simple printout.
    """
    model_class = get_model_class_by_alias(model)
    if not model_class:
        console.print(f"[bold red]Error:[/] Model alias '{model}' not found. Use 'fastccg models' to see available models.")
        raise typer.Exit(code=1)

    provider = model_class.provider
    api_key = key or os.getenv(f"{provider.upper()}_API_KEY")

    if not api_key:
        console.print(f"[bold red]Error:[/] API key for {provider.capitalize()} not found. Pass it with --key or set the {provider.upper()}_API_KEY environment variable.")
        raise typer.Exit(code=1)

    try:
        model_instance = fastccg.init_model(model_class, api_key=api_key)
        if temperature is not None:
            model_instance.temperature(temperature)
        if max_tokens is not None:
            model_instance.max_tokens(max_tokens)

        with console.status("[bold green]Asking AI..."):
            response = model_instance.ask(prompt)
        
        console.print(f"[bold green]AI Response:[/] {response.content}")

    except Exception as e:
        console.print(f"[bold red]An error occurred:[/] {e}")
        raise typer.Exit(code=1)

@app.command()
def chat(
    model: str = typer.Option("gpt-3.5-turbo", "--model", "-m", help="Model alias to use."),
    key: str = typer.Option(None, "--key", "-k", help="API key. If not provided, checks environment variables."),
    system: str = typer.Option(None, "--system", "-s", help="Initial system prompt."),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Enable/disable token streaming."),
    save_path: str = typer.Option(None, "--save", help="Save conversation history to a JSON file."),
):
    """
    Start an interactive terminal chat session.
    """
    model_class = get_model_class_by_alias(model)
    if not model_class:
        console.print(f"[bold red]Error:[/] Model alias '{model}' not found.")
        raise typer.Exit(code=1)

    provider = model_class.provider
    api_key = key or os.getenv(f"{provider.upper()}_API_KEY")

    if not api_key:
        console.print(f"[bold red]Error:[/] API key for {provider.capitalize()} not found.")
        raise typer.Exit(code=1)

    try:
        model_instance = fastccg.init_model(model_class, api_key=api_key)
        if system:
            model_instance.sys_prompt(system)

        console.print(f"[bold green]Starting chat with {model}.[/] Type 'exit' or 'reset'.")

        while True:
            prompt = console.input("[bold cyan]You: [/]")
            if prompt.lower() == 'exit':
                if save_path:
                    model_instance.save(save_path)
                    console.print(f"[yellow]Session saved to {save_path}.[/]")
                break
            if prompt.lower() == 'reset':
                model_instance.history.clear()
                console.print("[yellow]Chat history cleared.[/]")
                continue

            console.print("[bold green]AI: [/]", end="")
            if stream:
                full_response = ""
                with console.status("..."):
                    async def stream_response():
                        nonlocal full_response
                        async for chunk in model_instance.ask_stream(prompt):
                            console.print(chunk.content, end="", flush=True)
                            full_response += chunk.content
                    asyncio.run(stream_response())
                console.print()
            else:
                with console.status("[bold green]Asking AI..."):
                    response = model_instance.ask(prompt)
                console.print(response.content)

    except (KeyboardInterrupt, EOFError):
        if save_path:
            model_instance.save(save_path)
            console.print(f"\n[yellow]Session saved to {save_path}. Exiting.[/]")
        else:
            console.print("\n[yellow]Exiting.[/]")
    except Exception as e:
        console.print(f"[bold red]An error occurred:[/] {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()


