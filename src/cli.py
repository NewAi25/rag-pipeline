"""Command-line entry point.

Three commands:

    python -m src.cli ingest <pdf-or-folder>
    python -m src.cli ask    "<question>"
    python -m src.cli clear

Typer gives us nice ``--help`` output, type-checked arguments, and colored
error messages for free.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .config import load_settings
from .generate import generate_answer
from .ingest import ingest_pdf
from .retrieve import retrieve
from .vectorstore import get_vector_store

app = typer.Typer(
    add_completion=False,
    help="A tiny RAG demo: ingest a PDF (or folder of PDFs), then ask grounded questions about it.",
)


@app.command()
def ingest(
    path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Path to a PDF file, or a folder containing PDFs.",
    ),
) -> None:
    """Parse, chunk, embed, and store one PDF — or every PDF in a folder."""
    settings = load_settings()

    # Resolve the list of PDFs to ingest.
    if path.is_dir():
        pdfs = sorted(p for p in path.rglob("*.pdf") if p.is_file())
        if not pdfs:
            typer.secho(
                f"No PDFs found under {path}.", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=1)
        typer.echo(f"Found {len(pdfs)} PDF(s) in {path}.")
    else:
        pdfs = [path]

    grand_total = 0
    for pdf in pdfs:
        typer.echo(f"Ingesting {pdf} ...")
        try:
            n = ingest_pdf(pdf, settings=settings)
        except RuntimeError as e:
            typer.secho(str(e), fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        typer.secho(f"  + {n} chunks", fg=typer.colors.GREEN)
        grand_total += n

    typer.secho(
        f"Done. Wrote {grand_total} chunks to {settings.chroma_dir} "
        f"(collection: {settings.chroma_collection}).",
        fg=typer.colors.GREEN,
        bold=True,
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question, in quotes."),
) -> None:
    """Retrieve the most relevant chunks and answer with the LLM."""
    settings = load_settings()
    try:
        chunks = retrieve(question, settings=settings)
        answer = generate_answer(question, chunks, settings=settings)
    except RuntimeError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("\nAnswer:", fg=typer.colors.CYAN, bold=True)
    typer.echo(answer.text)

    typer.secho("\nSources used:", fg=typer.colors.CYAN, bold=True)
    if not answer.sources:
        typer.echo("  (none)")
        return
    for c in answer.sources:
        # Show a short preview so it's obvious which passage drove the answer.
        preview = c.text.replace("\n", " ").strip()
        if len(preview) > 120:
            preview = preview[:117] + "..."
        typer.echo(
            f"  - [{c.source}#chunk-{c.chunk_index}] "
            f"(distance={c.distance:.3f})  {preview}"
        )


@app.command()
def clear(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the interactive confirmation prompt.",
    ),
) -> None:
    """Wipe the vector store collection so you can re-ingest from scratch."""
    settings = load_settings()
    store = get_vector_store(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
    )
    n = store.count()
    if n == 0:
        typer.echo(
            f"Collection '{settings.chroma_collection}' is already empty."
        )
        return
    if not yes:
        typer.confirm(
            f"Delete all {n} chunks from collection "
            f"'{settings.chroma_collection}' at {settings.chroma_dir}?",
            abort=True,
        )
    store.reset()
    typer.secho(
        f"Cleared {n} chunks from '{settings.chroma_collection}'.",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
