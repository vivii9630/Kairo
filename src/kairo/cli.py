from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import print

from .client import HostedClient
from .local import LocalPipeline
from .models import Document, QueryRequest
from .ingest import ingest_any


app = typer.Typer(help="Kairo CLI: ingest documents and run queries (hosted or local mode)")


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Path to file (jsonl/csv/xlsx/pdf/docx/txt/audio)"),
    store: Path = typer.Option(Path(".kairo_store.json"), help="Local store path"),
):
    """Ingest a supported file into the local store."""
    pipeline = LocalPipeline(store_path=store)
    docs = ingest_any(path)
    pipeline.ingest_documents(docs)
    print(f"[green]Ingested {len(docs)} documents from {path} into {store}[/green]")


@app.command()
def query(
    query: str = typer.Argument(..., help="User query"),
    mode: str = typer.Option("local", help="Mode: local or hosted"),
    top_k: int = typer.Option(5, help="Top-K evidences"),
    store: Path = typer.Option(Path(".kairo_store.json"), help="Local store path"),
    base_url: Optional[str] = typer.Option(None, help="Hosted base URL"),
    api_key: Optional[str] = typer.Option(None, help="Hosted API key"),
):
    """Run a query against local store or hosted service."""
    if mode == "hosted":
        if not base_url:
            typer.echo("--base-url is required for hosted mode", err=True)
            raise typer.Exit(code=1)
        client = HostedClient(base_url=base_url, api_key=api_key)
        resp = client.query(QueryRequest(query=query, top_k=top_k))
    else:
        pipeline = LocalPipeline(store_path=store)
        resp = pipeline.query(QueryRequest(query=query, top_k=top_k))

    print("[bold]Answer:[/bold]", resp.answer)
    print("\n[bold]Evidences:[/bold]")
    for ev in resp.evidences:
        print(f"- ({ev.score:.3f}) {ev.document_id}: {ev.text[:200]}")

    if resp.steps:
        print("\n[bold]Steps:[/bold]")
        for step in resp.steps:
            print(f"- {step.name}: {step.output}")


if __name__ == "__main__":
    app()

