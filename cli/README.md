# kairo-cli

Typer CLI for Kairo. Ingest documents and run queries against local or
hosted backends.

```bash
kairo ingest examples/sample.jsonl
kairo query "temporal graphs" --top-k 3
kairo query "temporal graphs" --mode hosted --base-url https://api.example --api-key ...
```

Install:

```bash
pip install -e ./core ./ingest ./retrieval ./cli
```
