from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .data import generate_sample_data, load_providers, save_dataframe
from .scoring import score_providers


app = typer.Typer(add_completion=False, help="High Risk Providers CLI")


@app.command("generate-sample")
def generate_sample(
    n: int = typer.Option(1000, "--n", help="Number of providers to generate"),
    out: str = typer.Option(
        "./outputs/sample_providers.csv",
        "--out",
        help="Output CSV path",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    df = generate_sample_data(num_providers=n, seed=seed)
    save_dataframe(df, out)
    typer.echo(f"Wrote sample data to {out} ({len(df)} providers)")


@app.command("score")
def score(
    input: Optional[str] = typer.Option(
        None, "--input", "-i", help="Input providers CSV. If omitted, synthetic data used."
    ),
    out: str = typer.Option(
        "./outputs/provider_scores.csv",
        "--out",
        "-o",
        help="Output CSV path",
    ),
):
    df = load_providers(input)
    scored = score_providers(df)

    # Quick summary
    n = len(scored)
    high = int((scored["risk_label"] == "high").sum())
    elevated = int((scored["risk_label"] == "elevated").sum())

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)

    typer.echo(f"Scored {n} providers.")
    typer.echo(f"High risk: {high} | Elevated: {elevated}")
    typer.echo(f"Output written to {out}")


@app.command("summarize")
def summarize(
    input: str = typer.Argument(..., help="Path to scored CSV produced by 'score'"),
    top_k: int = typer.Option(10, "--top-k", help="Top K providers to display"),
):
    path = Path(input)
    if not path.exists():
        typer.secho(f"Input '{input}' not found", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    df = pd.read_csv(path)
    if "risk_score" not in df.columns:
        typer.secho(
            "Input CSV missing 'risk_score'. Please run 'score' first.",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)

    df = df.sort_values("risk_score", ascending=False)
    cols = [c for c in ["provider_id", "state", "risk_score", "risk_label"] if c in df.columns]
    preview = df[cols].head(top_k)

    typer.echo("Top providers by risk_score:")
    typer.echo(preview.to_string(index=False))


if __name__ == "__main__":
    app()