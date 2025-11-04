import sys
import subprocess
import typer

app = typer.Typer()


@app.command()
def pull(term: str = "endometriosis", limit: int = 200):
    sys.exit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "src.pipelines.pull_all",
                "--term",
                term,
                "--limit",
                str(limit),
            ]
        )
    )


@app.command()
def load():
    sys.exit(subprocess.call([sys.executable, "-m", "src.pipelines.normalize_load"]))


if __name__ == "__main__":
    app()
