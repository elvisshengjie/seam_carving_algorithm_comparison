# Working on the project together

## Get a local copy

```powershell
git clone https://github.com/elvisshengjie/seam_carving_algorithm_comparison.git
cd seam_carving_algorithm_comparison
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Use Python 3.11 or newer. To launch the desktop interface:

```powershell
python -m seam_carving gui
```

The repository owner can give teammates write access under GitHub **Settings >
Collaborators > Add people**. Each teammate must accept the invitation before
pushing to this repository. A public repository allows others to read and fork
the code; it does not automatically give them write access.

## Make a change

Update your copy first, so your commit goes on top of everyone else's work:

```powershell
git switch main
git pull --ff-only
```

Edit the code or documentation, run the tests, then commit only the relevant files:

```powershell
python -m unittest discover -s tests -v
git status
git add seam_carving tests README.md
git commit -m "Describe the change"
git push
```

Change the staged paths to match your task. For larger changes, push a separate
branch (`git switch -c my-change`, then `git push -u origin my-change`) and open
a pull request against `main` so a teammate can review it. GitHub Actions runs
the tests, a demo comparison and the source packaging on every push and pull
request; check that the run is green before relying on it.

Do not commit virtual environments, caches, generated output, credentials, or
the course handout PDFs. Keep experimental commands and input sources documented
so teammates can reproduce results. Discuss changes to the shared energy function
or seam objective before comparing algorithms that use different settings.

## Suggested work areas

- Core solvers and correctness checks: `seam_carving/seams.py`, `flow.py`, `tests/`.
- GUI and visualization: `gui.py`, `visualization.py`.
- Experiments and evaluation: `experiments.py`, documented runs on common inputs.
- Analysis and report: the assignment report (submitted on xSITE) and `README.md`.

Record any AI assistance accurately in the report's AI usage declaration.
