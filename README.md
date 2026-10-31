# ML Assignment 2 — m26csa004

## Run
```bash
python -m pip install numpy pandas matplotlib scikit-learn
python solution.py
```

The script uses the numeric digits of `m26csa004` as the seed (`26004`), creates and saves a stratified 6/2/2 split per identity, uses raw flattened pixels, implements bagging and SAMME itself, evaluates clean and masked test sets, and writes CSV/JSON results and plots.

Outputs are written to `results_m26csa004/`. Run from the folder containing `solution.py`.

## Important experimental notes
- The first run may need internet access for `fetch_olivetti_faces` if the dataset is not cached.
- Do not copy hypothetical results into the report: use the generated CSV/JSON values.
- The report must discuss observed results, not just generic bias–variance claims.
- Review `solution.py` and verify all output files before submission. The script is a reproducible starting point and should be run and checked in your environment.
