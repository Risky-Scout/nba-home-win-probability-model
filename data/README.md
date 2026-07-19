# Local data

Place the recruiter-provided file here as:

```text
data/nba-win-probability-data.csv
```

The raw assignment dataset is intentionally excluded from version control by
`.gitignore`. The expected SHA-256 hash is recorded in
`artifacts/data_audit.json`. A reviewer who already has the supplied file can
reproduce the project with:

```bash
make reproduce DATA=/absolute/path/to/nba-win-probability-data.csv
```
