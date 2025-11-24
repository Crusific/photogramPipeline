# Dataset module

Provides a small `DataSet` loader for JSON files with drone and target locations.

Features
- Loads JSON `data` arrays that may contain `null` slots.
- `DataSet(path, cyclic=True, skip_nulls=False, validate=False)`
- `next()`, `peek()`, `reset(index=0)` methods.
- Pythonic helpers: `__iter__`, `__len__`, `__getitem__`.
- `to_dict()` and `save(path)` for atomic persistence.

Quick start
```powershell
python example.py
```

Testing
- Install dependencies: `pip install -r requirements.txt`
- Run tests: `pytest -q`

Suggested CI
- Run tests on push using `pytest` in a small GitHub Actions workflow.
