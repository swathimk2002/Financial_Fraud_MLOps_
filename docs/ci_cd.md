# CI/CD Automation

## Purpose
GitHub Actions automates project testing whenever code changes are pushed.

## Configuration
Workflow file: .github/workflows/ci.yml

## Workflow Steps
1. Check out the repository.
2. Set up Python 3.11.
3. Install dependencies.
4. Run automated tests using pytest.

## Testing
- Test command: python -m pytest -v
- Latest local result: 70 passed, 3 warnings.
- GitHub Actions CI: Successfully completed.

## Benefits
- Detects code errors early.
- Checks project functionality after changes.
- Improves code reliability.
