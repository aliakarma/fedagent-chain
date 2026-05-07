# Pull Request

## Summary
Brief description of what this PR changes and why.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature that changes existing behavior)
- [ ] Documentation update
- [ ] Reproducibility fix
- [ ] Dependency update
- [ ] Refactoring (no functional change)

## Related Issues
Closes #

## Checklist
- [ ] My code follows the project style guide (`make lint` passes)
- [ ] I have added tests for new functionality
- [ ] All existing tests pass (`make test` passes)
- [ ] I have updated docstrings for changed public functions
- [ ] I have updated `CHANGELOG.md`
- [ ] If adding a new config parameter, I have added it to `configs/default.yaml`
- [ ] If changing evaluation metrics, I have verified regression tests still pass

## Testing
Describe how the changes were tested:
```bash
pytest tests/unit/test_<module>.py -v
```

## Notes for Reviewers
Any additional context that reviewers should know.
