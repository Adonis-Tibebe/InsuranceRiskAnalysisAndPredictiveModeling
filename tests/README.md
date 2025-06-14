# Tests

This directory contains unit tests for the source code in the `../src/` directory.

## Structure

- **test_EDA_analysis.py**  
  Contains tests for the data cleaning, imputation, and plotting functions provided in `src/EDA_analysis.py`.

## Running Tests

To run all tests using the built-in `unittest` framework, use:

```sh
python -m unittest discover
```

You can also use the VS Code test explorer for interactive test runs.

## Notes

- Make sure the `src/` directory is included in your Python path when running tests.
- Test data may use small samples or mocks for efficiency and clarity.
