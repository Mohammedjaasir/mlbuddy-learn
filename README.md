# MLPilot

Machine Learning automation and guidance system.

## Project Structure

- **mlpilot/auto/**: Automated machine learning tasks
  - `data.py`: Data handling and preprocessing
  - `trainer.py`: Model training utilities
  
- **mlpilot/guide/**: ML guidance and suggestions
  - `suggest.py`: Suggestions engine for ML workflows
  
- **mlpilot/explain/**: Model interpretation and visualization
  - `visualizer.py`: Model visualization tools
  
- **tests/**: Test suite

## Installation

```bash
pip install -e .
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```
