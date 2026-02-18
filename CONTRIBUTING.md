# Contributing to AI Chess Architect

Thank you for your interest in contributing to the AI Chess Architect! We welcome contributions that improve the AI model, the web interface, or the project's documentation.

## Getting Started

1.  **Fork the Repository**: Create a personal fork of the project on GitHub.
2.  **Clone the Fork**:
    ```bash
    git clone https://github.com/PunithaYaku/Chess-Puzzel-Gen.git
    cd Chess-Puzzel-Gen
    ```
3.  **Set Up the Environment**:
    Create a virtual environment and install the required dependencies:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    pip install -r requirements.txt
    ```

## Development Guidelines

### Branching Policy
-   Create a feature branch for any new work: `git checkout -b feature/your-feature-name`.
-   Keep your commits small and focused. Each commit should have a clear, descriptive message (e.g., `feat: ...`, `fix: ...`, `docs: ...`).
-   We encourage using a `CONTRIBUTIONS_LOG.md` to track your daily progress and goals.

### Testing
-   Before submitting a Pull Request, please ensure your changes do not break existing functionality.
-   If you add new logic, please include corresponding unit tests in the `tests/` directory.
-   Run tests using `pytest` (to be implemented).

## Code Style
-   Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.
-   Use descriptive variable and function names.
-   Add docstrings to all new functions and classes.

## Pull Request Process
1.  Ensure your code is up to date with the `main` branch.
2.  Push your changes to your fork.
3.  Submit a Pull Request to the `main` repository.
4.  Provide a clear description of the changes and why they are necessary.

## Community and Feedback
If you find a bug or have a suggestion, please open an issue on the GitHub repository. We appreciate your feedback!
