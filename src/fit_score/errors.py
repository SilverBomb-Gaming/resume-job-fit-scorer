"""Expected failures. The CLI prints these without a traceback."""


class FitScoreError(Exception):
    """Base class for errors the user can act on."""


class UsageError(FitScoreError):
    """The command line or input files are not usable."""


class RubricError(FitScoreError):
    """A rubric file is missing or invalid."""


class LLMError(FitScoreError):
    """The model provider could not complete a request."""


class ScoringError(FitScoreError):
    """The model response could not be turned into a score."""
