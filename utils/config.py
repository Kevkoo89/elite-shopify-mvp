from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    seasons: tuple[str, ...] = ("2324", "2425", "2526")
    default_league: str = "D1"
    default_budget: float = 1000.0
    sentiment_baseline: int = 50
    sentiment_positive_threshold: int = 60
    sentiment_negative_threshold: int = 40
    poisson_simulations: int = 10000


CONFIG = AppConfig()
