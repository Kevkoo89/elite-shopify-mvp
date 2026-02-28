from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    seasons: tuple[str, ...] = ("2324", "2425", "2526")
    default_domain: str = "sports"
    default_budget: float = 1000.0
    poisson_simulations: int = 10000


CONFIG = AppConfig()
