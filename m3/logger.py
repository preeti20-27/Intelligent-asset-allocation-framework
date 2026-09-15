import logging


logging.basicConfig(
    filename="monitor_warnings.log",
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def log_warning(action, state):
    logging.warning(
        "Action=%s | Date=%s | Drawdown=%.2f%% | "
        "MaxWeight=%.2f%% | Volatility=%.2f%% | "
        "WeeklyReturn=%.2f%%",
        action,
        state.date,
        state.drawdown_from_peak * 100,
        state.max_weight * 100,
        state.volatility * 100,
        state.weekly_return * 100
    )