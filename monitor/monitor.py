from monitor.state import PortfolioState
from monitor.rules import HOLD, WARN_DRAWDOWN, WARN_CONCENTRATION
from monitor.logger import log_warning


DRAWDOWN_THRESHOLD = -0.15
CONCENTRATION_THRESHOLD = 0.40


def check_portfolio(state: PortfolioState) -> str:

    if state.drawdown_from_peak <= DRAWDOWN_THRESHOLD:
        action = WARN_DRAWDOWN
        log_warning(action, state)
        return action

    if state.max_weight >= CONCENTRATION_THRESHOLD:
        action = WARN_CONCENTRATION
        log_warning(action, state)
        return action

    return HOLD
