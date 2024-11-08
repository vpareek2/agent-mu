import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

@contextmanager
def rank_logging_context(rank: int):
    """Add rank to all log records within this context."""
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)
    try:
        yield
    finally:
        logging.setLogRecordFactory(old_factory)

def setup_logging(log_dir: str = "logs", rank: int = 0) -> None:
    """Setup logging with rank-aware formatting."""
    if rank == 0:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers = [
            logging.FileHandler(f"{log_dir}/muzero_{timestamp}.log"),
            logging.StreamHandler()
        ]
    else:
        handlers = [logging.StreamHandler()]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [Rank %(rank)s] %(message)s',
        handlers=handlers
    )
