# laurus_llm_client/lauruslog.py
import logging


class LaurusLogger:
    """
    Centralized logger for the client library.
    Use:
        from laurus_llm_client.lauruslog import LOG
        LOG.info("Message")
    """

    @staticmethod
    def get_logger(
        name="laurus-llm-client", level=logging.INFO, suppress_uvicorn_access=True
    ):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Avoid duplicate handlers
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        logger.propagate = False  # prevent double logging

        # Optionally suppress uvicorn access logs (if library used in server context)
        if suppress_uvicorn_access:
            uvicorn_loggers = ["uvicorn.access"]
            for ul in uvicorn_loggers:
                ulog = logging.getLogger(ul)
                ulog.setLevel(logging.WARNING)
                ulog.propagate = False

        return logger


# singleton instance for easy import
LOG = LaurusLogger.get_logger()
