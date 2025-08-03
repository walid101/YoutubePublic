from abc import ABC, abstractmethod
import logging


class UtilitySource(ABC):
    """
    This class must be implemented by any of Util class we may use in the future.
    """

    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.logger.info("Initialized UtilitySource")
        pass

    @abstractmethod
    def cleanup(self):
        self.logger.info("Cleaning any artificats from UtilitySource...")
