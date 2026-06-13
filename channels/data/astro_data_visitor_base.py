import abc as _dvb_abc


class AstroDataVisitorBase(_dvb_abc.ABC):
    """Port of DataVisitorBase — abstract base for data visitors."""
    def __init__(self):
        self._notified = False
        self._notify_callback = None

    def register_notify_callback(self, cb):
        self._notify_callback = cb

    @_dvb_abc.abstractmethod
    def try_fetch(self) -> bool: ...

    def _notify(self):
        self._notified = True
        if self._notify_callback:
            self._notify_callback()
