import time

class AverageRateTracker:
    def __init__(self, duration_window):
        self._recent_count = 0
        self._recent_time = time.time()
        self._duration_window = duration_window
        self._recent_rate = 0

    def increment(self, inc):
    
        self._recent_count += inc

    def get_smoothed_rate(self):

        current_time = time.time()
        elapsed_time = current_time - self._recent_time
        if elapsed_time > self._duration_window:
            self._recent_rate = self._recent_count / elapsed_time
            self._recent_count = 0
            self._recent_time = current_time

        return self._recent_rate
