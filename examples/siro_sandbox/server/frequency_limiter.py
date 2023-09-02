import time
import asyncio

class FrequencyLimiter:
    def __init__(self, desired_frequency):
        self.desired_frequency = desired_frequency
        self.desired_period = 1 / desired_frequency if desired_frequency else None
        self.last_execution_time = time.time()

    def limit_frequency(self):
        if not self.desired_period:
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.last_execution_time

        # Calculate the remaining time to meet the desired period
        remaining_time = self.desired_period - elapsed_time

        # Delay the loop execution if needed
        if remaining_time > 0:
            time.sleep(remaining_time)

        self.last_execution_time = time.time()

    async def limit_frequency_async(self):
        if not self.desired_period:
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.last_execution_time
        remaining_time = self.desired_period - elapsed_time

        # Delay the loop execution if needed
        if remaining_time > 0:
            await asyncio.sleep(remaining_time)

        # current_time = time.time()
        # elapsed_time = current_time - self.last_execution_time
        # remaining_time = self.desired_period - elapsed_time
        # if remaining_time > 0:
        #     time.sleep(remaining_time)

        self.last_execution_time = time.time()
