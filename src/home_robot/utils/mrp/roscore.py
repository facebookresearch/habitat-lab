import os
import sys
import subprocess
import signal
import time


class RoscoreWrapper:
    """
    Thin ROS wrapper for launching and terminating via MRP

    MRP terminates processes by sending a SIGTERM.
    roscore is designed to exit gracefully under SIGINT (Ctrl-C).
    This wrapper catches the SIGTERM from `mrp down` and sends a SIGINT to the underlying roscore process.
    """

    def __init__(self):
        signal.signal(signal.SIGINT, self._term_handler)
        signal.signal(signal.SIGTERM, self._term_handler)

        print("Starting roscore...")
        self.roscore_proc = subprocess.Popen(["roscore"], stdout=subprocess.PIPE)
        print("roscore started.")

    def wait(self):
        self.roscore_proc.wait()

    def _term_handler(self, *args):
        print("Terminating roscore...")
        self.roscore_proc.send_signal(signal.SIGINT)
        time.sleep(2)
        os.system("killall rosmaster rosout")
        print("roscore terminated.")
        sys.exit(0)


if __name__ == "__main__":
    rw = RoscoreWrapper()
    rw.wait()
