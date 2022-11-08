import subprocess
import signal


class RoscoreWrapper:
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
        self.roscore_proc.wait()
        print("Starting roscore...")


if __name__ == "__main__":
    rw = RoscoreWrapper()
    rw.wait()
