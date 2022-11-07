import os
import signal

if __name__ == "__main__":
    # Launch rosmaster
    os.system("roscore")

    # Cleanup on termination
    signal.sigwait([signal.SIGTERM])
    os.system("killall", "rosmaster")
    os.system("killall", "rosout")
