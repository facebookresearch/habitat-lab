"""This is an example script."""
import sys

def greet(name):
    """Return greeting."""
    return "Hello {}!".format(name)

if __name__ == "__main__":
    name = sys.argv[1]
    print(greet(name))

