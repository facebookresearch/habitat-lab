from OpenGL import EGL


def initialize_egl():
    # Get the default display
    try:
        display = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
        if display == EGL.EGL_NO_DISPLAY:
            print("Failed to get EGL display.")
            return False
    except Exception as e:
        print(f"Could not get EGL display. {e}.")
        return False
    # Initialize EGL
    major, minor = EGL.EGLint(), EGL.EGLint()
    if not EGL.eglInitialize(display, major, minor):
        print("Failed to initialize EGL.")
        return False
    print(
        f"EGL initialized successfully. Version: {major.value}.{minor.value}"
    )
    # Terminate EGL
    if not EGL.eglTerminate(display):
        print("Failed to terminate EGL.")
        return False
    return True


if __name__ == "__main__":
    if initialize_egl():
        print("EGL initialization check passed.")
    else:
        print("EGL initialization check failed.")
