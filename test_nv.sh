#!/bin/bash

# Create a simple EGL test program.
cat << 'EOF' > egl_test.c
#include <EGL/egl.h>
#include <stdio.h>
int main() {
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        fprintf(stderr, "Failed to get EGL display.\n");
        return 1;
    }
    if (!eglInitialize(display, NULL, NULL)) {
        fprintf(stderr, "Failed to initialize EGL.\n");
        return 1;
    }
    printf("EGL is working correctly.\n");
    eglTerminate(display);
    return 0;
}
EOF
gcc -o egl_test egl_test.c -lEGL

# Check if the executable was created.
if [ ! -f ./egl_test ]; then
    echo "Failed to compile the EGL test program."
    rm egl_test.c
    exit 1
fi

# Run the test program
./egl_test
TEST_RESULT=$?

# Delete the test program.
rm -rf egl_test
rm egl_test.c

# Check the exit status
if [ $TEST_RESULT -ne 0 ]; then
    echo "Could not run simple EGL program."
    exit 1
fi

# Check if 'libEGL_nvidia.so.x' is present.
if ! ldconfig -p | grep -q "libEGL_nvidia.so."; then
    echo "Error: libEGL_nvidia.so.0 is not found. Please ensure NVIDIA drivers are installed correctly."
    exit 1
fi

# Check if /usr/share/glvnd/egl_vendor.d exists.
if [ ! -d /usr/share/glvnd/egl_vendor.d ]; then
    echo "Error: /usr/share/glvnd/egl_vendor.d directory does not exist. glvnd may be incorrectly installed."
    exit 1
fi

# Check if 10_nvidia.json exists in glvnd vendors.
if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
    echo "Error: 10_nvidia.json is not found in /usr/share/glvnd/egl_vendor.d. NVIDIA drivers may be incorrectly installed."
    exit 1
fi

# Check if 10_nvidia.json points to the libEGL_nvidia library.
if ! grep -q "libEGL_nvidia" /usr/share/glvnd/egl_vendor.d/10_nvidia.json; then
    echo "Error: 10_nvidia.json does not point to the libEGL_nvidia library. NVIDIA drivers may be incorrectly installed."
    exit 1
fi

echo "EGL test passed."
exit 0
