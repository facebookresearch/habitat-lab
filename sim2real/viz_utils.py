import cv2
import numpy as np


def generate_text_image(width: int, text: str) -> np.ndarray:
    """
    Generates an image of the given text with line breaks, honoring given width.

    Args:
        width (int): Width of the image.
        text (str): Text to be drawn.

    Returns:
        np.ndarray: Text drawn on white image with the given width.
    """
    # Define the parameters for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    line_spacing = 10  # Spacing between lines in pixels

    # Calculate the maximum width and height of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    max_width = width - 20  # Allow some padding
    max_height = text_size[1] + line_spacing

    # Split the text into words
    words = text.split()

    # Initialize variables for text positioning
    x = 10
    y = text_size[1]

    to_draw = []

    # Iterate over the words and add them to the image
    num_rows = 1
    for word in words:
        # Get the size of the word
        word_size, _ = cv2.getTextSize(word, font, font_scale, font_thickness)

        # Check if adding the word exceeds the maximum width
        if x + word_size[0] > max_width:
            # Add a line break before the word
            y += max_height
            x = 10
            num_rows += 1

        # Draw the word on the image
        to_draw.append((word, x, y))

        # Update the position for the next word
        x += word_size[0] + 5  # Add some spacing between words

    # Create a blank white image with the calculated dimensions
    image = 255 * np.ones((max_height * num_rows, width, 3), dtype=np.uint8)
    for word, x, y in to_draw:
        cv2.putText(
            image,
            word,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    return image


def add_text_to_image(
    image: np.ndarray, text: str, top: bool = False
) -> np.ndarray:
    """
    Adds text to the given image.

    Args:
        image (np.ndarray): Input image.
        text (str): Text to be added.
        top (bool, optional): Whether to add the text to the top or bottom of the image.

    Returns:
        np.ndarray: Image with text added.
    """
    width = image.shape[1]
    text_image = generate_text_image(width, text)
    if top:
        combined_image = np.vstack([text_image, image])
    else:
        combined_image = np.vstack([image, text_image])

    return combined_image
