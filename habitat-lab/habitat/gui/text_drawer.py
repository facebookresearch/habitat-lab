import os
import string
from enum import Enum
from typing import List, Tuple

import magnum as mn
from magnum import shaders, text

DEFAULT_FONT_PATH = "./fonts/ProggyClean.ttf"

# the maximum number of chars displayable in the app window
# using the magnum text module
MAX_DISPLAY_TEXT_CHARS = 512

# how much to displace window text relative to the center of the
# app window (e.g if you want the display text in the top left of
# the app window, you will displace the text
# window width * -TEXT_DELTA_FROM_CENTER in the x axis and
# window height * TEXT_DELTA_FROM_CENTER in the y axis, as the text
# position defaults to the middle of the app window)
TEXT_DELTA_FROM_CENTER = 0.49

# font size of the magnum in-window display text
DISPLAY_FONT_SIZE = 16.0


class TextOnScreenAlignment(Enum):
    TOP_LEFT = (TEXT_DELTA_FROM_CENTER, -TEXT_DELTA_FROM_CENTER)
    TOP_CENTER = (TEXT_DELTA_FROM_CENTER, 0)
    # We don't yet have right-alignment based on text width, so as a workaround we
    # use an x offset halfway between the center and the right side of the viewport.
    TOP_RIGHT = (TEXT_DELTA_FROM_CENTER, TEXT_DELTA_FROM_CENTER / 2)
    CENTER = (0, 0)
    # When using BOTTOM*, always include a small positive text_delta_y (~10), otherwise
    # your text will appear too close to the bottom of the screen.
    BOTTOM_LEFT = (-TEXT_DELTA_FROM_CENTER, -TEXT_DELTA_FROM_CENTER)
    BOTTOM_CENTER = (-TEXT_DELTA_FROM_CENTER, 0)
    BOTTOM_RIGHT = (-TEXT_DELTA_FROM_CENTER, TEXT_DELTA_FROM_CENTER / 2)


class TextDrawer:
    def __init__(
        self,
        framebuffer_size: mn.Vector2i,
        relative_path_to_font: str = DEFAULT_FONT_PATH,
        max_display_text_chars: int = MAX_DISPLAY_TEXT_CHARS,
        display_font_size: float = DISPLAY_FONT_SIZE,
    ) -> None:
        self._text_transform_pairs: List[Tuple[str, mn.Matrix3]] = []
        self._framebuffer_size = framebuffer_size

        # Load a TrueTypeFont plugin and open the font file
        self._display_font = text.FontManager().load_and_instantiate(
            "TrueTypeFont"
        )
        self._display_font.open_file(
            os.path.join(os.path.dirname(__file__), relative_path_to_font),
            13,
        )
        self._display_font_size = display_font_size
        self._max_display_text_chars = max_display_text_chars

        # Glyphs we need to render everything
        self._glyph_cache = text.GlyphCache(mn.Vector2i(256))
        self._display_font.fill_glyph_cache(
            self._glyph_cache,
            string.ascii_lowercase
            + string.ascii_uppercase
            + string.digits
            + ":-_+,.! %µ",
        )
        self._shader = shaders.VectorGL2D()
        self._window_text = text.Renderer2D(
            self._display_font,
            self._glyph_cache,
            display_font_size,
            text.Alignment.TOP_LEFT,
        )
        self._window_text.reserve(self._max_display_text_chars)

    def add_text(
        self,
        text_to_add,
        alignment: TextOnScreenAlignment = TextOnScreenAlignment.TOP_LEFT,
        text_delta_x: int = 0,
        text_delta_y: int = 0,
    ):
        """
        Adds `text_to_add` and corresponding window text transform to `self._text_transform_pairs`.
            :param text_to_add: text to be added to `self._text_transform_pairs` for drawing
            :param alignment: window text anchor of type TextOnScreenAlignment, defaults to top left corner
            :param text_delta_x: pixels delta to move/adjust window text anchor along X axis,
            :param text_delta_y: pixels delta to move/adjust window text anchor along Y axis,
        """

        # text object transform in window space is Projection matrix times Translation Matrix
        # put text in top left of window
        align_y, align_x = alignment.value
        window_text_transform = mn.Matrix3.projection(
            self._framebuffer_size
        ) @ mn.Matrix3.translation(
            mn.Vector2(self._framebuffer_size) * mn.Vector2(align_x, align_y)
            + mn.Vector2(text_delta_x, text_delta_y)
        )
        self._text_transform_pairs.append((text_to_add, window_text_transform))

    def draw_text(self):
        # make magnum text background transparent
        mn.gl.Renderer.enable(mn.gl.Renderer.Feature.BLENDING)
        mn.gl.Renderer.set_blend_function(
            mn.gl.Renderer.BlendFunction.ONE,
            mn.gl.Renderer.BlendFunction.ONE_MINUS_SOURCE_ALPHA,
        )
        mn.gl.Renderer.set_blend_equation(
            mn.gl.Renderer.BlendEquation.ADD, mn.gl.Renderer.BlendEquation.ADD
        )

        """Draws collected text on the screen"""
        self._shader.bind_vector_texture(self._glyph_cache.texture)
        for text_to_draw, transform in self._text_transform_pairs:
            self._shader.transformation_projection_matrix = transform
            self._shader.color = [1.0, 1.0, 1.0]
            self._window_text.render(text_to_draw)
            self._shader.draw(self._window_text.mesh)
        self._text_transform_pairs.clear()

        # sloppy: disable blending here so that other rendering subsystems (e.g.
        # DebugLineRender) will be forced to enable and configure blending when they
        # render
        mn.gl.Renderer.disable(mn.gl.Renderer.Feature.BLENDING)
