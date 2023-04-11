import os
import string

import magnum as mn
from magnum import shaders, text

DEFAULT_FONT_PATH = "./fonts/ProggyClean.ttf"

# the maximum number of chars displayable in the app window
# using the magnum text module
MAX_DISPLAY_TEXT_CHARS = 1024

# how much to displace window text relative to the center of the
# app window (e.g if you want the display text in the top left of
# the app window, you will displace the text
# window width * -TEXT_DELTA_FROM_CENTER in the x axis and
# window height * TEXT_DELTA_FROM_CENTER in the y axis, as the text
# position defaults to the middle of the app window)
TEXT_DELTA_FROM_CENTER = 0.49

# font size of the magnum in-window display text
DISPLAY_FONT_SIZE = 16.0


class TextDrawer:
    def __init__(
        self,
        framebuffer_size: mn.Vector2i,
        relative_path_to_font: str = DEFAULT_FONT_PATH,
        max_display_text_chars: int = MAX_DISPLAY_TEXT_CHARS,
        text_delta_from_center: float = TEXT_DELTA_FROM_CENTER,
        display_font_size: float = DISPLAY_FONT_SIZE,
    ) -> None:
        self._framebuffer_size = framebuffer_size

        # Load a TrueTypeFont plugin and open the font file
        self._display_font = text.FontManager().load_and_instantiate(
            "TrueTypeFont"
        )
        self._display_font.open_file(
            os.path.join(os.path.dirname(__file__), relative_path_to_font),
            13,
        )

        # Glyphs we need to render everything
        self._glyph_cache = text.GlyphCache(mn.Vector2i(256))
        self._display_font.fill_glyph_cache(
            self._glyph_cache,
            string.ascii_lowercase
            + string.ascii_uppercase
            + string.digits
            + ":-_+,.! %µ",
        )

        # magnum text object that displays CPU/GPU usage data in the app window
        self._window_text = text.Renderer2D(
            self._display_font,
            self._glyph_cache,
            display_font_size,
            text.Alignment.TOP_LEFT,
        )
        self._window_text.reserve(max_display_text_chars)

        # text object transform in window space is Projection matrix times Translation Matrix
        # put text in top left of window
        self._window_text_transform = mn.Matrix3.projection(
            self._framebuffer_size
        ) @ mn.Matrix3.translation(
            mn.Vector2(self._framebuffer_size)
            * mn.Vector2(
                -text_delta_from_center,
                text_delta_from_center,
            )
        )
        self._shader = shaders.VectorGL2D()

        # make magnum text background transparent
        mn.gl.Renderer.enable(mn.gl.Renderer.Feature.BLENDING)
        mn.gl.Renderer.set_blend_function(
            mn.gl.Renderer.BlendFunction.ONE,
            mn.gl.Renderer.BlendFunction.ONE_MINUS_SOURCE_ALPHA,
        )
        mn.gl.Renderer.set_blend_equation(
            mn.gl.Renderer.BlendEquation.ADD, mn.gl.Renderer.BlendEquation.ADD
        )

    def draw_text(self, text_to_draw):
        self._shader.bind_vector_texture(self._glyph_cache.texture)
        self._shader.transformation_projection_matrix = (
            self._window_text_transform
        )
        self._shader.color = [1.0, 1.0, 1.0]
        self._window_text.render(text_to_draw)
        self._shader.draw(self._window_text.mesh)
