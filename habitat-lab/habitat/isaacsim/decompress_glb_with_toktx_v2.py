# NOTE: You must clone the repo https://github.com/KhronosGroup/KTX-Software
'''
Example usage in terminal:

decompress_glb_with_toktx_v2.py /home/eric/projects/habitat-llm/data/fpss/objects/1/15c556c44d99689e202e4e60370995fd38e0f361.glb 
output3.glb --tool-path ../KTX-Software/build/Release/ktx 
'''

import os
import subprocess
import argparse
from pygltflib import GLTF2, Buffer, BufferView, Image
import base64
import shutil

def read_buffer_view_data_from_glb_data(glb_data, buffer_view):
    """Read data for a buffer view directly from the GLB binary data."""
    byte_offset = buffer_view.byteOffset
    byte_length = buffer_view.byteLength
    return glb_data[byte_offset:byte_offset + byte_length]

def decompress_textures_and_embed(glb_path, output_path, path_to_tool):
    # Load the original GLB file
    gltf = GLTF2().load(glb_path)
    
    # Ensure the GLB data is loaded
    if not gltf._glb_data:
        raise ValueError("No binary data found in GLTF file.")

    # Directory for temporary texture files
    temp_dir = "temp_textures"
    os.makedirs(temp_dir, exist_ok=True)

    # Collect binary data for new buffer
    new_buffer_data = bytearray(gltf._glb_data)

    # Decompress and replace each KTX2 texture that uses the KHR_texture_basisu extension
    for i, texture in enumerate(gltf.textures):
        if texture.extensions and "KHR_texture_basisu" in texture.extensions:
            print(f"Decompressing KTX2 texture {i}...")

            # Get the original KTX2 image source index from the extension
            ktx_source_index = texture.extensions["KHR_texture_basisu"]["source"]
            ktx_image = gltf.images[ktx_source_index]

            if ktx_image.uri:
                # Handle base64-encoded URI
                compressed_data = base64.b64decode(ktx_image.uri.split(",")[1])
            else:
                # Handle embedded binary buffer
                buffer_view = gltf.bufferViews[ktx_image.bufferView]
                compressed_data = read_buffer_view_data_from_glb_data(gltf._glb_data, buffer_view)

            # Save compressed data to a temporary KTX2 file
            ktx2_path = os.path.join(temp_dir, f"texture_{i}.ktx2")
            with open(ktx2_path, "wb") as f:
                f.write(compressed_data)

            # Decompress the KTX2 file to PNG using `ktx extract`
            png_path = os.path.join(temp_dir, f"texture_{i}.png")
            subprocess.run([path_to_tool, "extract", "--level", "0", ktx2_path, png_path], check=True)

            # Read PNG data and add it to the binary buffer
            with open(png_path, "rb") as f:
                decompressed_data = f.read()
            buffer_offset = len(new_buffer_data)
            new_buffer_data.extend(decompressed_data)

            # Create a new buffer view for the PNG data
            png_buffer_view = BufferView(
                buffer=0,  # We assume there's only one buffer
                byteOffset=buffer_offset,
                byteLength=len(decompressed_data)
            )
            buffer_view_index = len(gltf.bufferViews)
            gltf.bufferViews.append(png_buffer_view)

            # Update the image to use the new buffer view
            ktx_image.bufferView = buffer_view_index
            ktx_image.mimeType = "image/png"
            ktx_image.uri = None  # Clear the URI since we're embedding the data

            # Update texture to use the new image directly, removing the extension
            texture.source = ktx_source_index  # Set source to point to the new image
            texture.extensions.pop("KHR_texture_basisu", None)  # Remove the extension

    # Update the buffer with the new binary data
    gltf.buffers[0] = Buffer(byteLength=len(new_buffer_data), uri=None)
    gltf._glb_data = bytes(new_buffer_data)

    # Remove `KHR_texture_basisu` from extensions if present
    if "KHR_texture_basisu" in gltf.extensionsRequired:
        gltf.extensionsRequired.remove("KHR_texture_basisu")
    if "KHR_texture_basisu" in gltf.extensionsUsed:
        gltf.extensionsUsed.remove("KHR_texture_basisu")

    # Save the modified GLB file
    gltf.save(output_path)
    print(f"Decompressed GLB saved to {output_path}")

    # Clean up temporary files
    shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Decompress Basis-compressed textures in a GLB file by embedding them as binary data.")
    parser.add_argument("input_glb", type=str, help="Path to the input GLB file.")
    parser.add_argument("output_glb", type=str, help="Path to the output GLB file with uncompressed textures.")
    parser.add_argument("--tool-path", type=str, default="ktx", help="Path to the `ktx` tool.")

    args = parser.parse_args()

    decompress_textures_and_embed(args.input_glb, args.output_glb, args.tool_path)

if __name__ == "__main__":
    main()
