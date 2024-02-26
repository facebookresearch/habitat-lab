#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from typing import Dict

from magnum import (
    Matrix4,
    MeshPrimitive,
    Vector3,
    math,
    meshtools,
    scenetools,
    trade,
)

# Scene conversion plugins need access to image conversion plugins
importer_manager = trade.ImporterManager()
image_converter_manager = trade.ImageConverterManager()
scene_converter_manager = trade.SceneConverterManager()
scene_converter_manager.register_external_manager(image_converter_manager)

# Set up defaults for Basis import and conversion. Right now there's no
# capability of image passthrough without re-encoding, so the images have to be
# decoded to RGBA and then encoded back, which causes a slight quality loss
# every time.
importer_manager.metadata("BasisImporter").configuration["format"] = "RGBA8"
basis_configuration = image_converter_manager.metadata(
    "BasisImageConverter"
).configuration
basis_configuration["y_flip"] = False
basis_configuration["mip_gen"] = True
basis_configuration["mip_smallest_dimension"] = 4
basis_configuration["threads"] = 0

# Import / export plugins
importer = importer_manager.load_and_instantiate("AnySceneImporter")
converter = scene_converter_manager.load_and_instantiate("AnySceneConverter")

# Image resizer. Size is picked for each image individually in order to avoid
# upsampling but still target power-of-two textures.
resizer = image_converter_manager.load_and_instantiate(
    "StbResizeImageConverter"
)

# Meshoptimizer. Options set individually for each mesh.
meshoptimizer = trade.SceneConverterManager().load_and_instantiate(
    "MeshOptimizerSceneConverter"
)

# Meshoptimizer defaults. You might want to play with these, see plugin docs
# for more info: https://doc.magnum.graphics/magnum/classMagnum_1_1Trade_1_1MeshOptimizerSceneConverter.html
# This might be too harsh if you want to preserve fine details
meshoptimizer.configuration["simplifyTargetError"] = 1.0e-1
# Can avoid a certain kind of artifacts
# meshoptimizer.configuration['simplifyLockBorder'] = True
# Decimation factor. Smaller values decimate more, larger less.
the_magic_constant = 42 * 42 * 4

# glTF converter defaults. This makes it work with quantized inputs, however
# decimation will un-quantize again.
converter.configuration["textureCoordinateYFlipInMaterial"] = False
converter.configuration[
    "imageConverter"
] = "PngImageConverter"  # 'BasisKtxImageConverter'


def decimate(
    inputFile,
    outputFile,
    quiet=None,
    verbose=None,
    sloppy=False,
    simplify=True,
):
    if quiet:
        importer.flags |= trade.ImporterFlags.QUIET
        converter.flags |= trade.SceneConverterFlags.QUIET
        resizer.flags |= trade.ImageConverterFlags.QUIET
        meshoptimizer.flags |= trade.SceneConverterFlags.QUIET
    if verbose:
        importer.flags |= trade.ImporterFlags.VERBOSE
        converter.flags |= trade.SceneConverterFlags.VERBOSE
        resizer.flags |= trade.ImageConverterFlags.VERBOSE
        meshoptimizer.flags |= trade.SceneConverterFlags.VERBOSE

    importer.open_file(inputFile)
    converter.begin_file(outputFile)

    # For each mesh calculate the transformation scaling in the scene to make the
    # decimation reflect how large a particular mesh actually is there. The same
    # mesh can be referenced multiple times so it's doing a component-wise max of
    # all scales. This also ensures that meshes with quantized attributes are
    # treated equally to non-quantized.
    scene = importer.scene(0)
    mesh_assignments = scene.field(trade.SceneField.MESH)
    mesh_transformations = scenetools.absolute_field_transformations3d(
        scene, trade.SceneField.MESH
    )
    max_mesh_scaling: Dict[int, Vector3] = {}
    for mesh_id, transformation in zip(mesh_assignments, mesh_transformations):
        max_mesh_scaling[mesh_id] = math.max(
            max_mesh_scaling.get(mesh_id, Vector3(0.0)),
            math.abs(transformation.scaling()),
        )

    # Go through all meshes and decimate them
    size_before = 0
    size_after = 0
    total_source_tris = 0
    total_target_tris = 0
    total_simplified_tris = 0
    for i in range(importer.mesh_count):
        mesh = importer.mesh(i)

        # Calculate total triangle area of the *transformed* mesh. You might want
        # to fiddle with this heuristics, another option is calculating the mesh
        # AABB but that won't do the right thing for planar meshes.
        if simplify:
            # Transform the mesh to its max scale in the scene. For quantized meshes
            # this expands the position attribute to a floating-point Vector3.
            scaled_mesh = meshtools.transform3d(
                mesh, Matrix4.scaling(max_mesh_scaling.get(i, Vector3(1.0)))
            )
            if not scaled_mesh.is_indexed:
                converter.end_file()
                importer.close()
                print("Index-less meshes are not supported.")
                raise AssertionError()
            if scaled_mesh.primitive != MeshPrimitive.TRIANGLES:
                converter.end_file()
                importer.close()
                print("Non-triangle meshes are not supported.")
                raise AssertionError()
            triangle_count = scaled_mesh.index_count // 3
            total_source_tris += triangle_count

        # Perform decimation only if there's actually something, heh
        if simplify and triangle_count > 0:
            # get scaled bounding box
            positions = scaled_mesh.attribute(trade.MeshAttribute.POSITION)
            extent_min = Vector3(positions[0])
            extent_max = Vector3(positions[0])
            for pos in positions:
                extent_min = math.min(extent_min, pos)
                extent_max = math.max(extent_max, pos)

            dim = extent_max - extent_min

            target_size0 = 0.1
            target_count0 = 1000
            target_size1 = 1.0
            target_count1 = 5000
            size = (dim.x + dim.y + dim.z) / 3
            lerp_fraction = math.lerp_inverted(
                target_size0, target_size1, size
            )
            target_count = math.lerp(
                target_count0, target_count1, lerp_fraction
            )
            total_target_tris += target_count
            target = target_count / triangle_count

            # Running the simplifier only if simplification is actually desired
            if target < 1.0:
                meshoptimizer.configuration["simplify"] = simplify
                # You might want to enable this if the non-sloppy simplification fails
                # to reach the target by a wide margin
                meshoptimizer.configuration["simplifySloppy"] = sloppy  # temp
                meshoptimizer.configuration[
                    "simplifyTargetIndexCountThreshold"
                ] = target
                decimated_mesh = meshoptimizer.convert(mesh)

            # If simplification isn't desired or if it caused the mesh to disappear, run
            # just the nondestructive optimizations
            if target >= 1.0 or decimated_mesh.vertex_count == 0:
                meshoptimizer.configuration["simplify"] = False
                meshoptimizer.configuration["simplifySloppy"] = False
                decimated_mesh = meshoptimizer.convert(mesh)

            total_simplified_tris += decimated_mesh.index_count // 3

            # Stats
            size_before += len(mesh.index_data) + len(mesh.vertex_data)
            size_after += len(decimated_mesh.index_data) + len(
                decimated_mesh.vertex_data
            )
        else:
            decimated_mesh = mesh

        # Add it to the converter, preserve its name
        converter.add(decimated_mesh, name=importer.mesh_name(i))

    # Import all images, resize them, and put them to the output. This does Basis
    # conversion as well, so may take time. Use -v to see more info.
    for i in range(importer.image2d_count):
        image = importer.image2d(i)
        # Resize the image to nearest smaller power of two square, but at least 4x4
        # and at most 256x256
        nearest_smaller_power_of_two = math.clamp(
            1 << math.log2(image.size.min()), 4, 256
        )
        resizer.configuration["size"] = "{0} {0}".format(
            nearest_smaller_power_of_two
        )
        converter.add(resizer.convert(image), importer.image2d_name(i))

    # Add whatever else is there in the input file (materials, textures...) except
    # the things we're adding ourselves
    converter.add_importer_contents(
        importer, ~(trade.SceneContents.MESHES | trade.SceneContents.IMAGES2D)
    )

    # And ... done!
    converter.end_file()
    importer.close()

    return (total_source_tris, total_target_tris, total_simplified_tris)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    decimate(args.input, args.output, quiet=args.quiet, verbose=args.verbose)


if __name__ == "__main__":
    main()
