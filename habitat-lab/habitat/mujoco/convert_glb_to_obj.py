import trimesh
import pygltflib
import numpy as np
import os

def quaternion_to_matrix(q):
    """Convert a quaternion (x, y, z, w) into a 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

def get_node_transform(node):
    """Compute the transformation matrix for a given node."""
    T = np.eye(4)  # Identity matrix

    translation = np.array(node.translation) if hasattr(node, 'translation') and node.translation else np.zeros(3)
    rotation = quaternion_to_matrix(node.rotation) if hasattr(node, 'rotation') and node.rotation else np.eye(3)
    scale = np.diag(node.scale) if hasattr(node, 'scale') and node.scale else np.eye(3)

    T[:3, :3] = rotation @ scale
    T[:3, 3] = translation
    return T

def extract_vertices(accessor, buffer_data, gltf):
    """Extract vertex data from an interleaved buffer while ensuring correct stride and offsets."""
    buffer_view = gltf.bufferViews[accessor.bufferView]
    byte_offset = buffer_view.byteOffset + accessor.byteOffset
    stride = buffer_view.byteStride if buffer_view.byteStride else 12  # Default stride if not set
    dtype = np.float32 if accessor.componentType == 5126 else np.float16

    raw_data = np.frombuffer(buffer_data, dtype=dtype, offset=byte_offset)
    vertex_data = np.zeros((accessor.count, 3), dtype=dtype)

    for i in range(accessor.count):
        start_idx = i * (stride // dtype().nbytes)  # Convert stride to element count
        vertex_data[i] = raw_data[start_idx:start_idx + 3]

    return vertex_data

def traverse_scene(node_idx, parent_transform, gltf, buffer_data, meshes):
    """Recursively traverse the scene graph and apply transformations to meshes."""
    node = gltf.nodes[node_idx]
    node_transform = parent_transform @ get_node_transform(node)

    if node.mesh is not None:
        mesh = gltf.meshes[node.mesh]
        for primitive in mesh.primitives:
            if primitive.attributes.POSITION is None:
                continue

            position_accessor = gltf.accessors[primitive.attributes.POSITION]
            try:
                vertices = extract_vertices(position_accessor, buffer_data, gltf)
            except ValueError as e:
                print(f"Warning: {e}")
                continue

            indices = None
            if primitive.indices is not None:
                index_accessor = gltf.accessors[primitive.indices]
                index_buffer_view = gltf.bufferViews[index_accessor.bufferView]
                index_offset = index_buffer_view.byteOffset + index_accessor.byteOffset
                index_dtype = np.uint16 if index_accessor.componentType == 5123 else np.uint32
                indices = np.frombuffer(buffer_data, dtype=index_dtype, count=index_accessor.count, offset=index_offset).reshape(-1, 3)

            mesh = trimesh.Trimesh(vertices=vertices, faces=indices)
            meshes.append(mesh)

    if hasattr(node, "children"):
        for child_idx in node.children:
            traverse_scene(child_idx, node_transform, gltf, buffer_data, meshes)

def load_glb_meshes(glb_path):
    """Load all meshes from a GLTF/GLB file and apply hierarchical transformations."""
    gltf = pygltflib.GLTF2().load(glb_path)

    buffer_uri = gltf.buffers[0].uri
    if buffer_uri and not buffer_uri.startswith("data:"):
        buffer_path = os.path.join(os.path.dirname(glb_path), buffer_uri)
        with open(buffer_path, "rb") as f:
            buffer_data = f.read()
    else:
        buffer_data = gltf.binary_blob()

    meshes = []
    
    # Handle missing `scene` index by defaulting to 0
    scene_idx = gltf.scene if gltf.scene is not None else 0
    root_nodes = gltf.scenes[scene_idx].nodes if gltf.scenes else []

    for root_node in root_nodes:
        traverse_scene(root_node, np.eye(4), gltf, buffer_data, meshes)

    return meshes

def export_obj(meshes, output_path):
    """Merge all meshes into one and export to OBJ format."""
    combined_mesh = trimesh.util.concatenate(meshes)
    combined_mesh.export(output_path)

def convert_glb_to_obj(input_path, output_path):
    """Convert a GLB/GLTF file to an OBJ file, applying all transformations."""
    meshes = load_glb_meshes(input_path)
    if not meshes:
        raise ValueError(f"No meshes found in {input_path}")

    export_obj(meshes, output_path)
    print(f"Exported to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python convert_glb_to_obj.py input.gltf output.obj")
        sys.exit(1)

    input_gltf = sys.argv[1]
    output_obj = sys.argv[2]

    convert_glb_to_obj(input_gltf, output_obj)
