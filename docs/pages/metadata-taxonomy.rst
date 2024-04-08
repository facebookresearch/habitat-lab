Taxonomy of "user_defined" configurations in habitat-lab
########################################################

This resource page outlines the expected taxonomy of expected metadata fields and systems in habitat-lab leveraging the non-official "user_defined" Configuration fields for objects, stages, and scenes.

As outlined on the `Using JSON Files to configure Attributes <https://aihabitat.org/docs/habitat-sim/attributesJSON.html#user-defined-attributes>`_ doc page, "user_defined" attributes provide a generic, reserved JSON configuration node which can be filled with user data. The intent was that no "officially supported" metadata would use this field, leaving it open for arbitrary user metadata. However, several prototype and bleeding-edge features are actively leveraging this system. The purpose of this doc page is to enumerate those known uses and their taxonomy to guide further development and avoid potential conflict with ongoing/future development.


`Receptacles`_
==============

Who: Stages, RigidObjects, and ArticulatedObjects.
Where: stage_config.json, object_config.json, ao_config.json, scene_instance.json (overrides)

What: sub_config with key string containing "receptacle\_". "receptacle_mesh\_" defines a TriangleMeshReceptacle while "receptacle_aabb\_" defines a bounding box (AABB) Receptacle. See the `parse_receptacles_from_user_config <https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/datasets/rearrange/samplers/receptacle.py>`_ function.

Example:

.. code:: python

    "user_defined": {
        "receptacle_mesh_table0001_receptacle_mesh": {
            "name": "table0001_receptacle_mesh",
            "parent_object": "0a5df6da61cd2e78e972690b501452152309e56b", #handle of the parent ManagedObject's template
            "parent_link": "table0001", #if attached to an ArticulatedLink, this is the local index
            "position": [0,0,0], # position of the receptacle in parent's local space
            "rotation": [1,0,0,0],#orientation (quaternion) of the receptacle in parent's local space
            "scale": [1,1,1], #scale of the receptacles in parent's local space
            "up": [0,0,1], #up vector for the receptacle in parent's local space (for tilt culling and placement snapping)
            "mesh_filepath": "table0001_receptacle_mesh.glb" #filepath for the receptacle's mesh asset (.glb with triangulated faces expected)
        }
    }

`Scene Receptacle Filter Files`_
================================

Who: Scene Instances
Where: scene_instance.json
What: filepath (relative to dataset root directory) to the file containing Receptacle filter strings for the scene.

Example:

.. code:: python

    "user_defined": {
        "scene_filter_file": "scene_filter_files/102344022.rec_filter.json"
    }


`Object States`_
================

Who: RigidObjects and ArticulatedObjects
Where: object_config.json, ao_config.json, scene_instance.json (overrides)

What: sub_config containing any fields which pertain to the ObjectStateMachine and ObjectStateSpec logic. Exact taxonomy in flux. Consider this key reserved.

.. code:: python

    "user_defined": {
        "object_states": {

        }
    }

`Marker Sets`_
==============

Who: RigidObjects and ArticulatedObjects
Where: object_config.json, ao_config.json, scene_instance.json (overrides)

What: sub_config containing any 3D point sets which must be defined for various purposes.

.. code:: python

    "user_defined": {
        "marker_sets": {

            "handle_marker_sets":{ #these are handles for opening an ArticulatedObject's links.
                0: { # these marker sets are attached to link_id "0".
                    "handle_0": { #this is a set of 3D points.
                        0: [x,y,z] #we index because JSON needs a dict and Configuration cannot digest lists
                        1: [x,y,z]
                        2: [x,y,z]
                    },
                    ...
                },
                ...
            },

            "faucet_marker_set":{ #these are faucet points on sinks in object local space
                0: { # these marker sets are attached to link_id "0". "-1" implies base link or rigid object.
                    0: [x,y,z] #this is a faucet
                    ...
                },
                ...
            }
        }
    }

`ArticulatedObject "default link"`_
======================================

Who: ArticulatedObjects
Where: ao_config.json

What: The "default" link (integer index) is the one link which should be used if only one joint can be actuated. For example, the largest or most accessible drawer or door. Cannot be base link (-1).

.. code:: python

    "user_defined": {
        "default_link": 5 #the link id which is "default"
    }
