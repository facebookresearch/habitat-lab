import numpy as np
import os
import subprocess
from glob import glob
import copy
import json

root = '/Users/soyeonm/Downloads/objectnav_hm3d_v2 3'
moveto = '/Users/soyeonm/Downloads/rearrange_hm3d_v2_temp_example'
scene_dir = '/Users/soyeonm/Downloads/hm3d-example-glb-v0.2' #'/Users/soyeonm/Documents/SocialNavigation/habitat-lab_soyeonm/data/scene_datasets/hm3d_v0.2'
if not(os.path.exists(moveto)):
	os.makedirs(moveto)

splits = ["train", "val", "val_mini"]

def remove_keys_from_episde_except(episode_file, except_keys):
	episode_file_copy = copy.deepcopy(episode_file)
	for ep_i, ep in enumerate(episode_file['episodes']):
		for k in ep:
			if not(k in except_keys):
			#if k in drop_keys:
				episode_file_copy['episodes'][ep_i].pop(k)
	return episode_file_copy


def remove_keys_from_episde(episode_file, drop_keys):
	episode_file_copy = copy.deepcopy(episode_file)
	for ep_i, ep in enumerate(episode_file['episodes']):
		for k in ep:
			#if not(k in except_keys):
			if k in drop_keys:
				episode_file_copy['episodes'][ep_i].pop(k)
	return episode_file_copy

def add_keys(episode_file):
	#add 'ao_states', 'rigid_objs', 'targets'
	for ep_i, ep in enumerate(episode_file['episodes']):
		ep['ao_states'] = {}
		ep['rigid_objs'] = [['003_cracker_box.object_config.json', [[0.96581, 0.06349, 0.25134, -10.73403], [-0.05398, 0.99755, -0.04455, 0.62308], [-0.25355, 0.02946, 0.96687, 1.17491], [0.0, 0.0, 0.0, 1.0]]], ['025_mug.object_config.json', [[0.8904, 0.02247, 0.45461, -1.24393], [-0.02209, 0.99974, -0.00616, 0.63543], [-0.45463, -0.00455, 0.89067, -4.43167], [0.0, 0.0, 0.0, 1.0]]], ['005_tomato_soup_can.object_config.json', [[0.86322, 0.01795, 0.5045, -10.98469], [-0.04138, 0.99852, 0.03527, 1.01082], [-0.50312, -0.05133, 0.86269, -4.25872], [0.0, 0.0, 0.0, 1.0]]], ['005_tomato_soup_can.object_config.json', [[-0.4269, -0.06663, -0.90184, -4.75562], [-0.02482, 0.99777, -0.06197, 0.41843], [0.90396, -0.00407, -0.4276, -3.85932], [0.0, 0.0, 0.0, 1.0]]], ['024_bowl.object_config.json', [[-0.97898, 0.00012, 0.20398, -0.24359], [0.00054, 1.0, 0.00205, 0.52648], [-0.20398, 0.00212, -0.97897, -3.52276], [0.0, 0.0, 0.0, 1.0]]]] #[]
		ep['targets'] = {'003_cracker_box_:0000': [[-0.90545, 0.0, -0.42446, -0.19425], [0.0, 1.0, 0.0, 0.60769], [0.42446, 0.0, -0.90545, -3.02012], [0.0, 0.0, 0.0, 1.0]], '025_mug_:0000': [[0.24046, 0.0, -0.97066, -9.20009], [0.0, 1.0, 0.0, 0.50237], [0.97066, 0.0, 0.24046, 1.98839], [0.0, 0.0, 0.0, 1.0]]} #{}
		ep['info'] = {'object_labels': {'003_cracker_box_:0000': 'any_targets|0', '025_mug_:0000': 'any_targets|1'}} #{'object_labels': {}} #{'object_labels': {'003_cracker_box_:0000': 'any_targets|0', '025_mug_:0000': 'any_targets|1'}}
	return episode_file


def check_if_scene_exists(episode_file):
	count = 0
	new_ep = []
	scene_dir_globs = glob(scene_dir + '/*')
	scene_dir_scenes = [g.split('/')[-1] for g in scene_dir_globs]

	for ep_i, ep in enumerate(episode_file['episodes']):
		scene_name = ep['scene_id'].split('/')[-2]
		#Dont check, just say that it is 
		#scene_dir_scenes = [g.split('/')[-1] for g in glob(scene_dir + '/*')]
		#Just change scene_name to the first one
		ep['scene_id'] = ep['scene_id'].replace(ep['scene_id'].split('/')[-2], scene_dir_scenes[0])
		ep['scene_id'] = ep['scene_id'].replace(ep['scene_id'].split('/')[-1], scene_dir_scenes[0].split('-')[-1] + ".basis.glb")
		ep['scene_id'] =  'data/scene_datasets/'+ ep['scene_id']
		#breakpoint()
		scene_name = ep['scene_id'].split('/')[-2]
		if scene_name in scene_dir_scenes:
			count +=1
			new_ep.append(ep)
	return new_ep



for split in splits:
	content_folder = os.path.join(root, split, "content")
	for g in glob(os.path.join(content_folder, "*")):
		#unzip 
		if g[-3:] == '.gz':
			subprocess.run(["gunzip", g])
		#open this json
		json_g = g.replace('.gz', '')
		episode_file = json.load(open(json_g, 'r'))
		#Keep 'episode_id', 'scene_id', 'scene_dataset_config', 'start_position', 'start_rotation' and ditch the rest
		except_keys = set(['episode_id', 'scene_id', 'scene_dataset_config', 'start_position', 'start_rotation', 'info'])#set{['episode_id', 'scene_id', 'scene_dataset_config', 'start_position', 'start_rotation']}
		#episode_file = remove_keys_from_episde(episode_file, ['goals', 'start_room'])
		new_episodes = check_if_scene_exists(episode_file)
		if new_episodes !=[]:
			episode_file['episodes'] = new_episodes
			episode_file = remove_keys_from_episde_except(episode_file, except_keys)
			episode_file = add_keys(episode_file)
			#Save this 
			if not(os.path.exists(os.path.dirname(json_g.replace(root, moveto)))):
				os.makedirs(os.path.dirname(json_g.replace(root, moveto)))
			json.dump(episode_file, open(json_g.replace(root, moveto), 'w'))
			subprocess.run(["gzip", json_g.replace(root, moveto)])

