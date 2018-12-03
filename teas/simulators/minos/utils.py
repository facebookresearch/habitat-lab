import argparse
import csv

from easydict import EasyDict
from minos.config import sim_config
from minos.config.sim_args import add_sim_args, read_lines


def minos_args(config):
    """r This is a wrapper function for arguments to be passed to MINOS
    simulator
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--navmap', action='store_true', default=False,
                        help='Use navigation map')
    add_sim_args(parser)
    args = parser.parse_args([])  # parse empty argument input

    if len(args.scene_ids) == 1:
        if args.scene_ids[0].endswith('txt'):
            # Read scene ids from file
            args.scene_ids = read_lines(args.scene_ids[0])
        elif args.scene_ids[0].endswith('csv'):
            # Read scene ids from file
            csvfile = args.scene_ids[0]
            with open(csvfile) as f:
                reader = csv.DictReader(f)
                args.scene_ids = [r.get('id') for r in reader]

    if args.depth_noise:
        args.sensors = [{'name': 'depth', 'noise': True}]
    args.observations = {'color': True, 'depth': args.depth,
                         'forces': args.forces, 'audio': args.audio}
    for s in args.sensors:
        args.observations[s] = True
    args.collision_detection = {'mode': args.collision_mode}
    if args.add_object_at_goal:
        # print('add object at goal')
        args.modifications = [{
            'name': 'add',
            'modelIds': 'p5d.s__1957',
            'format': 'obj',
            'positionAt': 'goal'
        }]

    args.audio = {'debug': args.debug, 'debug_memory': args.debug_audio_memory}
    args.actionTraceLogFields = ['forces']
    args.auto_start = not args.manual_start
    if not args.auto_start:
        args.audio = {'port': 1112}
        args.port = 4899

    sim_args = sim_config.get(args.env_config, vars(args))
    sim_args = EasyDict(sim_args)

    # setting parameters from config
    params = vars(sim_args)
    for k, v in config.items():
        params[k] = v

    return params
