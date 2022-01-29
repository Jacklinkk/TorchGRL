"""Runner script for non-RL simulations in flow.

Usage
    python simulate.py EXP_CONFIG --no_render
"""
import argparse
import sys
import json
import os
from flow.core.experiment import Experiment

from flow.core.params import AimsunParams
from flow.utils.rllib import FlowParamsEncoder


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        description="Parse argument used when running a Flow simulation.",
        epilog="python simulate.py EXP_CONFIG --num_runs INT --no_render")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/non_rl.')

    # optional input parameters
    parser.add_argument(
        '--num_runs', type=int, default=1,
        help='Number of simulations to run. Defaults to 1.')
    parser.add_argument(
        '--no_render',
        action='store_true',
        help='Specifies whether to run the simulation during runtime.')
    parser.add_argument(
        '--aimsun',
        action='store_true',
        help='Specifies whether to run the simulation using the simulator '
             'Aimsun. If not specified, the simulator used is SUMO.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation.')

    return parser.parse_known_args(args)[0]


if __name__ == "__main__":
    flags = parse_args(sys.argv[1:])

    # Get the flow_params object.
    module = __import__("exp_configs.non_rl", fromlist=[flags.exp_config])
    flow_params = getattr(module, flags.exp_config).flow_params

    # Get the custom callables for the runner.
    if hasattr(getattr(module, flags.exp_config), "custom_callables"):
        callables = getattr(module, flags.exp_config).custom_callables
    else:
        callables = None

    flow_params['sim'].render = not flags.no_render
    flow_params['simulator'] = 'aimsun' if flags.aimsun else 'traci'

    # If Aimsun is being called, replace SumoParams with AimsunParams.
    if flags.aimsun:
        sim_params = AimsunParams()
        sim_params.__dict__.update(flow_params['sim'].__dict__)
        flow_params['sim'] = sim_params

    # Specify an emission path if they are meant to be generated.
    if flags.gen_emission:
        flow_params['sim'].emission_path = "./data"

        # Create the flow_params object
        fp_ = flow_params['exp_tag']
        dir_ = flow_params['sim'].emission_path
        with open(os.path.join(dir_, "{}.json".format(fp_)), 'w') as outfile:
            json.dump(flow_params, outfile,
                      cls=FlowParamsEncoder, sort_keys=True, indent=4)

    # Create the experiment object.
    exp = Experiment(flow_params, callables)

    # Run for the specified number of rollouts.
    exp.run(flags.num_runs, convert_to_csv=flags.gen_emission)
