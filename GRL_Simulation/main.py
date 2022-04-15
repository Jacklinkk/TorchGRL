from flow.core.params import VehicleParams, InFlows, SumoCarFollowingParams, SumoParams, EnvParams, InitialConfig, \
    NetParams, SumoLaneChangeParams, TrafficLightParams
from flow.controllers import IDMController, RLController
from controller import SpecificMergeRouter, NearestMergeRouter
from network import HighwayRampsNetwork, ADDITIONAL_NET_PARAMS

# ----------- Configurations -----------#
TRAINING = True
# TRAINING = False

# TESTING = True
TESTING = False

# DEBUG = True
DEBUG = False

RENDER = False
# RENDER = True

NEAREST_MERGE = False
# NEAREST_MERGE = True

NUM_HUMAN = 20
actual_num_human = 20

NUM_MERGE_0 = 10
NUM_MERGE_1 = 10

MAX_AV_SPEED = 14
MAX_HV_SPEED = 10

VEH_COLORS = ['red', 'red'] if NEAREST_MERGE else ['red', 'green']

#######################################################


Router = NearestMergeRouter if NEAREST_MERGE else SpecificMergeRouter

vehicles = VehicleParams()
vehicles.add(veh_id="human",
             lane_change_params=SumoLaneChangeParams('only_strategic_safe'),
             car_following_params=SumoCarFollowingParams(speed_mode='right_of_way', min_gap=5, tau=1,
                                                         max_speed=MAX_HV_SPEED),
             acceleration_controller=(IDMController, {}),
             routing_controller=(Router, {}),
             )

vehicles.add(veh_id="merge_0",
             lane_change_params=SumoLaneChangeParams('no_cooperative_safe'),
             car_following_params=SumoCarFollowingParams(speed_mode='no_collide', min_gap=1, tau=1,
                                                         max_speed=MAX_AV_SPEED),
             acceleration_controller=(RLController, {}),
             routing_controller=(Router, {}),
             color=VEH_COLORS[0])

vehicles.add(veh_id="merge_1",
             lane_change_params=SumoLaneChangeParams('no_cooperative_safe'),
             car_following_params=SumoCarFollowingParams(speed_mode='no_collide', min_gap=1, tau=1,
                                                         max_speed=MAX_AV_SPEED),
             acceleration_controller=(RLController, {}),
             routing_controller=(Router, {}),
             color=VEH_COLORS[1])

initial_config = InitialConfig(spacing='uniform')

inflow = InFlows()
inflow.add(veh_type="human",
           edge="highway_0",
           probability=0.2,
           depart_lane='random',
           depart_speed='random',
           route='routehighway_0_0',
           number=actual_num_human)

inflow.add(veh_type="merge_0",
           edge="highway_0",
           probability=0.1,
           depart_lane='random',
           depart_speed='random',
           route='routehighway_0_0',
           number=NUM_MERGE_0)

inflow.add(veh_type="merge_1",
           edge="highway_0",
           probability=0.1,
           depart_lane='random',
           depart_speed='random',
           route='routehighway_0_0',
           number=NUM_MERGE_1)

sim_params = SumoParams(sim_step=0.1, restart_instance=True, render=RENDER)

from specific_environment import MergeEnv

intention_dic = {"human": 0, "merge_0": 1, "merge_1": 1} if NEAREST_MERGE else {"human": 0, "merge_0": 1, "merge_1": 2}
terminal_edges = ['off_ramp_0', 'off_ramp_1', 'highway_2']

env_params = EnvParams(warmup_steps=100,
                       additional_params={"intention": intention_dic,
                                          "max_av_speed": MAX_AV_SPEED,
                                          "max_hv_speed": MAX_HV_SPEED})

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params['num_vehicles'] = NUM_HUMAN + NUM_MERGE_0 + NUM_MERGE_1
additional_net_params['num_cav'] = NUM_MERGE_0 + NUM_MERGE_1
additional_net_params['num_hv'] = NUM_HUMAN
additional_net_params['terminal_edges'] = terminal_edges

net_params = NetParams(inflows=inflow, additional_params=additional_net_params)

traffic_lights = TrafficLightParams()

network = HighwayRampsNetwork("highway_ramp", vehicles, net_params, initial_config, traffic_lights)


# ----------- Model Building -----------#
flow_params = dict(
    exp_tag='test_network',
    env_name=MergeEnv,
    network=network,
    simulator='traci',
    sim=sim_params,
    env=env_params,
    net=net_params,
    veh=vehicles,
    initial=initial_config,
    tls=traffic_lights
)

# number of time steps
flow_params['env'].horizon = 2500

from Experiment.DQN_experiments import Experiment

exp = Experiment(flow_params)
# run the sumo simulation
exp.run(num_runs=1, training=TRAINING,
        testing=TESTING,
        num_human=NUM_HUMAN,
        actual_num_human=actual_num_human,
        num_cav=(NUM_MERGE_0 + NUM_MERGE_1),
        model='GRL',
        debug=DEBUG)
