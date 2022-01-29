from flow.controllers.base_routing_controller import BaseRouter


# class SpecificMergeRouter(BaseRouter):
#     """docstring for ClassName"""
#     def choose_route(self, env):
#         """See parent class.

#         Adopt one of the current edge's routes if about to leave the network.
#         """

#         veh_type = env.k.vehicle.get_type(self.veh_id)
#         current_lane = env.k.vehicle.get_lane(self.veh_id)
#         current_edge = env.k.vehicle.get_edge(self.veh_id)
#         current_route = env.k.vehicle.get_route(self.veh_id)

#         if len(current_route) == 0:
#             return None

#         elif veh_type == 'merge_0' and current_edge == 'highway_0' and current_lane == 0:
#             return env.available_routes[current_edge][1][0]

#         elif current_edge == current_route[-1]: # reach the end of current edge
#             if veh_type.split('_')[0] == 'merge' and current_edge.split('_')[0] != 'ramp':
#                 if current_edge == env.available_routes[veh_type][0][0][-2] and current_lane==0:
#                     return env.available_routes[current_edge][1][0]
#             return env.available_routes[current_edge][0][0]

#         return None

class SpecificMergeRouter(BaseRouter):

    """docstring for ClassName"""



    def choose_route(self, env):
        """See parent class.

        Adopt one of the current edge's routes if about to leave the network.
        """

        veh_type = env.k.vehicle.get_type(self.veh_id)
        current_lane = env.k.vehicle.get_lane(self.veh_id)
        current_edge = env.k.vehicle.get_edge(self.veh_id)
        current_route = env.k.vehicle.get_route(self.veh_id)


        if len(current_route) == 0:
            return None

        if veh_type == 'merge_0' and current_edge=='highway_0' and current_lane == 0:
            route = env.available_routes[current_edge][1][0]
        elif veh_type == 'merge_1' and current_edge == 'highway_1' and current_lane == 0:
            route = env.available_routes[current_edge][1][0]
        elif current_edge == 'highway_0' or current_edge == 'highway_1':
            route = env.available_routes[current_edge][0][0]

        else:
            route =  None


        return route


class NearestMergeRouter(BaseRouter):
    def choose_route(self, env):
        veh_type = env.k.vehicle.get_type(self.veh_id)
        current_lane = env.k.vehicle.get_lane(self.veh_id)
        current_edge = env.k.vehicle.get_edge(self.veh_id)
        current_route = env.k.vehicle.get_route(self.veh_id)

        if len(current_route) == 0:
            return None

        if veh_type.split('_')[0] == 'merge' and (current_edge == 'highway_0' or current_edge == 'highway_1') and current_lane == 0:
            return env.available_routes[current_edge][1][0]
        elif current_edge == 'highway_0' or current_edge == 'highway_1':
            return env.available_routes[current_edge][0][0]
        else:
            return None




