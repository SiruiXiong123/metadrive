from collections import deque

import gymnasium as gym
import math
import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.component.lane.point_lane import PointLane
from metadrive.constants import Decoration, DEFAULT_AGENT
from metadrive.obs.top_down_obs import TopDownObservation
from metadrive.obs.top_down_obs_impl import WorldSurface, COLOR_BLACK, ObjectGraphics, LaneGraphics, \
    ObservationWindowMultiChannel
from metadrive.utils import import_pygame, clip
from metadrive.type import MetaDriveType
from metadrive.constants import PGDrivableAreaProperty

from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.navigation_module.edge_network_navigation import EdgeNetworkNavigation
from metadrive.component.navigation_module.trajectory_navigation import TrajectoryNavigation

pygame = import_pygame()
COLOR_WHITE = pygame.Color("white")
DEFAULT_TRAJECTORY_LANE_WIDTH = 4

def world_to_bev(world_pos, ego_pos, ego_heading, scaling, resolution):
        diff = world_pos - ego_pos                   # 相对 ego
        diff = (diff[0] * scaling, diff[1] * scaling)  # 缩放到像素
        diff = (diff[1], diff[0])                      # 坐标系适配

        p = pygame.math.Vector2(diff)
        p = p.rotate(np.rad2deg(ego_heading) + 90)     # 对齐 ego 方向
        p = (p[1], p[0])                               # 交换轴
        p = (
            clip(p[0] + resolution[0] / 2, -resolution[0], resolution[0]),
            clip(p[1] + resolution[1] / 2, -resolution[1], resolution[1])
        )
        return p

class TopDownMultiChannel(TopDownObservation):
    """
    Most of the source code is from Highway-Env, we only optimize and integrate it in MetaDrive
    See more information on its Github page: https://github.com/eleurent/highway-env
    """
    RESOLUTION = (100, 100)  # pix x pix
    MAP_RESOLUTION = (2000, 2000)  # pix x pix
    # MAX_RANGE = (50, 50)  # maximum detection distance = 50 M

    # CHANNEL_NAMES = ["road_network", "traffic_flow", "target_vehicle", "navigation", "past_pos"]
    # Add a separate channel that contains road lines only (no drivable area fill)
    # Keep channel list consistent with ObservationWindowMultiChannel. We intentionally
    # limit the BEV to two channels (road_network, road_lines) — past_pos is disabled.
    CHANNEL_NAMES = ["road_network", "road_lines"]

    def __init__(
        self,
        vehicle_config,
        onscreen,
        clip_rgb: bool,
        frame_stack: int = 5,
        post_stack: int = 5,
        frame_skip: int = 5,
        resolution=None,
        max_distance=50
    ):
        super(TopDownMultiChannel, self).__init__(
            vehicle_config, clip_rgb, onscreen=onscreen, resolution=resolution, max_distance=max_distance
        )
        #self.num_stacks = 2 + frame_stack
        # now we have: road_network, road_lines -> 2 channels
        self.num_stacks = 2
        self.stack_traffic_flow = deque([], maxlen=(frame_stack - 1) * frame_skip + 1)
        self.stack_past_pos = deque(
            [], maxlen=(post_stack - 1) * frame_skip + 1
        )  # In the coordination of target vehicle
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self._should_fill_stack = True
        # Temporary debug flag: when True, verify checkpoint vec2pix mapping by
        # inspecting pixels on the per-scene road_lines surface and saving
        # annotated debug images on mismatch. Turn off after debugging.
        self.debug_chk = True
        self.max_distance = max_distance
        self.scaling = self.resolution[0] / max_distance
        assert self.scaling == self.resolution[1] / self.max_distance

    def init_obs_window(self):
        # ObservationWindowMultiChannel expects the list of channel names that will
        # be provided in canvas_dict when rendering. We use the CHANNEL_NAMES directly.
        self.obs_window = ObservationWindowMultiChannel(self.CHANNEL_NAMES.copy(), (self.max_distance, self.max_distance), self.resolution)

    def init_canvas(self):
        self.canvas_background = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_navigation = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_road_network = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        # canvas that only draws lane center/lines, not the drivable area
        self.canvas_road_lines = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_runtime = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_ego = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        # past_pos channel intentionally disabled; keep the surface unused for now
        self.canvas_past_pos = pygame.Surface(self.resolution)  # A local view (unused)

    def reset(self, env, vehicle=None):
        # self.engine = env.engine
        self.road_network = env.current_map.road_network
        self.target_vehicle = vehicle
        self._should_draw_map = True
        self._should_fill_stack = True
    


    def draw_map(self):
        """
        :return: a big map surface, clip  and rotate to use a piece of it
        """
        # Setup the maximize size of the canvas
        # scaling and center can be easily found by bounding box
        b_box = self.road_network.get_bounding_box()
        self.canvas_navigation.fill(COLOR_BLACK)
        self.canvas_ego.fill(COLOR_BLACK)
        self.canvas_road_network.fill(COLOR_BLACK)
        # clear road_lines before drawing so it only contains lines
        self.canvas_road_lines.fill(COLOR_BLACK)
        self.canvas_runtime.fill(COLOR_BLACK)
        self.canvas_background.fill(COLOR_BLACK)
        self.canvas_background.set_colorkey(self.canvas_background.BLACK)
        x_len = b_box[1] - b_box[0]
        y_len = b_box[3] - b_box[2]
        max_len = max(x_len, y_len) + 20  # Add more 20 meters
        scaling = self.MAP_RESOLUTION[1] / max_len - 0.1
        assert scaling > 0

        # real-world distance * scaling = pixel in canvas
        self.canvas_background.scaling = scaling
        self.canvas_runtime.scaling = scaling
        self.canvas_navigation.scaling = scaling
        self.canvas_ego.scaling = scaling
        self.canvas_road_network.scaling = scaling
        # make sure road_lines uses the same scaling and centering so lines are visible
        self.canvas_road_lines.scaling = scaling

        centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
        self.canvas_runtime.move_display_window_to(centering_pos)
        self.canvas_navigation.move_display_window_to(centering_pos)
        self.canvas_ego.move_display_window_to(centering_pos)
        self.canvas_background.move_display_window_to(centering_pos)
        self.canvas_road_network.move_display_window_to(centering_pos)
        self.canvas_road_lines.move_display_window_to(centering_pos)

        if isinstance(self.target_vehicle.navigation, NodeNetworkNavigation):
            self.draw_navigation_node(self.canvas_background, (255, 0, 0))
        elif isinstance(self.target_vehicle.navigation, EdgeNetworkNavigation):
            # TODO: draw edge network navigation
            pass
        elif isinstance(self.target_vehicle.navigation, TrajectoryNavigation):
            self.draw_navigation_trajectory(self.canvas_background, (0, 255, 0))

        if isinstance(self.road_network, NodeRoadNetwork):
            for _from in self.road_network.graph.keys():
                decoration = True if _from == Decoration.start else False
                for _to in self.road_network.graph[_from].keys():
                    for l in self.road_network.graph[_from][_to]:
                        two_side = True if l is self.road_network.graph[_from][_to][-1] or decoration else False
                        LaneGraphics.LANE_LINE_WIDTH = 0.5
                        # draw filled drivable area to background
                        LaneGraphics.display(l, self.canvas_background, two_side)
                        # ensure road_lines is clean then draw only lane lines to road_lines canvas
                        # draw lines in white and slightly thicker for visibility in the separate channel
                        LaneGraphics.display(l, self.canvas_road_lines, two_side, use_line_color=False, color=(255, 255, 255))
        elif hasattr(self.engine, "map_manager"):
            for data in self.engine.map_manager.current_map.blocks[-1].map_data.values():
                if ScenarioDescription.POLYLINE in data:
                    LaneGraphics.display_scenario_line(
                        data[ScenarioDescription.POLYLINE], data[ScenarioDescription.TYPE], self.canvas_background
                    )
                    # also draw lines-only for scenario data into road_lines as white thicker lines
                    poly = data[ScenarioDescription.POLYLINE]
                    # choose skipping consistent with display_scenario_line
                    if MetaDriveType.is_broken_line(data[ScenarioDescription.TYPE]):
                        points_to_skip = math.floor(PGDrivableAreaProperty.STRIPE_LENGTH * 2 / 2) * 2
                    else:
                        points_to_skip = 1
                    for index in range(0, len(poly) - 1, points_to_skip):
                        if index + 1 < len(poly):
                            s_p = poly[index]
                            e_p = poly[index + 1]
                            pygame.draw.line(
                                self.canvas_road_lines,
                                (255, 255, 255),
                                self.canvas_road_lines.vec2pix([s_p[0], s_p[1]]),
                                self.canvas_road_lines.vec2pix([e_p[0], e_p[1]]),
                                max(1, self.canvas_road_lines.pix(PGDrivableAreaProperty.LANE_LINE_WIDTH) * 2)
                            )

        self.canvas_road_network.blit(self.canvas_background, (0, 0))
        # road_lines already has lane line drawings
        # Normalize road_lines surface: set any non-black pixel to pure white so left/right boundaries have consistent brightness
        try:
            import pygame.surfarray as surfarray
            arr = surfarray.pixels3d(self.canvas_road_lines)
            # any non-black channel -> set to 255
            mask = (arr.sum(axis=2) > 0)
            arr[mask] = 255
            del arr
        except Exception:
            # fallback: do nothing if surfarray is unavailable
            pass
        # road_lines already contains only line drawings (we drew lines into canvas_road_lines directly)
        self.obs_window.reset(self.canvas_runtime)
        self._should_draw_map = False

    def _refresh(self, canvas, pos, clip_size):
        canvas.set_clip((pos[0] - clip_size[0] / 2, pos[1] - clip_size[1] / 2, clip_size[0], clip_size[1]))
        canvas.fill(COLOR_BLACK)

    def draw_scene(self):
        # Set the active area that can be modify to accelerate
        assert len(self.engine.agents) == 1, "Don't support multi-agent top-down observation yet!"
        vehicle = self.engine.agents[DEFAULT_AGENT]
        pos = self.canvas_runtime.pos2pix(*vehicle.position)
        # print("Ego position (world coords):", vehicle.position, "-> (pix coords):", pos)

        clip_size = (int(self.obs_window.get_size()[0] * 1.1), int(self.obs_window.get_size()[0] * 1.1))

        # self._refresh(self.canvas_ego, pos, clip_size)
        self._refresh(self.canvas_runtime, pos, clip_size)
    # past_pos channel disabled (no per-step drawing)
    # self.canvas_past_pos.fill(COLOR_BLACK)
        # self._draw_ego_vehicle()

        # Draw vehicles
        # TODO PZH: I hate computing these in pygame-related code!!!
        ego_heading = vehicle.heading_theta
        ego_heading = ego_heading if abs(ego_heading) > 2 * np.pi / 180 else 0

        # current vehicle world position (used for ego marker drawing)
        raw_pos = vehicle.position

        for v in self.engine.get_objects(lambda o: isinstance(o, BaseVehicle) or isinstance(o, BaseTrafficParticipant)
                                         ).values():
            if v is vehicle:
                continue
            h = v.heading_theta
            h = h if abs(h) > 2 * np.pi / 180 else 0
            ObjectGraphics.display(object=v, surface=self.canvas_runtime, heading=h, color=ObjectGraphics.BLUE)

        # past_pos tracking/drawing disabled — we intentionally omit this channel
        # raw_pos = vehicle.position
        # self.stack_past_pos.append(raw_pos)
        # (no drawing into self.canvas_past_pos)

        # default road_lines surface for this scene (we'll use a copy so per-frame
        # markers (ego, nav) don't permanently modify the persistent canvas)
        try:
            road_lines_for_scene = self.canvas_road_lines.copy()
        except Exception:
            # fallback to reference if copy() unavailable
            road_lines_for_scene = self.canvas_road_lines

    # Debug print disabled: avoid noisy repeated logging during rendering
    # print("Navigation type:", type(self.target_vehicle.navigation))

        # Navigation visualization into channel2 is drawn onto a per-frame copy
        # (road_lines_for_scene). This avoids permanently modifying
        # self.canvas_road_lines and thus prevents residual/ghosted agent paths.

        # Draw navigation markers and lines onto the road_lines channel when enabled.
        # This uses vehicle.navigation.get_checkpoints() (preferred) or falls back to
        # navigation.checkpoints if needed. It respects vehicle config flags
        # show_navi_mark and show_line_to_navi_mark.
        try:
            nav = self.target_vehicle.navigation
            cfg = getattr(self.target_vehicle, "config", {}) or {}
            show_ckpt = cfg.get("show_navi_mark", False)
            show_line = cfg.get("show_line_to_navi_mark", False)

            ckpt1 = ckpt2 = None
            try:
                # prefer API that returns world coordinates
                ckpt1, ckpt2 = nav.get_checkpoints()
            except Exception:
                # fallback: if checkpoints is a list of world coords (trajectory), try that
                try:
                    if hasattr(nav, "checkpoints") and len(nav.checkpoints) > 0 and isinstance(nav.checkpoints[0], (list, tuple, np.ndarray)):
                        ckpt1 = nav.checkpoints[0]
                        ckpt2 = nav.checkpoints[1] if len(nav.checkpoints) > 1 else nav.checkpoints[-1]
                except Exception:
                    ckpt1 = ckpt2 = None

            def world_to_pix(surface, world_pos):
                p = surface.vec2pix([world_pos[0], world_pos[1]])
                return int(round(p[0])), int(round(p[1]))

            # draw checkpoint markers (use a slightly larger marker with dark outline
            # and bright inner color for better contrast in BEV channel)
            # NOTE: if show_line is enabled we intentionally skip checkpoint markers
            # so that only the navigation line is visible when the user requests it.
            if show_ckpt and ckpt1 is not None and not show_line:
                try:
                    # larger radius in world pixels scaled to surface
                    r_outer = max(3, int(round(self.canvas_road_lines.pix(1.0))))
                except Exception:
                    r_outer = 4
                r_inner = max(2, r_outer - 1)
                try:
                    x1, y1 = world_to_pix(road_lines_for_scene, ckpt1)
                    # dark outline
                    pygame.draw.circle(road_lines_for_scene, (0, 0, 0), (x1, y1), r_outer)
                    # bright inner fill (yellow) for visibility
                    pygame.draw.circle(road_lines_for_scene, (255, 255, 0), (x1, y1), r_inner)
                except Exception:
                    pass
                try:
                    if ckpt2 is not None:
                        x2, y2 = world_to_pix(road_lines_for_scene, ckpt2)
                        pygame.draw.circle(road_lines_for_scene, (0, 0, 0), (x2, y2), max(3, r_outer - 1))
                        pygame.draw.circle(road_lines_for_scene, (255, 200, 0), (x2, y2), max(2, r_inner - 1))
                except Exception:
                    pass

            # draw line(s) from ego to first checkpoint and optionally to the next checkpoint
            if show_line and ckpt1 is not None:
                try:
                    ego_pix = world_to_pix(road_lines_for_scene, self.target_vehicle.position)
                    ck1_pix = world_to_pix(road_lines_for_scene, ckpt1)
                    # draw a darker thicker line as outline for contrast
                    # increase width by 1.5x to make navigation lines more visible
                    base_outline = int(round(self.canvas_road_lines.pix(0.8)))
                    base_inner = int(round(self.canvas_road_lines.pix(0.4)))
                    outline_w = max(2, int(round(base_outline * 1.5)))
                    inner_w = max(1, int(round(base_inner * 1.5)))
                    pygame.draw.line(road_lines_for_scene, (0, 0, 0), ego_pix, ck1_pix, outline_w)
                    pygame.draw.line(road_lines_for_scene, (255, 220, 0), ego_pix, ck1_pix, inner_w)
                    if ckpt2 is not None:
                        ck2_pix = world_to_pix(road_lines_for_scene, ckpt2)
                        pygame.draw.line(road_lines_for_scene, (0, 0, 0), ck1_pix, ck2_pix, outline_w)
                        pygame.draw.line(road_lines_for_scene, (255, 220, 0), ck1_pix, ck2_pix, inner_w)
                except Exception:
                    pass
        except Exception:
            # navigation not present or not ready; skip gracefully
            pass

        # Now render the observation windows using the possibly-updated road_lines surface
        # Draw ego/world vehicle position onto the road_lines copy so it appears in channel 1
        try:
            # raw_pos is vehicle.position in world coords
            ego_world_pix = road_lines_for_scene.vec2pix([raw_pos[0], raw_pos[1]])
            ex = int(round(ego_world_pix[0]))
            ey = int(round(ego_world_pix[1]))
            # large marker with black border and white center for strong contrast
            # Compute marker size from vehicle physical top-down dimensions (meters -> pixels)
            try:
                w_pix = max(3, int(round(vehicle.top_down_width * self.scaling)))
                l_pix = max(3, int(round(vehicle.top_down_length * self.scaling)))
                # choose the smaller vehicle dimension as marker size and clamp to [3, 7]
                square_size = int(max(3, min(min(w_pix, l_pix), 7)))
            except Exception:
                # fallback fixed smaller marker
                square_size = 5
            half = square_size // 2
            outer = (ex - half, ey - half, square_size, square_size)
            inner = (ex - half + 1, ey - half + 1, max(1, square_size - 2), max(1, square_size - 2))
            try:
                # draw black outer square then white inner square
                pygame.draw.rect(road_lines_for_scene, (0, 0, 0), outer)
                pygame.draw.rect(road_lines_for_scene, (255, 255, 255), inner)
            except Exception:
                # fallback: single white pixel
                road_lines_for_scene.fill((255, 255, 255), ((ex, ey), (1, 1)))
            # Also draw a circular halo marker to make ego position more visible
            try:
                halo_r = max(4, square_size // 3)
                # black border then larger white inner circle on the per-scene road_lines
                pygame.draw.circle(road_lines_for_scene, (0, 0, 0), (ex, ey), halo_r + 2)
                pygame.draw.circle(road_lines_for_scene, (255, 255, 255), (ex, ey), halo_r + 1)
                # Add an additive semi-transparent white halo to make the marker visually brighter
                try:
                    size = int((halo_r + 4) * 2)
                    halo_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                    c = (size // 2, size // 2)
                    pygame.draw.circle(halo_surf, (255, 255, 255, 140), c, halo_r + 2)
                    surf_pos = (int(ex - c[0]), int(ey - c[1]))
                    try:
                        road_lines_for_scene.blit(halo_surf, surf_pos, special_flags=pygame.BLEND_ADD)
                    except Exception:
                        road_lines_for_scene.blit(halo_surf, surf_pos)
                except Exception:
                    pass
            except Exception:
                pass
            # Intentionally do NOT draw ego marker onto the global road_network canvas.
            # This keeps channel 1 (road_network) free of ego-position markers.
            pass
        except Exception:
            pass

        # Render only the two persistent canvases we keep in the BEV
        ret = self.obs_window.render(
            canvas_dict=dict(
                road_network=self.canvas_road_network,
                road_lines=road_lines_for_scene,
            ),
            position=pos,
            heading=vehicle.heading_theta
        )
        return ret

    def get_observation_window(self):
        # Return whatever the observation window provides for configured channels
        return self.obs_window.get_observation_window()

    def _transform(self, img):
        # img = np.mean(img, axis=-1)
        # Use Atari-like processing

        # img = img[..., 0]
        # img = np.dot(img[..., :], [0.299, 0.587, 0.114])
        img = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114

        if self.norm_pixel:
            img = img.astype(np.float32) / 255
        else:
            img = img.astype(np.uint8)
        return img

    def observe(self, vehicle: BaseVehicle):
        self.render()
        surface_dict = self.get_observation_window()
        # road_network and road_lines are larger internal surfaces; ensure they are scaled to observation resolution
        if "road_network" in surface_dict:
            surface_dict["road_network"] = pygame.transform.smoothscale(surface_dict["road_network"], self.resolution)
        if "road_lines" in surface_dict:
            surface_dict["road_lines"] = pygame.transform.smoothscale(surface_dict["road_lines"], self.resolution)
        img_dict = {k: pygame.surfarray.array3d(surface) for k, surface in surface_dict.items()}

        # Gray scale
        img_dict = {k: self._transform(img) for k, img in img_dict.items()}

        # if self._should_fill_stack:
        #     self.stack_past_pos.clear()
        #     self.stack_traffic_flow.clear()
        #     for _ in range(self.stack_traffic_flow.maxlen):
        #         self.stack_traffic_flow.append(img_dict["traffic_flow"])
        #     self._should_fill_stack = False
        # self.stack_traffic_flow.append(img_dict["traffic_flow"])

        # Build BEV image channels without traffic_flow
        img = [
            img_dict["road_network"] * 2,
            # new channel: road_lines (only lines, no drivable area fill)
            img_dict.get("road_lines", np.zeros_like(img_dict["road_network"])),
        ]  # past_pos and traffic_flow omitted intentionally

        # Stacked traffic flow
        # stacked = np.zeros_like(img_navigation)
        indices = self._get_stack_indices(len(self.stack_traffic_flow))
        # for i in reversed(indices):
        #     stacked = self.stack_traffic_flow[i] + stacked / 2
        # if self.norm_pixel:
        #     stacked = np.clip(stacked, 0.0, 1.0)
        # else:
        #     stacked = np.clip(stacked, 0, 255)
        # for i in indices:
        #     img.append(self.stack_traffic_flow[i])

        # Stack
        img = np.stack(img, axis=2)
        if self.norm_pixel:
            img = np.clip(img, 0, 1.0)
        else:
            img = np.clip(img, 0, 255)
        return np.transpose(img, (1, 0, 2))

    def draw_navigation_node(self, canvas, color=(255, 0, 0)): #color=(255, 0, 0)
        checkpoints = self.target_vehicle.navigation.checkpoints
        # print("Checkpoints:", checkpoints)
        for i, c in enumerate(checkpoints[:-1]):
            lanes = self.road_network.graph[c][checkpoints[i + 1]]
            # print("  Number of lanes in this segment:", len(lanes))
            for lane in lanes:
                # print("   Lane ID:", getattr(lane, "index", None))
                # print("   Lane length:", lane.length)
                # print("   Lane width:", lane.width)
                LaneGraphics.draw_drivable_area(lane, canvas, color=color)

    def draw_navigation_trajectory(self, canvas, color=(255, 0, 0)): #color=(255, 0, 0)
        lane = PointLane(self.target_vehicle.navigation.checkpoints, DEFAULT_TRAJECTORY_LANE_WIDTH)
        LaneGraphics.draw_drivable_area(lane, canvas, color=color)

    def _get_stack_indices(self, length, frame_skip=None):
        frame_skip = frame_skip or self.frame_skip
        num = int(math.ceil(length / frame_skip))
        indices = []
        for i in range(num):
            indices.append(length - 1 - i * frame_skip)
        return indices

    @property
    def observation_space(self):
        shape = self.obs_shape + (self.num_stacks, )
        if self.norm_pixel:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)