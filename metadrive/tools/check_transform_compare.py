"""
Standalone script (no modifications to library code) that creates a TopDownMetaDrive env,
extracts the TopDownMultiChannel observation object and computes:
 - surface-level pixel positions from vec2pix for checkpoints and ego
 - the expected display coordinates after the ObservationWindow render pipeline
It prints all intermediate values so you can confirm whether channel 2 mapping matches agent mapping.
"""
import os
import numpy as np
import math
import pprint

from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.constants import DEFAULT_AGENT


def compute_display_coords(obs_obj, surface, world_pos, ego_world_pos, heading):
    """
    Follow the ObservationWindow.render steps to map a world position -> final display pixel.
    Returns: (vec2pix_px, display_px)
    """
    # vec2pix on the surface
    vec_px = surface.vec2pix([world_pos[0], world_pos[1]])

    # vehicle pixel center on the same surface (used as position argument to render)
    pvx, pvy = surface.pos2pix(ego_world_pos[0], ego_world_pos[1])

    # Get receptive field double from obs_window (reset populates canvas_rotate size)
    # We will call reset manually to ensure sizes are set
    obs_obj.obs_window.reset(surface)
    RFdx, RFdy = obs_obj.obs_window.get_size()
    # After reset, receptive_field_double used in blit is obs_window.receptive_field_double
    # But get_size() returns canvas_rotate size which equals receptive_field_double

    # canvas_rotate will be size RFdx x RFdy
    # Coordinates of the point inside canvas_rotate before rotation
    cx = vec_px[0] - pvx + RFdx / 2
    cy = vec_px[1] - pvy + RFdy / 2

    # center-relative vector
    v = (cx - RFdx / 2, cy - RFdy / 2)
    # rotation in degrees (ObservationWindow._rotate uses rad2deg(heading) + 90)
    rotation_deg = math.degrees(heading) + 90
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    # rotate v by theta
    vx_rot = cos_t * v[0] - sin_t * v[1]
    vy_rot = sin_t * v[0] + cos_t * v[1]

    # scale used in _rotate
    canvas_uncropped = obs_obj.obs_window.canvas_uncropped
    canvas_rotate = obs_obj.obs_window.canvas_rotate
    scale = canvas_uncropped.get_size()[0] / canvas_rotate.get_size()[0]

    # apply scale
    vx_s = vx_rot * scale
    vy_s = vy_rot * scale

    # new_canvas center
    newW, newH = obs_obj.obs_window.canvas_uncropped.get_size()
    # point in new_canvas coords
    x_new = vx_s + newW / 2
    y_new = vy_s + newH / 2

    # crop to display (canvas_display)
    dispW, dispH = obs_obj.obs_window.canvas_display.get_size()
    # crop top-left in new_canvas
    left = newW / 2 - dispW / 2
    top = newH / 2 - dispH / 2
    disp_x = x_new - left
    disp_y = y_new - top

    return vec_px, (int(round(disp_x)), int(round(disp_y))), dict(
        pvx=pvx, pvy=pvy, RFdx=RFdx, RFdy=RFdy, cx=cx, cy=cy, v=v,
        rotation_deg=rotation_deg, scale=scale, newW=newW, newH=newH, dispW=dispW, dispH=dispH
    )


if __name__ == '__main__':
    cfg = dict(
        map="OO",
        num_scenarios=1,
        use_render=False,
        start_seed=123,
        frame_stack=3,
        post_stack=1,
        frame_skip=5,
        norm_pixel=True,
        resolution_size=84,
        distance=30,
    )
    env = TopDownMetaDrive(cfg)
    try:
        obs, _ = env.reset()
        obs_obj = env.observations[DEFAULT_AGENT]
        # Force a render so the map is drawn and canvases are initialized
        obs_obj.render()

        vehicle = env.engine.agents[DEFAULT_AGENT]
        nav = vehicle.navigation

        if hasattr(nav, '_get_info_for_checkpoint'):
            _, _, cp_cur = nav._get_info_for_checkpoint(lanes_id=0, ref_lane=nav.current_ref_lanes[0], ego_vehicle=vehicle)
            _, _, cp_next = nav._get_info_for_checkpoint(lanes_id=1, ref_lane=(nav.next_ref_lanes[0] if nav.next_ref_lanes is not None else nav.current_ref_lanes[0]), ego_vehicle=vehicle)
        else:
            print('Navigation has no _get_info_for_checkpoint')
            raise SystemExit(0)

        # We will use road_lines surface for mapping (same as in draw_scene)
        surface = obs_obj.canvas_road_lines.copy()

        # Also compute for ego
        raw_pos = vehicle.position

        cur_vec_px, cur_disp_px, cur_meta = compute_display_coords(obs_obj, surface, cp_cur, raw_pos, vehicle.heading_theta)
        next_vec_px, next_disp_px, next_meta = compute_display_coords(obs_obj, surface, cp_next, raw_pos, vehicle.heading_theta)
        ego_vec_px, ego_disp_px, ego_meta = compute_display_coords(obs_obj, surface, raw_pos, raw_pos, vehicle.heading_theta)

        print('\n=== Transform Comparison ===')
        print('Checkpoint current world:', cp_cur)
        print('  vec2pix on surface ->', cur_vec_px)
        print('  display coords (hand-calculated) ->', cur_disp_px)
        pprint.pprint(cur_meta)

        print('\nCheckpoint next world:', cp_next)
        print('  vec2pix on surface ->', next_vec_px)
        print('  display coords (hand-calculated) ->', next_disp_px)
        pprint.pprint(next_meta)

        print('\nEgo world:', raw_pos)
        print('  vec2pix on surface ->', ego_vec_px)
        print('  display coords (hand-calculated) ->', ego_disp_px)
        pprint.pprint(ego_meta)

        # Now compute the display coords by actually rendering and finding the pixel
        # Render the observation window into a surface
        pos = obs_obj.canvas_runtime.pos2pix(*vehicle.position)
        rendered = obs_obj.obs_window.render(surface, pos, vehicle.heading_theta)
        # save the rendered image for inspection
        out_dir = os.path.join(os.getcwd(), 'tools_debug')
        os.makedirs(out_dir, exist_ok=True)
        try:
            import pygame
            pygame.image.save(rendered, os.path.join(out_dir, 'rendered_display.png'))
            print('\nSaved rendered display to tools_debug/rendered_display.png')
        except Exception as e:
            print('Could not save rendered display:', e)

    finally:
        env.close()
