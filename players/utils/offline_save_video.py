from __future__ import annotations

import os
import yaml
import imageio
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence
from matplotlib.patches import Ellipse, Polygon, Rectangle
from matplotlib.transforms import Affine2D

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class VehicleState:
    """Store one vehicle pose for offline rendering."""

    x: float
    y: float
    heading: float


class OfflineRaceVideoRenderer:
    """Render offline multi-vehicle race videos with telemetry, a global mini-map, and a track-aligned local BEV."""

    def __init__(
        self,
        map_yaml_path: str,
        refline_xy: np.ndarray,
        output_fps: int = 20,
        figure_size: Tuple[float, float] = (14.0, 10.0),
        dpi: int = 120,
        vehicle_length: float = 0.58,
        vehicle_width: float = 0.31,
        vehicle_ref_offset: float = 0.0,
        ego_color: str = "#CC0000",
        opp_color: str = "#0500C3",
        refline_color: str = "#3A7D44",
        obstacle_gray: float = 0.85,
        free_gray: float = 1.0,
        show_heading_line: bool = True,
        heading_line_scale: float = 0.35,
        axis_margin: float = 1.5,
        fixed_axis: bool = False,
        background_cache: bool = True,
        follow_ego: bool = True,
        camera_window: Tuple[float, float] = (25.0, 25.0),
        bev_window: Tuple[float, float] = (12.0, 12.0),
        history_seconds: float = 0.5,
        hud_speed_max: float = 8.0,
        steer_limit: float = 0.4,
        show_ego_hud: bool = True,
        boundary_offsets: Optional[np.ndarray] = None,
        hud_panel_fraction: float = 0.28,
        vehicle_sprite_dir: Optional[str] = None,
        use_vehicle_sprites: bool = True,
        vehicle_sprite_scale: float = 1.0,
        sprite_forward_angle_deg: float = 45.0,
    ) -> None:
        """Initialize the renderer state, map assets, and HUD layout."""
        self.map_yaml_path = map_yaml_path
        self.refline_xy = np.asarray(refline_xy, dtype=np.float32)
        self.output_fps = int(output_fps)
        self.figure_size = figure_size
        self.dpi = int(dpi)

        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)
        self.vehicle_ref_offset = float(vehicle_ref_offset)

        # Current vehicle drawing uses vehicle_colors below rather than these public color fields.
        self.ego_color = ego_color
        self.opp_color = opp_color
        self.refline_color = refline_color
        # Current map drawing uses fixed display colors in _load_map rather than these tone fields.
        self.obstacle_gray = float(obstacle_gray)
        self.free_gray = float(free_gray)

        self.show_heading_line = bool(show_heading_line)
        self.heading_line_scale = float(heading_line_scale)
        self.axis_margin = float(axis_margin)
        self.fixed_axis = bool(fixed_axis)
        # The current capture path redraws each scene and does not restore this optional cache.
        self.background_cache = bool(background_cache)
        self.follow_ego = bool(follow_ego)
        self.camera_window = tuple(camera_window)
        self.bev_window = tuple(bev_window)
        self.history_seconds = float(history_seconds)
        self.hud_speed_max = float(hud_speed_max)
        self.steer_limit = float(steer_limit)
        self.show_ego_hud = bool(show_ego_hud)
        self.boundary_offsets = boundary_offsets
        self.hud_panel_fraction = float(np.clip(hud_panel_fraction, 0.18, 0.42))
        self.vehicle_sprite_dir = vehicle_sprite_dir
        self.use_vehicle_sprites = bool(use_vehicle_sprites)
        self.vehicle_sprite_scale = float(vehicle_sprite_scale)
        self.sprite_forward_angle_deg = float(sprite_forward_angle_deg)

        # Current history drawing reads _vehicle_histories while the two buffers below are only reset.
        self._ego_history: List[Tuple[float, float, float]] = []
        self._opp_history: List[Tuple[float, float, float]] = []
        self._vehicle_histories: List[List[Tuple[float, float, float]]] = []
        self._last_sim_time: float = 0.0

        # Match each role-indexed sprite with the color used by its history, trajectory, and HUD marker.
        self.vehicle_colors = [
            "#EE0808",  # ego.png: red
            "#0101CF",  # opp1.png: blue
            "#FE7E02",  # opp2.png: orange
            "#02AA63",  # opp3.png: green
            "#D856A0",  # opp4.png: pink
            "#FEEA04",  # opp5.png: yellow
            "#54BDFD",  # opp6.png: light blue
            "#730DE3",  # opp7_purple.png: purple
            "#11E9DD",  # opp8_cyan.png: cyan
            "#545454",  # opp9_charcoal.png: charcoal
            "#B4F704",  # opp10_lime.png: lime
            "#744F2F",  # opp11_bronze.png: bronze
        ]
        self.vehicle_sprite_files = [
            "agent_assets/ego.png",
            "agent_assets/opp1.png",
            "agent_assets/opp2.png",
            "agent_assets/opp3.png",
            "agent_assets/opp4.png",
            "agent_assets/opp5.png",
            "agent_assets/opp6.png",
            "agent_assets/opp7_purple.png",
            "agent_assets/opp8_cyan.png",
            "agent_assets/opp9_charcoal.png",
            "agent_assets/opp10_lime.png",
            "agent_assets/opp11_bronze.png",
        ]
        self.vehicle_role_names = [
            "ego",
            "opp1",
            "opp2",
            "opp3",
            "opp4",
            "opp5",
            "opp6",
            "opp7",
            "opp8",
            "opp9",
            "opp10",
            "opp11",
        ]

        # This original text tone is retained but is not referenced by the current theme.
        original_text_gray = (127 / 255.0, 127 / 255.0, 127 / 255.0, 127 / 255.0)
        original_green = (0.0, 1.0, 0.0, 1.0)
        self.theme = {
            "hud_bg": "#F6F7F9",
            "hud_panel": "#F6F7F9",
            "hud_panel_edge": "#DADFE6",
            "hud_text": "gray",
            "hud_muted": "black",
            "hud_grid": "black",
            "hud_green": original_green,
            "hud_green_soft": original_green,
            "hud_green_dim": self.refline_color,
            "hud_gray": "gray",
            "hud_dark": "#F6F7F9",
            "canvas_bg": "#F6F7F9",
            "seam_color": "#E6EAF0",
        }

        self.frames: List[np.ndarray] = []

        self._map_img: Optional[np.ndarray] = None
        self._map_extent: Optional[Tuple[float, float, float, float]] = None
        self._axis_limits: Optional[Tuple[float, float, float, float]] = None
        self._fig = None
        self._ax = None
        self._hud_ax = None
        self._mini_map_ax = None
        self._fpv_ax = None
        self._background_rgba = None
        # Current rendering clears HUD axes wholesale and does not populate this reserved list.
        self._hud_artists: List[object] = []
        self._vehicle_sprites: Dict[int, np.ndarray] = {}

        self.left_boundary_xy: Optional[np.ndarray] = None
        self.right_boundary_xy: Optional[np.ndarray] = None
        if self.boundary_offsets is not None and self.refline_xy is not None:
            self._compute_track_boundaries()

        self._load_vehicle_sprites()
        self._load_map()
        self._prepare_scene()

    def _ensure_vehicle_history_size(self, num_vehicles: int) -> None:
        """Grow the per-vehicle history lists to match the active fleet size."""
        while len(self._vehicle_histories) < int(num_vehicles):
            self._vehicle_histories.append([])

    def _get_vehicle_color(self, vehicle_idx: int) -> str:
        """Return the display color assigned to one vehicle index."""
        return self.vehicle_colors[int(vehicle_idx) % len(self.vehicle_colors)]

    def _get_vehicle_role_name(self, vehicle_idx: int) -> str:
        """Return the display role name assigned to one vehicle index."""
        if int(vehicle_idx) < len(self.vehicle_role_names):
            return self.vehicle_role_names[int(vehicle_idx)]
        return f"agent_{int(vehicle_idx)}"

    def _default_vehicle_sprite_dir(self) -> str:
        """Return the directory used to look up vehicle sprite PNG files."""
        if self.vehicle_sprite_dir is not None:
            return self.vehicle_sprite_dir
        try:
            return os.path.dirname(os.path.abspath(__file__))
        except NameError:
            return os.getcwd()

    def _load_vehicle_sprites(self) -> None:
        """Load and cache role-indexed vehicle sprites from the configured asset directory."""
        self._vehicle_sprites = {}
        if not self.use_vehicle_sprites:
            return

        sprite_dir = self._default_vehicle_sprite_dir()
        for vehicle_idx, file_name in enumerate(self.vehicle_sprite_files):
            sprite_path = os.path.join(sprite_dir, file_name)
            if not os.path.exists(sprite_path):
                continue
            try:
                raw_sprite = imageio.imread(sprite_path)
                prepared_sprite = self._prepare_vehicle_sprite_image(raw_sprite)
            except Exception:
                # Preserve silent polygon fallback when an optional sprite cannot be loaded.
                prepared_sprite = None
            if prepared_sprite is not None:
                self._vehicle_sprites[int(vehicle_idx)] = prepared_sprite

    def _crop_sprite_to_alpha(self, rgba: np.ndarray, threshold: float = 0.02) -> np.ndarray:
        """Crop transparent margins from one RGBA sprite image."""
        alpha = rgba[..., 3]
        ys, xs = np.where(alpha > float(threshold))
        if xs.size == 0 or ys.size == 0:
            return rgba
        pad = 2
        x0 = max(int(xs.min()) - pad, 0)
        x1 = min(int(xs.max()) + pad + 1, rgba.shape[1])
        y0 = max(int(ys.min()) - pad, 0)
        y1 = min(int(ys.max()) + pad + 1, rgba.shape[0])
        return rgba[y0:y1, x0:x1, :]

    def _prepare_vehicle_sprite_image(self, raw_sprite: np.ndarray) -> Optional[np.ndarray]:
        """Convert one vehicle sprite to cropped canonical RGBA with transparent background."""
        sprite = np.asarray(raw_sprite)
        if sprite.ndim != 3 or sprite.shape[2] not in (3, 4):
            return None

        sprite = sprite.astype(np.float32)
        if sprite.max() > 1.0:
            sprite = sprite / 255.0

        rgb = sprite[..., :3]
        if sprite.shape[2] == 4:
            alpha = sprite[..., 3].copy()
        else:
            alpha = np.ones(rgb.shape[:2], dtype=np.float32)

        # Remove near-white sprite backgrounds while preserving non-white highlights.
        white_mask = np.all(rgb > 0.955, axis=2)
        near_white = np.all(rgb > 0.905, axis=2)
        softness = np.clip((0.955 - np.min(rgb, axis=2)) / 0.050, 0.0, 1.0)
        alpha[white_mask] = 0.0
        alpha[near_white] *= softness[near_white]

        rgba = np.dstack([rgb, alpha]).astype(np.float32)
        rgba = self._crop_sprite_to_alpha(rgba)

        if Image is not None:
            image = Image.fromarray(np.clip(rgba * 255.0, 0, 255).astype(np.uint8), mode="RGBA")
            resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
            image = image.rotate(-float(self.sprite_forward_angle_deg), resample=resample, expand=True)
            rgba = np.asarray(image).astype(np.float32) / 255.0
            rgba = self._crop_sprite_to_alpha(rgba)

        if rgba.shape[0] < 2 or rgba.shape[1] < 2:
            return None
        return rgba

    def _get_vehicle_sprite(self, vehicle_idx: int) -> Optional[np.ndarray]:
        """Return a cached sprite for one vehicle index when available."""
        return self._vehicle_sprites.get(int(vehicle_idx))

    def _draw_vehicle_sprite_on_axis(
        self,
        ax,
        vehicle_idx: int,
        center_x: float,
        center_y: float,
        heading_angle: float,
        length: float,
        width: float,
        zorder: int,
        alpha: float = 1.0,
    ) -> bool:
        """Draw one cached vehicle sprite on an axis in data coordinates."""
        sprite = self._get_vehicle_sprite(vehicle_idx)
        if sprite is None:
            return False

        sprite_length = float(length) * float(self.vehicle_sprite_scale)
        sprite_width = float(width) * float(self.vehicle_sprite_scale)
        transform = Affine2D().rotate(float(heading_angle)).translate(float(center_x), float(center_y)) + ax.transData

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.imshow(
            sprite,
            extent=[-0.5 * sprite_length, 0.5 * sprite_length, -0.5 * sprite_width, 0.5 * sprite_width],
            origin="upper",
            interpolation="bilinear",
            transform=transform,
            zorder=zorder,
            alpha=float(alpha),
            clip_on=True,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return True

    def capture_multi(
        self,
        vehicle_states: List[VehicleState],
        sim_time: Optional[float] = None,
        title: Optional[str] = None,
        vehicle_trajs: Optional[List[Optional[np.ndarray]]] = None,
        ego_goal_grid: Optional[np.ndarray] = None,
        extra_text: Optional[str] = None,
        render_info: Optional[Dict[str, float]] = None,
        follow_vehicle_idx: int = 0,
    ) -> None:
        """Render and store one multi-vehicle frame while retaining the unused title argument for compatibility."""
        if self._fig is None or self._ax is None:
            self._prepare_scene()

        self._redraw_static_scene()

        num_vehicles = len(vehicle_states)
        self._ensure_vehicle_history_size(num_vehicles)
        if vehicle_trajs is None:
            vehicle_trajs = [None] * num_vehicles

        if sim_time is not None:
            self._last_sim_time = float(sim_time)
            cutoff_time = float(sim_time) - self.history_seconds
            for i, state in enumerate(vehicle_states):
                self._vehicle_histories[i].append((float(sim_time), float(state.x), float(state.y)))
                self._vehicle_histories[i] = [p for p in self._vehicle_histories[i] if p[0] >= cutoff_time]

        self._draw_vehicle_histories(num_vehicles)
        self._draw_vehicle_trajs(vehicle_trajs)
        self._draw_goal_grid(ego_goal_grid)
        self._set_camera(vehicle_states, follow_vehicle_idx)

        for i, state in enumerate(vehicle_states):
            self._draw_vehicle(state, color=self._get_vehicle_color(i), label=f"agent_{i}", vehicle_idx=i)

        if self.show_ego_hud:
            self._draw_ego_hud(render_info=render_info, vehicle_states=vehicle_states, follow_vehicle_idx=follow_vehicle_idx)

        if extra_text is not None:
            self._ax.text(
                0.02,
                0.96,
                extra_text,
                transform=self._ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                color="#222222",
                zorder=40,
            )

        # Keep title as a compatibility-only argument and render no overlay.

        self.frames.append(self._figure_to_rgb())
        self._clear_dynamic_artists()

    def _draw_vehicle_histories(self, num_vehicles: int) -> None:
        """Draw short trajectory histories for all visible vehicles."""
        for i in range(num_vehicles):
            hist = self._vehicle_histories[i]
            if len(hist) <= 1:
                continue
            hist_xy = np.array([[p[1], p[2]] for p in hist], dtype=np.float32)
            self._ax.plot(
                hist_xy[:, 0],
                hist_xy[:, 1],
                linestyle="-",
                linewidth=3.0 if i == 0 else 2.0,
                color=self._get_vehicle_color(i),
                alpha=0.9 if i == 0 else 0.7,
                zorder=12,
            )

    def _draw_vehicle_trajs(self, vehicle_trajs: Sequence[Optional[np.ndarray]]) -> None:
        """Draw only the optional planned trajectory supplied for vehicle index zero."""
        for i, traj in enumerate(vehicle_trajs):
            if i != 0 or traj is None or len(traj) == 0:
                continue
            traj = np.asarray(traj)
            self._ax.scatter(traj[:, 0], traj[:, 1], s=9, color=self._get_vehicle_color(i), alpha=0.55, zorder=13)

    def _draw_goal_grid(self, ego_goal_grid: Optional[np.ndarray]) -> None:
        """Draw the ego goal sample grid when debug points are provided."""
        if ego_goal_grid is None or len(ego_goal_grid) == 0:
            return
        ego_goal_grid = np.asarray(ego_goal_grid)
        self._ax.scatter(
            ego_goal_grid[:, 0],
            ego_goal_grid[:, 1],
            s=10,
            c=self._get_vehicle_color(0),
            alpha=0.35,
            marker="o",
            linewidths=0,
            zorder=11,
        )

    def _set_camera(self, vehicle_states: Sequence[VehicleState], follow_vehicle_idx: int) -> None:
        """Set followed-vehicle camera limits when enabled and thereby override limits established during the static redraw."""
        if not self.follow_ego or len(vehicle_states) == 0:
            return
        follow_idx = int(np.clip(follow_vehicle_idx, 0, len(vehicle_states) - 1))
        follow_state = vehicle_states[follow_idx]
        half_w = 0.5 * float(self.camera_window[0])
        half_h = 0.5 * float(self.camera_window[1])
        self._ax.set_xlim(follow_state.x - half_w, follow_state.x + half_w)
        self._ax.set_ylim(follow_state.y - half_h, follow_state.y + half_h)

    def _compute_track_boundaries(self) -> None:
        """Compute left and right track boundaries from the centerline and lateral offsets."""
        centerline = np.asarray(self.refline_xy, dtype=np.float32)
        offsets = np.asarray(self.boundary_offsets, dtype=np.float32)

        if centerline.ndim != 2 or centerline.shape[1] != 2:
            raise ValueError("refline_xy must have shape (N, 2)")
        if offsets.shape[0] != 2 or offsets.shape[1] != centerline.shape[0]:
            raise ValueError("boundary_offsets must have shape (2, N)")

        dx = np.gradient(centerline[:, 0])
        dy = np.gradient(centerline[:, 1])
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
        tx = dx / norm
        ty = dy / norm
        nx = -ty
        ny = tx

        d_right = offsets[0]
        d_left = offsets[1]

        left_boundary = np.zeros_like(centerline)
        right_boundary = np.zeros_like(centerline)
        left_boundary[:, 0] = centerline[:, 0] + d_left * nx
        left_boundary[:, 1] = centerline[:, 1] + d_left * ny
        right_boundary[:, 0] = centerline[:, 0] - d_right * nx
        right_boundary[:, 1] = centerline[:, 1] - d_right * ny

        self.left_boundary_xy = left_boundary
        self.right_boundary_xy = right_boundary

    def capture(
        self,
        ego_state: VehicleState,
        opp_state: VehicleState,
        sim_time: Optional[float] = None,
        title: Optional[str] = None,
        ego_traj: Optional[np.ndarray] = None,
        opp_traj: Optional[np.ndarray] = None,
        ego_goal_grid: Optional[np.ndarray] = None,
        extra_text: Optional[str] = None,
        render_info: Optional[Dict[str, float]] = None,
    ) -> None:
        """Delegate the two-vehicle convenience interface to capture_multi()."""
        self.capture_multi(
            vehicle_states=[ego_state, opp_state],
            sim_time=sim_time,
            title=title,
            vehicle_trajs=[ego_traj, opp_traj],
            ego_goal_grid=ego_goal_grid,
            extra_text=extra_text,
            render_info=render_info,
            follow_vehicle_idx=0,
        )

    def save(self, video_path: str) -> None:
        """Save the collected RGB frames to a video file on disk."""
        if len(self.frames) == 0:
            raise RuntimeError("No frames were captured. Cannot save video.")

        print(f"Saving offline video to {video_path} with {len(self.frames)} frames at {self.output_fps} FPS...")
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        with imageio.get_writer(video_path, fps=self.output_fps) as writer:
            for frame in self.frames:
                writer.append_data(frame)

    def reset_frames(self) -> None:
        """Reset stored frames and all per-vehicle history buffers."""
        self.frames = []
        self._ego_history = []
        self._opp_history = []
        self._vehicle_histories = []

    def close(self) -> None:
        """Close figure resources and clear cached renderer state."""
        if self._fig is not None:
            plt.close(self._fig)
        self._fig = None
        self._ax = None
        self._hud_ax = None
        self._mini_map_ax = None
        self._fpv_ax = None
        self._background_rgba = None
        self._ego_history = []
        self._opp_history = []
        self._vehicle_histories = []

    def _load_map(self) -> None:
        """Load the occupancy map image and convert it to a light-gray race track background."""
        with open(self.map_yaml_path, "r", encoding="utf-8") as f:
            map_meta = yaml.safe_load(f)

        image_path = map_meta["image"]
        resolution = float(map_meta["resolution"])
        origin = map_meta["origin"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(self.map_yaml_path), image_path)

        map_img = imageio.imread(image_path)
        if map_img.ndim == 3:
            map_img = map_img[..., 0]
        map_img = np.asarray(map_img, dtype=np.float32)
        if map_img.max() > 1.0:
            map_img = map_img / 255.0

        threshold = 0.5
        h, w = map_img.shape
        display_img = np.ones((h, w, 3), dtype=np.float32)
        track_mask = map_img >= threshold
        background_mask = map_img < threshold
        display_img[track_mask] = [216 / 255.0, 222 / 255.0, 233 / 255.0]
        display_img[background_mask] = [1.0, 1.0, 1.0]

        x_min = float(origin[0])
        y_min = float(origin[1])
        x_max = x_min + w * resolution
        y_max = y_min + h * resolution

        self._map_img = display_img
        self._map_extent = (x_min, x_max, y_min, y_max)
        self._axis_limits = (x_min - self.axis_margin, x_max + self.axis_margin, y_min - self.axis_margin, y_max + self.axis_margin)

    def _prepare_scene(self) -> None:
        """Create a wide figure with a left HUD and a physically square main track axis."""
        self._fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
        self._fig.patch.set_facecolor(self.theme["canvas_bg"])

        fig_w, fig_h = float(self.figure_size[0]), float(self.figure_size[1])
        square_width = min(0.98, fig_h / max(fig_w, 1e-6))
        main_left = max(self.hud_panel_fraction, 1.0 - square_width)
        main_width = max(0.05, min(square_width, 1.0 - main_left))
        main_height = min(1.0, main_width * fig_w / max(fig_h, 1e-6))
        main_bottom = 0.5 * (1.0 - main_height)

        self.hud_panel_fraction = main_left
        self._ax = self._fig.add_axes([main_left, main_bottom, main_width, main_height])
        self._hud_ax = self._fig.add_axes([0.0, 0.0, main_left, 1.0])
        self._layout_hud_axes()
        self._redraw_static_scene()
        if self.background_cache:
            self._fig.canvas.draw()
            self._background_rgba = self._fig.canvas.copy_from_bbox(self._fig.bbox)

    def _layout_hud_axes(self) -> None:
        """Create physically square track-centric BEV and mini-track inset axes inside the left HUD bar."""
        fig_w, fig_h = float(self.figure_size[0]), float(self.figure_size[1])
        hud_w = float(self.hud_panel_fraction)
        inset_h = min(0.255, hud_w * 0.72 * fig_w / max(fig_h, 1e-6))
        inset_w = inset_h * fig_h / max(fig_w, 1e-6)
        inset_left = 0.5 * max(hud_w - inset_w, 0.0)
        self._fpv_ax = self._fig.add_axes([inset_left, 0.345, inset_w, inset_h])
        self._mini_map_ax = self._fig.add_axes([inset_left, 0.030, inset_w, inset_h])

    def _redraw_static_scene(self) -> None:
        """Reset all axes, redraw the main map and track geometry, and clear the HUD inset axes."""
        self._ax.clear()
        self._ax.set_facecolor(self.theme["canvas_bg"])
        self._ax.imshow(self._map_img, origin="upper", extent=self._map_extent, interpolation="nearest")

        if self.left_boundary_xy is not None and len(self.left_boundary_xy) > 0:
            self._ax.plot(
                self.left_boundary_xy[:, 0],
                self.left_boundary_xy[:, 1],
                color="black",
                linewidth=4.0,
                linestyle="-",
                alpha=1.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=5,
            )
        if self.right_boundary_xy is not None and len(self.right_boundary_xy) > 0:
            self._ax.plot(
                self.right_boundary_xy[:, 0],
                self.right_boundary_xy[:, 1],
                color="black",
                linewidth=4.0,
                linestyle="-",
                alpha=1.0,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=5,
            )
        if self.refline_xy is not None and len(self.refline_xy) > 0:
            self._ax.plot(
                self.refline_xy[:, 0],
                self.refline_xy[:, 1],
                color=self.refline_color,
                linewidth=1.5,
                alpha=0.9,
                zorder=6,
            )

        self._ax.set_aspect("equal", adjustable="box")
        self._ax.axis("off")
        if self.fixed_axis and self._axis_limits is not None:
            xmin, xmax, ymin, ymax = self._axis_limits
            self._ax.set_xlim(xmin, xmax)
            self._ax.set_ylim(ymin, ymax)
        self._ax.grid(False)

        self._hud_ax.clear()
        self._hud_ax.set_facecolor(self.theme["hud_bg"])
        self._hud_ax.set_xlim(0.0, 1.0)
        self._hud_ax.set_ylim(0.0, 1.0)
        self._hud_ax.axis("off")

        for hud_inset in [self._mini_map_ax, self._fpv_ax]:
            hud_inset.clear()
            hud_inset.set_facecolor(self.theme["canvas_bg"])
            for spine in hud_inset.spines.values():
                spine.set_visible(False)
            hud_inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def _restore_background(self) -> None:
        """Restore the cached figure background when this helper is explicitly called."""
        if self._background_rgba is not None:
            self._fig.canvas.restore_region(self._background_rgba)

    def _draw_vehicle(self, state: VehicleState, color: str, label: str, vehicle_idx: int = 0) -> None:
        """Draw one vehicle with a sprite and fall back to a rotated polygon footprint."""
        center_x, center_y = self._vehicle_center_world(
            x=state.x,
            y=state.y,
            heading=state.heading,
            ref_offset=self.vehicle_ref_offset,
        )

        sprite_drawn = self._draw_vehicle_sprite_on_axis(
            self._ax,
            vehicle_idx=int(vehicle_idx),
            center_x=center_x,
            center_y=center_y,
            heading_angle=float(state.heading),
            length=self.vehicle_length,
            width=self.vehicle_width,
            zorder=21,
            alpha=1.0,
        )

        if sprite_drawn:
            return

        corners = self._vehicle_corners_world(
            x=state.x,
            y=state.y,
            heading=state.heading,
            length=self.vehicle_length,
            width=self.vehicle_width,
            ref_offset=self.vehicle_ref_offset,
        )
        poly = Polygon(corners, closed=True, facecolor=color, edgecolor=color, linewidth=1.0, zorder=21)
        self._ax.add_patch(poly)

        if self.show_heading_line:
            hx = center_x + self.heading_line_scale * np.cos(state.heading)
            hy = center_y + self.heading_line_scale * np.sin(state.heading)
            self._ax.plot([center_x, hx], [center_y, hy], color="black", linewidth=1.3, zorder=22)

    @staticmethod
    def _vehicle_center_world(x: float, y: float, heading: float, ref_offset: float) -> Tuple[float, float]:
        """Convert the state reference point into the vehicle geometric center."""
        cx = x + ref_offset * np.cos(heading)
        cy = y + ref_offset * np.sin(heading)
        return cx, cy

    @staticmethod
    def _vehicle_corners_world(
        x: float,
        y: float,
        heading: float,
        length: float,
        width: float,
        ref_offset: float = 0.0,
    ) -> np.ndarray:
        """Compute the four vehicle corners in world coordinates."""
        cx = x + ref_offset * np.cos(heading)
        cy = y + ref_offset * np.sin(heading)
        half_l = 0.5 * length
        half_w = 0.5 * width
        local_corners = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]])
        c = np.cos(heading)
        s = np.sin(heading)
        rot = np.array([[c, -s], [s, c]])
        world_corners = local_corners @ rot.T
        world_corners[:, 0] += cx
        world_corners[:, 1] += cy
        return world_corners


    def _draw_vehicle_legend(self, vehicle_states: Sequence[VehicleState]) -> None:
        """Draw vehicle color legend in the top-right HUD area."""
        if vehicle_states is None or len(vehicle_states) == 0:
            return

        legend_count = min(3, len(vehicle_states))

        x0 = 0.66
        y0 = 0.948
        row_gap = 0.040
        rect_w = 0.060
        rect_h = 0.018
        text_gap = 0.025

        for i in range(legend_count):
            y = y0 - i * row_gap
            color = self._get_vehicle_color(i)
            label = self._get_vehicle_role_name(i)

            self._hud_ax.add_patch(
                Rectangle(
                    (x0, y - 0.5 * rect_h),
                    rect_w,
                    rect_h,
                    transform=self._hud_ax.transAxes,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=12,
                )
            )

            self._draw_text_line(
                x0 + rect_w + text_gap,
                y,
                label,
                self.theme["hud_muted"],
                fontsize=11,
                weight="bold",
                ha="left",
                va="center",
            )

    def _draw_ego_hud(
        self,
        render_info: Optional[Dict[str, float]],
        vehicle_states: Sequence[VehicleState],
        follow_vehicle_idx: int,
    ) -> None:
        """Draw ego telemetry together with the global mini-map and a track-aligned BEV for the followed vehicle."""
        render_info = {} if render_info is None else dict(render_info)
        follow_idx = int(np.clip(follow_vehicle_idx, 0, max(len(vehicle_states) - 1, 0)))

        track_name = str(render_info.get("track_name", render_info.get("map_name", "Unknown")))
        lap_time = float(render_info.get("lap_time", self._last_sim_time))
        ego_speed = self._pick_first_float(render_info, ["ego_speed", "speed", "v", "ego_v"], 0.0)
        ego_delta = self._pick_first_float(render_info, ["ego_delta", "ego_steer", "delta", "steer"], 0.0)

        self._draw_hud_background()
        self._draw_text_line(0.08, 0.95, "TRACK", self.theme["hud_muted"], fontsize=13, weight="bold")
        self._draw_text_line(0.08, 0.918, track_name, self.theme["hud_text"], fontsize=18, weight="bold")

        self._draw_vehicle_legend(vehicle_states)

        self._draw_text_line(0.08, 0.868, "LAPTIME", self.theme["hud_muted"], fontsize=13, weight="bold")
        self._draw_text_line(0.08, 0.836, self._format_lap_time(lap_time), self.theme["hud_text"], fontsize=18, weight="bold")

        control_box = self._draw_hud_panel(0.06, 0.655, 0.88, 0.15, title="CONTROL INPUTS")
        self._draw_delta_speed_panel(control_box, ego_delta, ego_speed)

        self._draw_text_line(0.08, 0.635, "BIRD'S-EYE VIEW", "black", fontsize=13, weight="bold")
        self._draw_ego_bev(vehicle_states=vehicle_states, follow_vehicle_idx=follow_idx)

        self._draw_text_line(0.08, 0.315, "TOP-DOWN TRACK VIEW", "black", fontsize=13, weight="bold")
        self._draw_minimap(vehicle_states=vehicle_states, follow_vehicle_idx=follow_idx)

    @staticmethod
    def _format_lap_time(lap_time: float) -> str:
        """Format lap time with two integer digits before the decimal point."""
        lap_time = max(float(lap_time), 0.0)
        integer = int(lap_time) % 100
        hundredths = int(round((lap_time - int(lap_time)) * 100.0))
        if hundredths >= 100:
            integer = (integer + 1) % 100
            hundredths = 0
        return f"{integer:02d}.{hundredths:02d} s"

    def _draw_hud_background(self) -> None:
        """Draw the HUD background and a soft seam strip that visually connects both panels."""
        self._hud_ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, facecolor=self.theme["hud_bg"], edgecolor="none", zorder=1))
        self._hud_ax.add_patch(Rectangle((0.975, 0.0), 0.025, 1.0, transform=self._hud_ax.transAxes, facecolor=self.theme["seam_color"], edgecolor="none", zorder=2))

    def _draw_hud_panel(self, x: float, y: float, w: float, h: float, title: str) -> Tuple[float, float, float, float]:
        """Reserve one HUD content area without drawing separator lines."""
        self._draw_text_line(x + 0.02, y + h - 0.01, title, self.theme["hud_muted"], fontsize=13, weight="bold", va="top")
        return (x + 0.02, y + 0.02, w - 0.04, h - 0.045)

    def _draw_text_line(
        self,
        x: float,
        y: float,
        text: str,
        color: str,
        fontsize: float = 12,
        weight: str = "normal",
        ha: str = "left",
        va: str = "center",
    ) -> None:
        """Draw one HUD text label in normalized sidebar coordinates."""
        self._hud_ax.text(
            x,
            y,
            text,
            transform=self._hud_ax.transAxes,
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            fontweight=weight,
            zorder=10,
        )

    def _draw_steering_wheel_gauge(self, box: Tuple[float, float, float, float], delta: float) -> None:
        """Draw steering δ as a large aspect-correct wheel with spokes aligned to the wheel radius."""
        x, y, w, h = box
        radius = min(w * 0.24, h * 0.23) * 1.2
        wheel_radius = 2.0 * radius

        cx = x + w * 0.30
        cy = y + h * 0.42
        cy_icon = cy + 0.02

        bbox = self._hud_ax.get_position()
        fig_w, fig_h = float(self.figure_size[0]), float(self.figure_size[1])
        y_scale = (bbox.width * fig_w) / max(bbox.height * fig_h, 1e-6)

        delta_clip = float(np.clip(delta, -self.steer_limit, self.steer_limit))
        ratio = 0.0 if self.steer_limit <= 1e-6 else float(np.clip(delta_clip / self.steer_limit, -1.0, 1.0))
        angle = ratio * np.deg2rad(95.0)

        self._draw_text_line(cx + 0.02, y + h * 0.25, "Steering (rad):", self.theme["hud_muted"], fontsize=11, weight="normal", ha="center")
        self._draw_text_line(cx, y + h * 0.05, f"{delta_clip:+.3f}", self.theme["hud_text"], fontsize=13, weight="normal", ha="center")

        self._hud_ax.add_patch(Ellipse((cx, cy_icon), 2.0 * wheel_radius, 2.0 * wheel_radius * y_scale, transform=self._hud_ax.transAxes, facecolor="none", edgecolor=self.theme["hud_grid"], linewidth=2.0, zorder=6))

        hub_radius = wheel_radius * 0.16
        self._hud_ax.add_patch(Ellipse((cx, cy_icon), 2.0 * hub_radius, 2.0 * hub_radius * y_scale, transform=self._hud_ax.transAxes, facecolor=self.theme["hud_grid"], edgecolor="none", zorder=7))

        spoke_inner = wheel_radius * 0.20
        spoke_outer = wheel_radius * 0.86
        spoke_angles = np.array([np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi / 3.0, np.pi / 2.0 + 4.0 * np.pi / 3.0]) + angle

        for spoke_angle in spoke_angles:
            x1 = cx + spoke_inner * np.cos(spoke_angle)
            y1 = cy_icon + spoke_inner * y_scale * np.sin(spoke_angle)
            x2 = cx + spoke_outer * np.cos(spoke_angle)
            y2 = cy_icon + spoke_outer * y_scale * np.sin(spoke_angle)
            self._hud_ax.plot([x1, x2], [y1, y2], transform=self._hud_ax.transAxes, color=self.theme["hud_grid"], linewidth=2.0, zorder=7)

        marker_angle = np.pi / 2.0 + angle
        marker_radius = wheel_radius * 0.92
        marker_x = cx + marker_radius * np.cos(marker_angle)
        marker_y = cy_icon + marker_radius * y_scale * np.sin(marker_angle)

        marker_size = wheel_radius * 0.11
        self._hud_ax.add_patch(Ellipse((marker_x, marker_y), 2.0 * marker_size, 2.0 * marker_size * y_scale, transform=self._hud_ax.transAxes, facecolor=self.theme["hud_green"], edgecolor="none", alpha=0.95, zorder=8))

    def _draw_delta_speed_panel(self, box: Tuple[float, float, float, float], delta: float, ego_speed: float) -> None:
        """Draw the steering wheel and speed bar above their aligned labels and numeric values."""
        x, y, w, h = box
        delta_box = (x, y, w * 0.50, h)
        speed_box = (x + w * 0.54, y, w * 0.42, h)
        self._draw_steering_wheel_gauge(delta_box, delta)
        self._draw_horizontal_speed_bar(speed_box, ego_speed)

    def _draw_horizontal_speed_bar(self, box: Tuple[float, float, float, float], ego_speed: float) -> None:
        """Draw velocity as a horizontal progress bar with its label and numeric value below."""
        x, y, w, h = box
        speed_ratio = 0.0 if self.hud_speed_max <= 1e-6 else float(np.clip(ego_speed / self.hud_speed_max, 0.0, 1.0))
        text_x = x + w * 0.50

        self._draw_text_line(text_x, y + h * 0.25, "Velocity (m/s):", self.theme["hud_muted"], fontsize=11, weight="normal", ha="center")
        self._draw_text_line(text_x, y + h * 0.02, f"{ego_speed:05.2f}", self.theme["hud_text"], fontsize=13, weight="normal", ha="center")

        bar_x = x + w * 0.10
        bar_y = y + h * 0.34 + 0.02
        bar_w = w * 0.80
        bar_h = h * 0.16

        self._hud_ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h, transform=self._hud_ax.transAxes, facecolor="white", edgecolor=self.theme["hud_panel_edge"], linewidth=1.0, zorder=5))
        self._hud_ax.add_patch(Rectangle((bar_x, bar_y), bar_w * speed_ratio, bar_h, transform=self._hud_ax.transAxes, facecolor=self.theme["hud_green"], edgecolor="none", alpha=0.88, zorder=6))
        n = 5
        for i in range(n):
            tick_x = bar_x + bar_w * i / float(n-1)
            self._hud_ax.plot([tick_x, tick_x], [bar_y, bar_y + bar_h], transform=self._hud_ax.transAxes, color=self.theme["hud_grid"], linewidth=0.8, zorder=7)

    def _draw_minimap(self, vehicle_states: Sequence[VehicleState], follow_vehicle_idx: int) -> None:
        """Draw a global centerline-only mini-map with the followed vehicle highlighted."""
        ax = self._mini_map_ax
        ax.clear()
        ax.set_facecolor(self.theme["canvas_bg"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if self.refline_xy is not None and len(self.refline_xy) > 0:
            ref = np.asarray(self.refline_xy)
            ax.plot(ref[:, 0], ref[:, 1], color=self.refline_color, linewidth=2.0, alpha=0.8)
        # Keep the mini-map uncluttered by drawing only the reference centerline.

        for i, hist in enumerate(self._vehicle_histories[: len(vehicle_states)]):
            if len(hist) <= 1:
                continue
            hist_xy = np.asarray([[p[1], p[2]] for p in hist], dtype=float)
            ax.plot(hist_xy[:, 0], hist_xy[:, 1], color=self._get_vehicle_color(i), linewidth=1.0 if i == follow_vehicle_idx else 0.7, alpha=0.35)

        for i, state in enumerate(vehicle_states):
            x, y = float(state.x), float(state.y)
            color = self._get_vehicle_color(i)
            if i == follow_vehicle_idx:
                for s, alpha in [(180, 0.10), (100, 0.22), (42, 0.95)]:
                    ax.scatter([x], [y], s=s, color=color, alpha=alpha, edgecolors="none", zorder=5)
            else:
                for s, alpha in [(65, 0.10), (30, 0.20), (12, 0.95)]:
                    ax.scatter([x], [y], s=s, color=color, alpha=alpha, edgecolors="none", zorder=5)

        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        if self._axis_limits is not None:
            xmin, xmax, ymin, ymax = self._axis_limits
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    def _draw_ego_bev(self, vehicle_states: Sequence[VehicleState], follow_vehicle_idx: int) -> None:
        """Draw a track-aligned BEV anchored to the followed vehicle with that vehicle in the lower half."""
        ax = self._fpv_ax
        ax.clear()
        ax.set_facecolor(self.theme["canvas_bg"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axis("off")

        if len(vehicle_states) == 0:
            ax.text(0.5, 0.5, "No vehicle", ha="center", va="center", color=self.theme["hud_muted"], fontsize=10, transform=ax.transAxes)
            return

        follow_idx = int(np.clip(follow_vehicle_idx, 0, len(vehicle_states) - 1))
        ego = vehicle_states[follow_idx]
        window_half_w = 0.5 * float(self.bev_window[0])
        window_half_h = 0.5 * float(self.bev_window[1])
        ego_display_y = -0.36 * window_half_h

        def nearest_track_heading() -> float:
            """Return the reference-line tangent heading nearest to the followed vehicle."""
            if self.refline_xy is None or len(self.refline_xy) < 2:
                return float(ego.heading)
            ref = np.asarray(self.refline_xy, dtype=float)
            d2 = (ref[:, 0] - float(ego.x)) ** 2 + (ref[:, 1] - float(ego.y)) ** 2
            idx = int(np.argmin(d2))
            prev_idx = (idx - 1) % len(ref)
            next_idx = (idx + 1) % len(ref)
            tangent = ref[next_idx] - ref[prev_idx]
            if np.linalg.norm(tangent) < 1e-9:
                tangent = np.array([np.cos(float(ego.heading)), np.sin(float(ego.heading))], dtype=float)
            return float(np.arctan2(tangent[1], tangent[0]))

        track_heading = nearest_track_heading()

        def world_to_track(points_xy: np.ndarray) -> np.ndarray:
            """Transform world points into followed-vehicle track-local lateral-forward coordinates."""
            points_xy = np.asarray(points_xy, dtype=float)
            if len(points_xy) == 0:
                return points_xy.reshape(0, 2)
            dx = points_xy[:, 0] - float(ego.x)
            dy = points_xy[:, 1] - float(ego.y)
            c = np.cos(track_heading)
            s = np.sin(track_heading)
            forward = dx * c + dy * s
            lateral = dx * s - dy * c
            return np.column_stack([lateral, forward])

        def local_xy(points: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            """Convert world points into plotted track-centric BEV arrays."""
            if points is None or len(points) == 0:
                return None
            local = world_to_track(np.asarray(points, dtype=float))
            return local[:, 0], local[:, 1] + ego_display_y

        def clipped_mask(plot_x: np.ndarray, plot_y: np.ndarray, scale: float = 1.15) -> np.ndarray:
            """Return a simple rectangular visibility mask for the BEV panel."""
            return (
                (np.abs(plot_x) <= window_half_w * scale)
                & (plot_y >= -window_half_h * scale)
                & (plot_y <= window_half_h * scale)
            )

        def plot_local_line(points: Optional[np.ndarray], color: str, linewidth: float, alpha: float, zorder: int) -> None:
            """Plot clipped track-centric polyline segments without connecting disjoint visible chunks."""
            values = local_xy(points)
            if values is None:
                return

            plot_x, plot_y = values
            mask = clipped_mask(plot_x, plot_y)

            if np.count_nonzero(mask) < 2:
                return

            visible_idx = np.flatnonzero(mask)
            if visible_idx.size < 2:
                return

            # Split index-discontinuous visible points so unrelated track portions are not connected.
            breaks = np.where(np.diff(visible_idx) > 1)[0] + 1
            index_segments = np.split(visible_idx, breaks)

            # Split large local-coordinate jumps to avoid closed-loop and noncontiguous connections.
            max_jump = 0.35 * max(float(self.bev_window[0]), float(self.bev_window[1]))

            for segment_idx in index_segments:
                if segment_idx.size < 2:
                    continue

                seg_x = plot_x[segment_idx]
                seg_y = plot_y[segment_idx]

                jump = np.hypot(np.diff(seg_x), np.diff(seg_y))
                jump_breaks = np.where(jump > max_jump)[0] + 1
                sub_segments = np.split(np.arange(segment_idx.size), jump_breaks)

                for sub in sub_segments:
                    if sub.size < 2:
                        continue

                    ax.plot(
                        seg_x[sub],
                        seg_y[sub],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=zorder,
                        solid_capstyle="round",
                    )

        def split_visible_boundary_segments(
            lx: np.ndarray,
            ly: np.ndarray,
            rx: np.ndarray,
            ry: np.ndarray,
            mask: np.ndarray,
        ) -> List[np.ndarray]:
            """Split visible boundary indices so fill polygons never connect disjoint track chunks."""
            visible_idx = np.flatnonzero(mask)
            if visible_idx.size < 2:
                return []

            # Split original waypoint-index discontinuities.
            breaks = np.where(np.diff(visible_idx) > 1)[0] + 1
            index_segments = np.split(visible_idx, breaks)

            # Split again when local BEV coordinates jump too far.
            max_jump = 0.35 * max(float(self.bev_window[0]), float(self.bev_window[1]))
            segments = []

            for idx in index_segments:
                if idx.size < 2:
                    continue

                left_jump = np.hypot(np.diff(lx[idx]), np.diff(ly[idx]))
                right_jump = np.hypot(np.diff(rx[idx]), np.diff(ry[idx]))
                jump = np.maximum(left_jump, right_jump)

                jump_breaks = np.where(jump > max_jump)[0] + 1
                local_segments = np.split(np.arange(idx.size), jump_breaks)

                for local_idx in local_segments:
                    if local_idx.size < 2:
                        continue
                    segments.append(idx[local_idx])

            return segments


        if self.left_boundary_xy is not None and self.right_boundary_xy is not None:
            left_values = local_xy(self.left_boundary_xy)
            right_values = local_xy(self.right_boundary_xy)

            if left_values is not None and right_values is not None:
                lx, ly = left_values
                rx, ry = right_values

                mask = clipped_mask(lx, ly, scale=1.25) & clipped_mask(rx, ry, scale=1.25)
                fill_segments = split_visible_boundary_segments(lx, ly, rx, ry, mask)

                for seg_idx in fill_segments:
                    fill_x = np.concatenate([lx[seg_idx], rx[seg_idx][::-1]])
                    fill_y = np.concatenate([ly[seg_idx], ry[seg_idx][::-1]])

                    if fill_x.size < 4 or fill_y.size < 4:
                        continue
                    if not np.all(np.isfinite(fill_x)) or not np.all(np.isfinite(fill_y)):
                        continue

                    ax.fill(
                        fill_x,
                        fill_y,
                        facecolor="#EEF3FA",
                        edgecolor="none",
                        alpha=0.9,
                        zorder=1,
                    )

        plot_local_line(self.left_boundary_xy, "#333333", 1.8, 0.85, 2)
        plot_local_line(self.right_boundary_xy, "#333333", 1.8, 0.85, 2)
        plot_local_line(self.refline_xy, self.refline_color, 1.3, 0.95, 3)

        for i, hist in enumerate(self._vehicle_histories[: len(vehicle_states)]):
            if len(hist) <= 1:
                continue
            hist_xy = np.asarray([[p[1], p[2]] for p in hist], dtype=float)
            local_hist = world_to_track(hist_xy)
            plot_x = local_hist[:, 0]
            plot_y = local_hist[:, 1] + ego_display_y
            ax.plot(plot_x, plot_y, color=self._get_vehicle_color(i), linewidth=1.2 if i == follow_idx else 0.8, alpha=0.45, zorder=4)

        for i, state in enumerate(vehicle_states):
            color = self._get_vehicle_color(i)
            corners_world = self._vehicle_corners_world(
                x=state.x,
                y=state.y,
                heading=state.heading,
                length=self.vehicle_length,
                width=self.vehicle_width,
                ref_offset=self.vehicle_ref_offset,
            )
            local_corners = world_to_track(corners_world)
            plot_cx = local_corners[:, 0]
            plot_cy = local_corners[:, 1] + ego_display_y
            # Preserve the historical sprite anchor at the state reference point while polygon corners use vehicle_ref_offset.
            center_local = world_to_track(np.array([[state.x, state.y]], dtype=float))[0]
            center_x = center_local[0]
            center_y = center_local[1] + ego_display_y

            if abs(center_x) > window_half_w * 1.3 or center_y < -window_half_h * 1.3 or center_y > window_half_h * 1.3:
                continue

            heading_probe_world = np.array([[float(state.x) + np.cos(float(state.heading)), float(state.y) + np.sin(float(state.heading))]], dtype=float)
            heading_probe_local = world_to_track(heading_probe_world)[0]
            heading_delta = heading_probe_local - center_local
            local_heading_angle = float(np.arctan2(heading_delta[1], heading_delta[0]))

            sprite_drawn = self._draw_vehicle_sprite_on_axis(
                ax,
                vehicle_idx=int(i),
                center_x=center_x,
                center_y=center_y,
                heading_angle=local_heading_angle,
                length=self.vehicle_length,
                width=self.vehicle_width,
                zorder=7 if i == follow_idx else 6,
                alpha=1.0,
            )

            if not sprite_drawn:
                ax.add_patch(
                    Polygon(
                        np.column_stack([plot_cx, plot_cy]),
                        closed=True,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=1.0,
                        alpha=1.0,
                        zorder=6 if i == follow_idx else 5,
                    )
                )
                ax.scatter([center_x], [center_y], s=32 if i == follow_idx else 16, color=color, edgecolors="none", alpha=0.95, zorder=7)

                if i == follow_idx:
                    hx = center_x + self.vehicle_length * 0.55 * heading_delta[0]
                    hy = center_y + self.vehicle_length * 0.55 * heading_delta[1]
                    ax.plot([center_x, hx], [center_y, hy], color="black", linewidth=1.0, alpha=0.85, zorder=8)

        ax.set_xlim(-window_half_w, window_half_w)
        ax.set_ylim(-window_half_h, window_half_h)
        ax.set_aspect("equal", adjustable="box")

    @staticmethod
    def _pick_first_float(render_info: Dict[str, float], keys: Sequence[str], default: float) -> float:
        """Return the first numeric value found under a list of candidate telemetry keys."""
        for key in keys:
            value = render_info.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)

    def _figure_to_rgb(self) -> np.ndarray:
        """Convert the current figure canvas into an RGB uint8 frame array."""
        self._fig.canvas.draw()
        buf = np.asarray(self._fig.canvas.buffer_rgba())
        return buf[..., :3].copy()

    def _clear_dynamic_artists(self) -> None:
        """Redraw the static base scene to prepare all axes for the next frame."""
        self._redraw_static_scene()
