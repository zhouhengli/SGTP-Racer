import os

from players.planner.planner_generators import ocp_based_multi_generator, flow_based_generator
from players.planner.flow_planner.flow_trajectory_generator import setup_runtime

from players.planner.lattice_planner.lattice_planner import LatticePlanner
from players.planner.lattice_planner.lattice_generator import lattice_multi_generator

from players.planner.mpc_planner.evo_mpcc import MPCPlanner
from players.planner.mpc_planner.mpc_generator import mpc_based_multi_generator

from players.planner.spliner.spliner_planner import setup_spliner_planner
from players.planner.spliner.spliner_generator import spliner_based_multi_generator

from players.planner.End2Race.end2race_generator import setup_end2race_planner, end2race_mppi_generator

from players.planner.biased_mppi.biased_mppi_ancillary import AncillaryBiasedMPPIPlanner
from players.planner.biased_mppi.biased_mppi_post_selection import BiasedMPPIPostSelectionPlanner
from players.utils.common import load_config, get_map_paths


def make_planner_variants(setup_fn, generator_fn, prefix):
    return {f"{prefix}": [setup_fn, generator_fn]}

def setup_biased_mppi_planner(
    args,
    map_name,
    raceline_file,
    config_path,
    v_scale=None,
    ocp_conf=None,
    game_block_conf=None,
    biased_type=None,
):
    config = load_config(config_path)
    map_directory, map_path = get_map_paths(map_name)
    raceline_path = os.path.join(
        map_directory,
        f"{map_name}_{raceline_file}.csv",
    )

    planner_cls = (
        AncillaryBiasedMPPIPlanner
        if str(biased_type or "none").lower() == "ancillary"
        else BiasedMPPIPostSelectionPlanner
    )

    planner = planner_cls(
        args,
        config,
        map_path,
        raceline_path,
        wb=args.wheel_base,
        v_scale=v_scale,
        ocp_conf=ocp_conf,
        game_block_conf=game_block_conf,
        biased_type=biased_type,
    )

    return planner, map_directory

def setup_lattice_planner(
    args,
    map_name,
    raceline_file,
    config_path,
    v_scale=None,
    ocp_conf=None,
    game_block_conf=None,
    biased_type=None,
):
    del ocp_conf, game_block_conf, biased_type

    config = load_config(config_path)
    lattice_config = load_config("players/config/lattice_config.yaml")

    for k, v in vars(lattice_config).items():
        setattr(config, k, v)

    map_directory, map_path = get_map_paths(map_name)
    raceline_path = os.path.join(
        map_directory,
        f"{map_name}_{raceline_file}.csv",
    )

    planner = LatticePlanner(
        args,
        config,
        map_path,
        raceline_path,
        wb=args.wheel_base,
        v_scale=v_scale,
    )

    return planner, map_directory

def setup_mpc_planner(
    args,
    map_name,
    raceline_file,
    config_path,
    v_scale=None,
    ocp_conf=None,
    game_block_conf=None,
    biased_type=None,
):
    del ocp_conf, game_block_conf, biased_type

    config = load_config(config_path)
    ocp_config = load_config("players/config/ocp_config.yaml")
    for k, v in vars(ocp_config).items():
        setattr(config, k, v)
    map_directory, map_path = get_map_paths(map_name)
    raceline_path = os.path.join(
        map_directory,
        f"{map_name}_{raceline_file}.csv",
    )

    planner = MPCPlanner(
        config,
        map_path,
        raceline_path,
        wb=args.wheel_base,
        v_scale=v_scale,
        max_opps=args.num_agents - 1,
    )

    return planner, map_directory

PLANNER_SPECS = [
    ("mppi", setup_biased_mppi_planner, ocp_based_multi_generator),
    ("flow_planner", setup_runtime, flow_based_generator),
    ("lattice_planner", setup_lattice_planner, lattice_multi_generator),
    ("spliner", setup_spliner_planner, spliner_based_multi_generator),
    ("mpc", setup_mpc_planner, mpc_based_multi_generator),
    ("end2race", setup_end2race_planner, end2race_mppi_generator),
]

planners = {}
for name, setup_fn, generator_fn in PLANNER_SPECS:
    planners.update(make_planner_variants(setup_fn, generator_fn, name))
