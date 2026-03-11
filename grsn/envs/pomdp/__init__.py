from gym.envs.registration import register
import gym

# Notation:
# F: full observed (original env)
# P: position/angle observed
# V: velocity observed

def _make_pendulum_f():
    return gym.make("Pendulum-v1")

def _make_pendulum_p():
    return gym.make("Pendulum-v1")

def _make_pendulum_v():
    return gym.make("Pendulum-v1")

def _make_cartpole_f():
    return gym.make("CartPole-v1")

def _make_cartpole_p():
    return gym.make("CartPole-v1")

def _make_cartpole_v():
    return gym.make("CartPole-v1")

def _make_lunarlander_f():
    return gym.make("LunarLander-v2")

def _make_lunarlander_p():
    return gym.make("LunarLander-v2")

def _make_lunarlander_v():
    return gym.make("LunarLander-v2")

register(
    "Pendulum-F-v0",
    entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(
        env_fn=_make_pendulum_f, partially_obs_dims=[0, 1, 2]
    ),  # angle & velocity
    max_episode_steps=200,
)

register(
    "Pendulum-P-v0",
    entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(env_fn=_make_pendulum_p, partially_obs_dims=[0, 1]),  # angle
    max_episode_steps=200,
)

register(
    "Pendulum-V-v0",
    entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(env_fn=_make_pendulum_v, partially_obs_dims=[2]),  # velocity
    max_episode_steps=200,
)

register(
    "CartPole-F-v0",
    entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(
        env_fn=_make_cartpole_f, partially_obs_dims=[0, 1, 2, 3]
    ),  # angle & velocity
    max_episode_steps=200,  # reward threshold for solving the task: 195
)

register(
    "CartPole-P-v0",
    entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(env_fn=_make_cartpole_p, partially_obs_dims=[0, 2]),
    max_episode_steps=200,
)

register(
    "CartPole-V-v0",
    entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(env_fn=_make_cartpole_v, partially_obs_dims=[1, 3]),
    max_episode_steps=200,
)

register(
    "LunarLander-F-v0",
    entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(
        env_fn=_make_lunarlander_f, partially_obs_dims=list(range(8))
    ),  # angle & velocity
    max_episode_steps=1000,  # reward threshold for solving the task: 200
)

register(
    "LunarLander-P-v0",
    entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(env_fn=_make_lunarlander_p, partially_obs_dims=[0, 1, 4, 6, 7]),
    max_episode_steps=1000,
)

register(
    "LunarLander-V-v0",
    entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
    kwargs=dict(env_fn=_make_lunarlander_v, partially_obs_dims=[2, 3, 5, 6, 7]),
    max_episode_steps=1000,
)

### Below are pybullect (roboschool) environments, using BLT for Bullet
try:
    # Fix for pybullet_envs compatibility with newer gym versions (0.26+)
    # Need to patch before importing pybullet_envs
    import gym.envs.registration as _reg
    if not hasattr(_reg.registry, 'env_specs'):
        class _RegistryWrapper(dict):
            def __init__(self, d):
                super().__init__(d)
                self.env_specs = self
        _reg.registry = _RegistryWrapper(_reg.registry)
    import pybullet_envs

    def _make_hopper_f():
        return gym.make("HopperBulletEnv-v0")

    def _make_hopper_p():
        return gym.make("HopperBulletEnv-v0")

    def _make_hopper_v():
        return gym.make("HopperBulletEnv-v0")

    def _make_walker_f():
        return gym.make("Walker2DBulletEnv-v0")

    def _make_walker_p():
        return gym.make("Walker2DBulletEnv-v0")

    def _make_walker_v():
        return gym.make("Walker2DBulletEnv-v0")

    def _make_ant_f():
        return gym.make("AntBulletEnv-v0")

    def _make_ant_p():
        return gym.make("AntBulletEnv-v0")

    def _make_ant_v():
        return gym.make("AntBulletEnv-v0")

    def _make_cheetah_f():
        return gym.make("HalfCheetahBulletEnv-v0")

    def _make_cheetah_p():
        return gym.make("HalfCheetahBulletEnv-v0")

    def _make_cheetah_v():
        return gym.make("HalfCheetahBulletEnv-v0")

    """
    The observation space can be divided into several parts:
    np.concatenate(
    [
        z - self.initial_z, # pos
        np.sin(angle_to_target), # pos
        np.cos(angle_to_target), # pos
        0.3 * vx, # vel
        0.3 * vy, # vel
        0.3 * vz, # vel
        r, # pos
        p # pos
    ], # above are 8 dims
    [j], # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
    [self.feet_contact], # depends on foot_list, belongs to pos
    ])
    """
    register(
        "HopperBLT-F-v0",
        entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
        kwargs=dict(
            env_fn=_make_hopper_f,
            partially_obs_dims=list(range(15)),
        ),  # full obs
        max_episode_steps=1000,
    )

    register(
        "HopperBLT-P-v0",
        entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
        kwargs=dict(
            env_fn=_make_hopper_p,
            partially_obs_dims=[0, 1, 2, 6, 7, 8, 10, 12, 14],  # one foot
        ),  # pos
        max_episode_steps=1000,
    )

    register(
        "HopperBLT-V-v0",
        entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
        kwargs=dict(
            env_fn=_make_hopper_v,
            partially_obs_dims=[3, 4, 5, 9, 11, 13],
        ),  # vel
        max_episode_steps=1000,
    )

    register(
        "WalkerBLT-F-v0",
        entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
        kwargs=dict(
            env_fn=_make_walker_f,
            partially_obs_dims=list(range(22)),
        ),  # full obs
        max_episode_steps=1000,
    )

    register(
        "WalkerBLT-P-v0",
        entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
        kwargs=dict(
            env_fn=_make_walker_p,
            partially_obs_dims=[0, 1, 2, 6, 7, 8, 10, 12, 14, 16, 18, 20, 21],  # 2 feet
        ),  # pos
        max_episode_steps=1000,
    )

    register(
        "WalkerBLT-V-v0",
        entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
        kwargs=dict(
            env_fn=_make_walker_v,
            partially_obs_dims=[3, 4, 5, 9, 11, 13, 15, 17, 19],
        ),  # vel
        max_episode_steps=1000,
    )

    register(
        "AntBLT-F-v0",
        entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
        kwargs=dict(
            env_fn=_make_ant_f,
            partially_obs_dims=list(range(28)),
        ),  # full obs
        max_episode_steps=1000,
    )

    register(
        "AntBLT-P-v0",
        entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
        kwargs=dict(
            env_fn=_make_ant_p,
            partially_obs_dims=[
                0,
                1,
                2,
                6,
                7,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                25,
                26,
                27,
            ],  # 4 feet
        ),  # pos
        max_episode_steps=1000,
    )

    register(
        "AntBLT-V-v0",
        entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
        kwargs=dict(
            env_fn=_make_ant_v,
            partially_obs_dims=[3, 4, 5, 9, 11, 13, 15, 17, 19, 21, 23],
        ),  # vel
        max_episode_steps=1000,
    )

    register(
        "HalfCheetahBLT-F-v0",
        entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
        kwargs=dict(
            env_fn=_make_cheetah_f,
            partially_obs_dims=list(range(26)),
        ),  # full obs
        max_episode_steps=1000,
    )

    register(
        "HalfCheetahBLT-P-v0",
        entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
        kwargs=dict(
            env_fn=_make_cheetah_p,
            partially_obs_dims=[
                0,
                1,
                2,
                6,
                7,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                21,
                22,
                23,
                24,
                25,
            ],  # 6 feet
        ),  # pos
        max_episode_steps=1000,
    )

    register(
        "HalfCheetahBLT-V-v0",
        entry_point="grsn.envs.pomdp.wrappers:POMDPWrapper",
        kwargs=dict(
            env_fn=_make_cheetah_v,
            partially_obs_dims=[3, 4, 5, 9, 11, 13, 15, 17, 19],
        ),  # vel
        max_episode_steps=1000,
    )
except ImportError:
    print("pybullet_envs not available, skipping PyBullet environment registrations")
