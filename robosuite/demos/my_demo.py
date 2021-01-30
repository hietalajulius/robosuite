from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *


if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = "PickPlace" #choose_environment()
    print("env name chosen:", options["env_name"])

    # If a multi-arm environment has been chosen, choose configuration and appropriate robot(s)
    if "TwoArm" in options["env_name"]:
        # Choose env config and add it to options
        options["env_configuration"] = choose_multi_arm_config()

        # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
        if options["env_configuration"] == 'bimanual':
            options["robots"] = 'Baxter'
        else:
            options["robots"] = []

            # Have user choose two robots
            print("A multiple single-arm configuration was chosen.\n")

            for i in range(2):
                print("Please choose Robot {}...\n".format(i))
                options["robots"].append(choose_robots(exclude_bimanual=True))

    # Else, we simply choose a single (single-armed) robot to instantiate in the environment
    else:
        options["robots"] = "Panda" #choose_robots(exclude_bimanual=True)
        print("robot chosen:", options["robots"])

    # Choose controller
    controller_name = "IK_POSE" #choose_controller()
    print("robot chosen:", controller_name)

    # Load the desired controller
    options["controller_configs"] = load_controller_config(
        default_controller=controller_name)

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=10,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    print(f"LOW: {low}")
    print(f"HIGH: {high}")

    site_pos = env.sim.data.get_site_xpos("gripper0_grip_site").copy()
    increases = np.array([
                        [-0.8169702, 0.988936, 0.823965],
                        [-0.94578123, 0.9432483, 0.7019442],
                        [-0.96698755, 0.81606424, 0.78746295],
                        [-0.97489953, 0.09362749, 0.86515844],
                        [-0.97881544, -0.5612759, 0.98431313],
                        [-0.9972523, -0.99255085, 0.9692926],
                        [-0.997425, -0.98192877, 0.104845054],
                        [-0.99912775, -0.6859133, -0.9893226],
                        [-0.9040002, 0.7857158, -0.9968162],
                        [-0.545036, -0.34209844, -0.9746304],
                        ])*0.03
    #increase = np.array([0,0,0.001])
    # do visualization
    split = 20
    modded_increases = []
    for incr in increases:
        print("incr", incr)
        for s in range(1,split+1):
            modded_increases.append([x / s for x in incr])
        
    print("modded", modded_increases)
    increases = np.array(modded_increases, dtype=object)
    for i in range(10000):
        print("i", i)
        increase = increases[i]
        action = np.random.uniform(low, high)
        action = np.zeros(7)
        action[:3] = increase
        obs, reward, done, _ = env.step(action)
        site_pos_new = env.sim.data.get_site_xpos("gripper0_grip_site").copy()
        print("error vector", (site_pos + increase) - site_pos_new)
        print("error norm", np.linalg.norm((site_pos + increase) - site_pos_new))
        site_pos = site_pos_new
        env.render()
