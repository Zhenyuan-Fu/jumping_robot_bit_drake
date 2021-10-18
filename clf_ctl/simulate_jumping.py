#!/usr/bin/env python

from pydrake.all import *
from controllers import *
from planners import BasicTrunkPlanner, TowrTrunkPlanner, OptPlanner
import os
import sys
from pydrake.geometry import (DrakeVisualizer, SceneGraph)
from pydrake.multibody.plant import (
    ConnectContactResultsToDrakeVisualizer,
    CoulombFriction,
)
from pydrake.geometry import (
    HalfSpace,
)
from pydrake import getDrakePath
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import DrakeVisualizer
from pydrake.geometry.render import (
    ClippingRange,
    DepthRange,
    DepthRenderCamera,
    RenderCameraCore,
    RenderLabel,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import BodyIndex
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import ConnectMeshcatVisualizer
from pydrake.systems.sensors import (
    CameraInfo,
    RgbdSensor,
)
from pinocchio.utils import *
from pinocchio import computeCentroidalDynamics, \
    computeCentroidalMapTimeVariation, RobotWrapper
import pinocchio

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)


reserved_labels = [
    RenderLabel.kDoNotRender,
    RenderLabel.kDontCare,
    RenderLabel.kEmpty,
    RenderLabel.kUnspecified,
]


def colorize_labels(image):
    """Colorizes labels."""
    cc = mpl.colors.ColorConverter()
    color_cycle = plt.rcParams["axes.prop_cycle"]
    colors = np.array([cc.to_rgb(c["color"]) for c in color_cycle])
    bg_color = [0, 0, 0]
    image = np.squeeze(image)
    background = np.zeros(image.shape[:2], dtype=bool)
    for label in reserved_labels:
        background |= image == int(label)
    color_image = colors[image % len(colors)]
    color_image[background] = bg_color
    return color_image


############### Common Parameters ###################
show_trunk_model = True
use_lcm = False

planning_method = "opt_traj_matlab"  # "towr" "osc_traj" "opt_traj_matlab"
control_method = "COMID"
# ID = Inverse Dynamics (standard QP),
# B = Basic (simple joint-space PD),
# COMID = Inverse Dynamics (CoM QP),
# MPTC = task-space passivity (plan to do)
# PC = passivity-constrained (plan to do)
# CLF = control-lyapunov-function based (plan to do)
# wbc = donghyun kim wbc (plan to do)
sim_time = 40.0
dt = 0.004
target_realtime_rate = 1.0

show_diagram = False
make_plots = True
jump_running = True  # is dong jumping task or not


#####################################################

builder = DiagramBuilder()

##################### load pinocchio urdf ###################

robot_urdf_path_pin = '/home/zhenyuan_fu/underactuated_jumping_test/jumping_robot_bit/urdf/jumping_robot_bit_pin.urdf'
p_model = pinocchio.buildModelFromUrdf(robot_urdf_path_pin,
                                       pinocchio.JointModelFreeFlyer())
##################### load drake urdf ########################

plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)
parser = Parser(plant, scene_graph)
robot_description_path_drake = "../jumping_robot_bit/urdf/jumping_robot_bit.urdf"
drake_path = getDrakePath()
robot_description_file = "drake/" + os.path.relpath(robot_description_path_drake,
                                                    start=drake_path)
biped = parser.AddModelFromFile(FindResourceOrThrow(robot_description_file))

# Add a flat ground with friction
X_BG = RigidTransform()
surface_friction = CoulombFriction(
    static_friction=1.0,
    dynamic_friction=1.0)
plant.RegisterCollisionGeometry(
    plant.world_body(),  # the body for which this object is registered
    X_BG,  # The fixed pose of the geometry frame G in the body frame B
    HalfSpace(),  # Defines the geometry of the object
    "ground_collision",  # A name
    surface_friction)  # Coulomb friction coefficients
plant.RegisterVisualGeometry(
    plant.world_body(),
    X_BG,
    HalfSpace(),
    "ground_visual",
    np.array([146/255, 209/255, 234/255, 0.2]))  # Color set to be completely transparent [0.5,0.5,0.5,0.0]
# Turn off gravity
# g = plant.mutable_gravity_field()
# g.set_gravity_vector([0,0,0])

plant.Finalize()
assert plant.geometry_source_is_registered()
#################### planner's trajectory visualization #######################
# Add custom visualizations for the trunk model and foot model
# The visualizations are the desired trajectories from planner
v_trunk_source = scene_graph.RegisterSource("v_trunk")
v_trunk_frame = GeometryFrame("v_trunk")
scene_graph.RegisterFrame(v_trunk_source, v_trunk_frame)
v_trunk_shape = Box(0.22, 0.34, 0.38)
v_trunk_color = np.array([0.1, 0.1, 0.1, 0.4])
# v_trunk_color = np.array([0., 0., 0., 0.])
X_v_trunk = RigidTransform()
X_v_trunk.set_translation(np.array([0.0, 0.0, 0.1]))
v_trunk_geometry = GeometryInstance(X_v_trunk, v_trunk_shape, "v_trunk")
if show_trunk_model:
    v_trunk_geometry.set_illustration_properties(MakePhongIllustrationProperties(v_trunk_color))
scene_graph.RegisterGeometry(v_trunk_source, v_trunk_frame.id(), v_trunk_geometry)
v_trunk_frame_ids = {"v_trunk":v_trunk_frame.id()}
print("v_trunk_frame.id()", v_trunk_frame.id())
# The visualizations are the desired foot trajectories from planner
for foot in ["lfoot", "rfoot"]:
    foot_frame = GeometryFrame(foot)
    scene_graph.RegisterFrame(v_trunk_source, foot_frame)

    foot_shape = Sphere(0.03)
    X_foot = RigidTransform()
    foot_geometry = GeometryInstance(X_foot, foot_shape, foot)
    if show_trunk_model:
        foot_geometry.set_illustration_properties(
            MakePhongIllustrationProperties(v_trunk_color))

    scene_graph.RegisterGeometry(v_trunk_source, foot_frame.id(), foot_geometry)
    v_trunk_frame_ids[foot] = foot_frame.id()
    print("trunk_frame_ids[foot]", v_trunk_frame_ids[foot])
########################## planner and  controller ############################
if planning_method == "basic":
    jump_running = False
    planner = builder.AddSystem(BasicTrunkPlanner(v_trunk_frame_ids))
elif planning_method == "towr":
    jump_running = False
    planner = builder.AddSystem(TowrTrunkPlanner(v_trunk_frame_ids))
elif planning_method == "opt_traj_matlab":
    jump_running = True
    planner = builder.AddSystem(OptPlanner(v_trunk_frame_ids))
else:
    print("Invalid planning method %s" % planning_method)
    sys.exit(1)
if control_method == "B":
    controller = builder.AddSystem(BasicController(plant, dt, p_model, jump_running, use_lcm=use_lcm))
elif control_method == "ID":
    controller = builder.AddSystem(IDController(plant, dt, p_model, jump_running, use_lcm=use_lcm))
elif control_method == "COMID":
    controller = builder.AddSystem(COMIDController(plant, dt, p_model, jump_running, use_lcm=use_lcm))
else:
    print("Invalid control method %s" % control_method)
    sys.exit(1)
# Create high-level trunk-model planner and low-level whole-body controller

# Set up the Scene Graph
# builder.Connect(
#         scene_graph.get_query_output_port(),
#         plant.get_geometry_query_input_port())
# builder.Connect(
#         plant.get_geometry_poses_output_port(),
#         scene_graph.get_source_pose_port(plant.get_source_id()))
builder.Connect(
        planner.GetOutputPort("trunk_geometry"),
        scene_graph.get_source_pose_port(v_trunk_source))
# Connect the trunk-model planner to the controller
if not control_method == "B":
    builder.Connect(planner.GetOutputPort("trunk_trajectory"), controller.get_input_port(1))

# Add loggers
logger = LogOutput(controller.GetOutputPort("output_metrics"), builder)

# Set up the Visualizer
DrakeVisualizer.AddToBuilder(builder, scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant)
# logger = LogOutput(plant.get_state_output_port(), builder)
# logger.set_name('Logger')

# Connect the controller to the simulated plant
builder.Connect(controller.GetOutputPort("biped_torques"),
                plant.get_actuation_input_port(biped))
builder.Connect(plant.get_state_output_port(),
                controller.GetInputPort("biped_state"))
# controller = builder.AddSystem(Controller(plant))
# builder.Connect(plant.get_state_output_port(), controller.get_input_port(0))
# builder.Connect(controller.get_output_port(0), plant.get_actuation_input_port())
# controller.set_name('DBFC-Controller')

# ground contact param settings
plant.set_penetration_allowance(0.001)
plant.set_stiction_tolerance(0.001)


# give names to the blocks (just to make the plot nicer)
#     draw the framework of controllers
#     display(SVG(pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].create_svg()))
diagram = builder.Build()
diagram.set_name("diagram")
diagram_context = diagram.CreateDefaultContext()

# Visualize the diagram
if show_diagram:
    plt.figure()
    plot_system_graphviz(diagram, max_depth=2)
    plt.show()

# Simulator setup
simulator = Simulator(diagram)
# simulator.Initialize()
simulator.set_target_realtime_rate(1.)

# set initial state (position q0 and velocity qd0)
plant_context = diagram.GetMutableSubsystemContext(
    plant, simulator.get_mutable_context())
# q0 = np.asarray([1.0, 0.0, 0.0, 0.0,  # base orientation
#                  0.0, 0.0, 1.19441,  # base position
#                  0.0, 0.0, 0.0, 0.0,  #
#                  0.0, 0.0, 0.0, 0.0,  #
#                  0.0, 0.0, 0.0, 0.0])  #
# q0[-6] = -0.05
# q0[-5] = 0.05

q0 = np.asarray([1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.044e+00,
                 -0.0e-01, 0.0e-01, -2.22e-03, 2.22e-03,
                5.583e-01, -5.583e-01, -1.174e+00, 1.174e+00,
                -6.24e-01, 6.24e-01, 2.25e-02, -2.25e-02])

arms_degree_setting = 20.  # set shoulder joints

# q_nom[0:4] = np.array([0.996, 0., 0.087, 0.])
# q_nom[11] = 80*math.pi/180
# q_nom[12] = -80*math.pi/180
q0[7] = arms_degree_setting * math.pi / 180
q0[8] = -arms_degree_setting * math.pi / 180

qd0 = np.zeros(plant.num_velocities())
plant.SetPositions(plant_context, q0)
plant.SetVelocities(plant_context, qd0)
# plant.get_actuation_input_port().FixValue(
#     plant_context, np.zeros(plant.num_actuators()))
################################### CHECK #####################################

foot_right = plant.GetFrameByName("foot_right")
foot_left = plant.GetFrameByName("foot_left")
world_frame = plant.world_frame()
p_right_Foot_contact = np.array([0,0,0])
p_left_Foot_contact = np.array([0,0,0])
p_WFoot_right_W = plant.CalcPointsPositions(context=plant_context, frame_B=foot_right, p_BQi=p_right_Foot_contact, frame_A=world_frame)
p_WFoot_left_W = plant.CalcPointsPositions(context=plant_context, frame_B=foot_left, p_BQi=p_left_Foot_contact, frame_A=world_frame)
print(f"p_WFoot_right_W{p_WFoot_right_W}")
print(f"p_WFoot_left_W{p_WFoot_left_W}")

com_init = plant.CalcCenterOfMassPositionInWorld(plant_context)
print(f"Initial CoM position:{com_init}")

###############################################################################
simulator.AdvanceTo(sim_time)

############################# generate figures ################################

# TODO plots
if make_plots:
    # logger that records the state trajectory during simulation
    global C_old
    C_old = np.array([[0.], [0.], [0.]])
    global L_old
    L_old = np.array([[0.], [0.], [0.]])
    global Time
    global floating_state_data
    global plt_data1, plt_data2, plt_data3, C_dot_current_data, AngularMomentum_data
    plt_data1 = []
    plt_data2 = []
    plt_data3 = []
    floating_state_data = []
    C_dot_current_data = []
    AngularMomentum_data = []

    # Plot stuff
    t = logger.sample_times()[10:]
    V = logger.data()[0,10:]
    err = logger.data()[1,10:]
    res = logger.data()[2,10:]
    Vdot = logger.data()[3,10:]

    plt.figure()
    #plt.subplot(4,1,1)
    #plt.plot(t, res, linewidth='2')
    #plt.ylabel("Residual")

    plt.subplot(3,1,1)
    plt.plot(t, Vdot, linewidth='2')
    plt.axhline(0,linestyle='dashed', color='grey')
    plt.ylabel("$\dot{V}$")

    plt.subplot(3,1,2)
    plt.plot(t, V, linewidth='2')
    plt.ylabel("$V$")

    plt.subplot(3,1,3)
    plt.plot(t, err, linewidth='2')
    plt.ylabel("$\|y_1-y_2\|^2$")
    plt.xlabel("time (s)")

    plt.show()
