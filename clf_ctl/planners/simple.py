import numpy as np
from pydrake.all import *
from pydrake.math import RollPitchYaw

class BasicTrunkPlanner(LeafSystem):
    """
    1.Implements the simplest possible trunk-model planner, which generates
    desired positions, velocities, and accelerations for the feet, center-of-mass,
    and body frame orientation.
    2.And implement the foot lifting, squatting.
    """
    def __init__(self, frame_ids):
        LeafSystem.__init__(self)

        # Dictionary of geometry frame ids {"trunk": trunk_frame_id, "lf": lf_foot_frame_id, ...}
        self.frame_ids = frame_ids

        # We'll use an abstract output port so we can send all the
        # data we'd like to include in a dictionary format
        self.DeclareAbstractOutputPort(
                "trunk_trajectory",
                lambda: AbstractValue.Make({}),
                self.SetTrunkOutputs)

        # Another output port is used to send geometry data regarding the
        # trunk model to the scene graph for visualization
        fpv = FramePoseVector()
        for frame in self.frame_ids:
            fpv.set_value(frame_ids[frame], RigidTransform())

        self.DeclareAbstractOutputPort(
                "trunk_geometry",
                lambda: AbstractValue.Make(fpv),
                self.SetGeometryOutputs)

        # The output data is a class-level object so we can be sure we're sending
        # the same info to the controller as to the scene graph
        self.output_dict = {}
        self.SimpleStanding()  # set initial values to self.output_dict

    def SimpleStanding(self):
        """
        Set output values corresponing to simply
        standing on all four feet.
        """
        # Foot positions
        self.output_dict["rpy_lfoot"] = np.zeros(3)
        self.output_dict["rpy_rfoot"] = np.zeros(3)
        self.output_dict["p_lfoot"] = np.array([ 0.0056, 0.104, 0.0])   # bit jumping robot
        self.output_dict["p_rfoot"] = np.array([ 0.005, -0.086, 0.0])
        # Foot velocities
        self.output_dict["rpyd_lfoot"] = np.zeros(3)
        self.output_dict["rpyd_rfoot"] = np.zeros(3)
        self.output_dict["pd_lfoot"] = np.zeros(3)
        self.output_dict["pd_rfoot"] = np.zeros(3)
        # Foot accelerations
        self.output_dict["rpydd_lfoot"] = np.zeros(3)
        self.output_dict["rpydd_rfoot"] = np.zeros(3)
        self.output_dict["pdd_lfoot"] = np.zeros(3)
        self.output_dict["pdd_rfoot"] = np.zeros(3)
        # Foot contact states: [lf,rf,lh,rh], True indicates being in contact.
        self.output_dict["contact_states"] = [True,True]
        # Foot contact forces, where each row corresponds to a foot [lfoot,rfoot].
        self.output_dict["f_cj"] = np.zeros((6,2))
        # Body pose
        self.output_dict["rpy_body"] = np.array([0.0, 0.0, 0.0])
        self.output_dict["p_body"] = np.array([0.0, 0.0, 1.19441-0.15]) # np.array([0.0+0.00082946, 0.0+0.10619907, 1.19441-0.15])
        # Body velocities
        self.output_dict["rpyd_body"] = np.zeros(3)
        self.output_dict["pd_body"] = np.zeros(3)
        # Body accelerations
        self.output_dict["rpydd_body"] = np.zeros(3)
        self.output_dict["pdd_body"] = np.zeros(3)
        # Max control input (accelerations)
        self.output_dict["u2_max"] = 0.0

    def OrientationTest(self, t):
        """
        Given the current time t, generate output values for
        for a simple orientation test.
        """
        self.SimpleStanding()
        self.output_dict["rpy_body"] = np.array([0.0, 0.2*np.sin(t), 0.*np.cos(t)])
        self.output_dict["rpyd_body"] = np.array([0.0, 0.2*np.cos(t), -0.*np.sin(t)])
        self.output_dict["rpydd_body"] = np.array([0.0, -0.2*np.sin(t), -0.*np.cos(t)])
        self.output_dict["p_body"] = np.array([ 0.06*np.sin(0.4 * t), 0.0, 1.19441 - 0.15 - 0.12*np.sin(0.4 * t)])
        # self.output_dict["p_body"] = np.array([0.0, 0.0, -0.13 * 0.5 * np.cos(0.5 * t)])
        # self.output_dict["p_body"] = np.array([0.0, 0.0, 0.13 * 0.5 * 0.5 * np.sin(0.5 * t)])

    def RaiseFoot(self, t):
        """
        Modify the simple standing output values to lift one foot
        off the ground.
        """
        self.SimpleStanding()

        if t>2.0 and t<6:
            self.output_dict["p_body"] = np.array([-0.00082946 * (t-2.0) / (4), 0.10619907 * (t-2.0) / (4), 1.19441-0.15])
            # To lift the right leg
            if t>5.0 and t<6:
                self.output_dict["contact_states"] = [True,False]
                self.output_dict["p_rfoot"] = np.array([  0.00058538 , -0.08807213 , 0.35* (t-5.0)])# p_WFoot_right_W[[ 0.00058538][-0.08807213][ 0.0008917 ]
        # To keep right foot on air
        if t>=6:
            self.output_dict["contact_states"] = [True, False]
            self.output_dict["p_rfoot"] = np.array([0.00058538 , -0.08807213 , 0.35 ])
            self.output_dict["p_body"] = np.array([-0.00082946, 0.10619907,
                                                   1.19441 - 0.15 ])
        # To squat after 8s
        if t>=8:
            # self.output_dict["rpy_body"] = np.array(
            #     [0.0, 0.2 * np.sin(t-8), 0. * np.cos(t)])
            # self.output_dict["rpyd_body"] = np.array(
            #     [0.0, 0.2 * np.cos(t-8), -0. * np.sin(t)])
            # self.output_dict["rpydd_body"] = np.array(
            #     [0.0, -0.2 * np.sin(t-8), -0. * np.cos(t)])
            self.output_dict["p_body"] = np.array([-0.00082946 , 0.10619907, 1.19441 - 0.15- 0.1*np.sin(1 * (t-8))])

    def EdgeTest(self):
        """
        Move the trunk right to the edge of feasibility, ensuring that
        friction constraints become active (may require a smaller timestep)
        """
        self.SimpleStanding()
        self.output_dict["p_body"] += np.array([-0.1, 0.63, 0.0])

    def SetTrunkOutputs(self, context, output):
        self.output_dict = output.get_mutable_value()

        self.SimpleStanding()
        #self.output_dict["p_body"] += np.array([0,0,0.05])
        # self.OrientationTest(context.get_time())
        # self.EdgeTest()
        self.RaiseFoot(context.get_time())

    def SetGeometryOutputs(self, context, output):
        fpv = output.get_mutable_value()
        fpv.clear()

        X_trunk = RigidTransform()
        X_trunk.set_rotation(RollPitchYaw(self.output_dict["rpy_body"]))
        X_trunk.set_translation(self.output_dict["p_body"])

        fpv.set_value(self.frame_ids["v_trunk"], X_trunk)

        for foot in ["lfoot","rfoot"]:
            X_foot = RigidTransform()
            X_foot.set_translation(self.output_dict["p_%s" % foot])
            fpv.set_value(self.frame_ids[foot], X_foot)
