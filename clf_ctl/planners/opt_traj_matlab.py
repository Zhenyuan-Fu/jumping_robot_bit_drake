import math

import numpy as np
from pydrake.all import *
from pydrake.math import RollPitchYaw
import scipy.io as sio
import matplotlib.pyplot as plt


class OptPlanner(LeafSystem):
    """
    Implements the jumping planner, which generates
    desired positions, velocities, and accelerations for the feet, center-of-mass,
    and body orientation.
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

        load_path = '/home/zhenyuan_fu/PycharmProjects/jumping_robot_bit/clf_ctl/opti_trajectories/four_ms_traj.mat'
        load_data = sio.loadmat(load_path)
        plt.close('all')

        traj_total = load_data['traj']

        time_opti = traj_total[0, 0][0]
        self.dt = 0.004
        time_opti_index = 1 / self.dt * time_opti

        q_opti = traj_total[0, 0][1]
        qd_opti = traj_total[0, 0][2]
        torques_opti = traj_total[0, 0][3]

        self.com_pos_opti = traj_total[0, 0][4]
        self.com_pos_opti[[1, 2], :] = self.com_pos_opti[[2, 1], :]
        com_z_offset = 0.62 - 0.5789372527399674
        self.com_pos_opti[2, :] = self.com_pos_opti[2, :] + com_z_offset

        self.com_vel_opti = traj_total[0, 0][5]
        self.com_vel_opti[[1, 2], :] = self.com_vel_opti[[2, 1], :]

        self.com_acc_opti = traj_total[0, 0][6]
        self.com_acc_opti[[1, 2], :] = self.com_acc_opti[[2, 1], :]

        self.pitch_opti = - q_opti[2, :]
        self.pitchd_opti = - qd_opti[2, :]

        # print(time_opti_index[0])
        # print(com_pos_opti) # 这一时刻的数值

    def SimpleStanding(self):
        """
        Set output values corresponing to simply
        standing on all four feet.
        """
        # Foot positions
        self.output_dict["rpy_lfoot"] = np.zeros(3)
        self.output_dict["rpy_rfoot"] = np.zeros(3)
        self.output_dict["p_lfoot"] = np.array(
            [0.0056, 0.104, 0.0])  # bit jumping robot
        self.output_dict["p_rfoot"] = np.array([0.005, -0.086, 0.0])
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
        self.output_dict["contact_states"] = [True, True]
        # Foot contact forces, where each row corresponds to a foot [lfoot,rfoot].
        self.output_dict["f_cj"] = np.zeros((6, 2))
        # Body pose
        self.output_dict["rpy_body"] = np.array([0.0, 0.0, 0.0])
        self.output_dict["p_body"] = np.array([0.0, 0.0,
                                               1.19441 - 0.15])  # np.array([0.0+0.00082946, 0.0+0.10619907, 1.19441-0.15])
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
        self.output_dict["rpy_body"] = np.array(
            [0.0, 0.2 * np.sin(t), 0. * np.cos(t)])
        self.output_dict["rpyd_body"] = np.array(
            [0.0, 0.2 * np.cos(t), -0. * np.sin(t)])
        self.output_dict["rpydd_body"] = np.array(
            [0.0, -0.2 * np.sin(t), -0. * np.cos(t)])
        self.output_dict["p_body"] = np.array([0.06 * np.sin(0.4 * t), 0.0,
                                               1.19441 - 0.15 - 0.12 * np.sin(
                                                   0.4 * t)])
        # self.output_dict["p_body"] = np.array([0.0, 0.0, -0.13 * 0.5 * np.cos(0.5 * t)])
        # self.output_dict["p_body"] = np.array([0.0, 0.0, 0.13 * 0.5 * 0.5 * np.sin(0.5 * t)])

    def RaiseFoot(self, t):
        """
        Modify the simple standing output values to lift one foot
        off the ground.
        """
        self.SimpleStanding()

        if t > 2.0 and t < 6:
            self.output_dict["p_body"] = np.array(
                [-0.00082946 * (t - 2.0) / (4), 0.10619907 * (t - 2.0) / (4),
                 1.19441 - 0.15])
            # if t>5.0 and t<18:
            #     self.output_dict["p_body"] += np.array([-0.00082946 / (3.0 / 0.005), 0.10619907 / (3.0 / 0.005),0.0])  # self.output_dict["p_body"] = np.array([0.0, 0.0, 1.19441-0.15]) # np.array([0.0+0.00082946, 0.0+0.10619907, 1.19441-0.15])
            if t > 5.0 and t < 6:
                self.output_dict["contact_states"] = [True, False]
                self.output_dict["p_rfoot"] = np.array(
                    [0.00058538, -0.08807213, 0.35 * (
                        t - 5.0)])  # p_WFoot_right_W[[ 0.00058538][-0.08807213][ 0.0008917 ]
        if t >= 6:
            self.output_dict["contact_states"] = [True, False]
            self.output_dict["p_rfoot"] = np.array(
                [0.00058538, -0.08807213, 0.35])
            self.output_dict["p_body"] = np.array([-0.00082946, 0.10619907,
                                                   1.19441 - 0.15])
        if t >= 8:
            # self.output_dict["rpy_body"] = np.array(
            #     [0.0, 0.2 * np.sin(t-8), 0. * np.cos(t)])
            # self.output_dict["rpyd_body"] = np.array(
            #     [0.0, 0.2 * np.cos(t-8), -0. * np.sin(t)])
            # self.output_dict["rpydd_body"] = np.array(
            #     [0.0, -0.2 * np.sin(t-8), -0. * np.cos(t)])
            self.output_dict["p_body"] = np.array([-0.00082946, 0.10619907,
                                                   1.19441 - 0.15 - 0.1 * np.sin(
                                                       1 * (t - 8))])

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
        # self.output_dict["p_body"] += np.array([0,0,0.05])
        # self.OrientationTest(context.get_time())
        # self.EdgeTest()
        # self.RaiseFoot(context.get_time())
        self.To_jump(context.get_time())

    def SetGeometryOutputs(self, context, output):
        fpv = output.get_mutable_value()
        fpv.clear()

        X_trunk = RigidTransform()
        X_trunk.set_rotation(RollPitchYaw(self.output_dict["rpy_body"]))
        X_trunk.set_translation(self.output_dict["p_body"])

        fpv.set_value(self.frame_ids["v_trunk"], X_trunk)

        for foot in ["lfoot", "rfoot"]:
            X_foot = RigidTransform()
            X_foot.set_translation(self.output_dict["p_%s" % foot])
            fpv.set_value(self.frame_ids[foot], X_foot)

    ############ 以下为跳跃轨迹 #########
    # def Jumping(self, t):
    #
    #     # 取每一时刻的数据
    #     p_com_d = com_z_pos_d[t]
    #     pd_com_d = com_z_vel_d[t]
    #     pdd_com_d = com_z_acc_d[t]
    #     self.output_dict["rpy_body"] = np.array(
    #         [0.0, 0.0 * np.sin(t), 0. * np.cos(t)])
    #     self.output_dict["rpyd_body"] = np.array(
    #         [0.0, 0.0 * np.cos(t), -0. * np.sin(t)])
    #     self.output_dict["rpydd_body"] = np.array(
    #         [0.0, -0.0 * np.sin(t), -0. * np.cos(t)])
    #     self.output_dict["p_com"] = np.array(p_com_d)
    #     self.output_dict["pd_com"] = np.array(pd_com_d)
    #     self.output_dict["pdd_com"] = np.array(pdd_com_d)

        # TODO to track feet traj during flight and landing
        # global feet_pos_d, feet_vel_d, feet_acc_d
        # data_name = '/home/ryanfu/underactuated/Com_Traj/Com_pos_vel_acc_reference.dat'
        #
        # com_file = open(data_name, 'r')
        #
        # feet_pos_d = []
        # feet_vel_d = []
        # feet_acc_d = []
        # for line in com_file:
        #     line = line.strip().split('\t')
        #     feet_pos_d.append(float(line[0]))
        #     feet_vel_d.append(float(line[1]))
        #     feet_acc_d.append(float(line[2]))
        # com_file.close()
        # # get desired traj data in every loop
        # p_feet_d = feet_pos_d[t]
        # pd_feet_d = feet_vel_d[t]
        # pdd_feet_d = feet_acc_d[t]
        # self.output_dict["rpy_body"] = np.array([0.0, 0.0*np.sin(t), 0.*np.cos(t)])
        # self.output_dict["rpyd_body"] = np.array([0.0, 0.0*np.cos(t), -0.*np.sin(t)])
        # self.output_dict["rpydd_body"] = np.array([0.0, -0.0*np.sin(t), -0.*np.cos(t)])
        # self.output_dict["p_lfoot"] = np.array([ 0.0056, 0.104, 0.0])   # bit jumping robot
        # self.output_dict["p_rfoot"] = np.array([ 0.005, -0.086, 0.0])
        # self.output_dict["p_com"] = np.array(p_feet_d)
        # self.output_dict["pd_com"] = np.array(pd_feet_d)
        # self.output_dict["pdd_com"] = np.array(pdd_feet_d)

    def To_jump(self, t):
        # desired CoM when standing
        p_com_x = 0.005
        p_com_y = 0.000
        p_com_z = 0.620

        # initial CoM when t=0
        p_com_x_t0 = 0.011751
        p_com_y_t0 = 0.00726061
        p_com_z_t0 = 0.88974034

        # pitch setting
        pitch_setting = 20 * math.pi / 180

        # state setting: firstly, robot squat to target CoM, then take off
        # attention: there is no trajs for flight and landing!!
        t_move = 3.0
        t_prepare = 4.0
        t_takeoff = t_prepare + 0.40

        if t < t_move:
            self.output_dict["p_com"] = np.array(
                [p_com_x_t0 + ((p_com_x - p_com_x_t0) * t / t_move),
                 p_com_y_t0 + ((p_com_y - p_com_y_t0) * t / t_move),
                 p_com_z_t0 + ((p_com_z - p_com_z_t0) * t / t_move)])
            self.output_dict["rpy_body"] = np.array(
                [0.0, pitch_setting * t / t_move, 0.0])
            self.output_dict["pd_com"] = np.zeros(3)
            self.output_dict["pdd_com"] = np.zeros(3)
        elif t_move <= t < t_prepare:
            self.output_dict["p_com"] = np.array([p_com_x, p_com_y, p_com_z])
            self.output_dict["pd_com"] = np.zeros(3)
            self.output_dict["pdd_com"] = np.zeros(3)
            self.output_dict["rpy_body"] = np.array([0.0, pitch_setting, 0.0])
        elif t_prepare <= t < t_takeoff:
            time_index = int(1/self.dt * (t-t_prepare))
            self.output_dict["p_com"] = self.com_pos_opti[:, time_index]
            self.output_dict["pd_com"] = self.com_vel_opti[:, time_index]
            self.output_dict["pdd_com"] = self.com_acc_opti[:, time_index]
            self.output_dict["rpy_body"] = np.array([0.0, self.pitch_opti[time_index], 0.0])
            self.output_dict["rpyd_body"] = np.array([0.0, self.pitchd_opti[time_index], 0.0])
            # self.output_dict["rpy_body"] = np.array([0.0, self.q_opti[, time_index], 0.0])
            # self.output_dict["rpyd_body"] = np.array([0.0, pitch_setting, 0.0])
        elif t >= t_takeoff :
            self.output_dict["p_com"] = np.array([p_com_x, p_com_y, p_com_z])
            self.output_dict["pd_com"] = np.zeros(3)
            self.output_dict["pdd_com"] = np.zeros(3)
            self.output_dict["rpy_body"] = np.array([0.0, pitch_setting, 0.0])
        # pd_com pdd_com setting


        # Foot positions
        self.output_dict["rpy_lfoot"] = np.zeros(3)
        self.output_dict["rpy_rfoot"] = np.zeros(3)
        self.output_dict["p_lfoot"] = np.array(
            [0.0056, 0.104, 0.0])  # bit jumping robot
        self.output_dict["p_rfoot"] = np.array([0.005, -0.086, 0.0])
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
        self.output_dict["contact_states"] = [True, True]
        # Foot contact forces, where each row corresponds to a foot [lfoot,rfoot].
        self.output_dict["f_cj"] = np.zeros((6, 2))

        self.output_dict["u2_max"] = 0.0


        # # Body pose
        # self.output_dict["rpy_body"] = np.array([0.0, pitch_setting, 0.0])
        # self.output_dict["p_body"] = np.array([0.0, 0.0,
        #                                        1.19441 - 0.15])  # np.array([0.0+0.00082946, 0.0+0.10619907, 1.19441-0.15])
        # # Body velocities
        # self.output_dict["rpyd_body"] = np.zeros(3)
        # self.output_dict["pd_body"] = np.zeros(3)
        # # Body accelerations
        # self.output_dict["rpydd_body"] = np.zeros(3)
        # self.output_dict["pdd_body"] = np.zeros(3)

        # Max control input (accelerations)

