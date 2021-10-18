import math

import numpy as np
from pydrake.math import RollPitchYaw
from controllers.basic_controller import *

from pinocchio.utils import *
from pinocchio import computeCentroidalDynamics, \
    computeCentroidalMapTimeVariation, RobotWrapper
import pinocchio
from pydrake.multibody.tree import JointActuator_


def JointDrakeToPin():
    joint_drake2pin = np.zeros((12, 12))
    joint_drake2pin[0, 2] = 1
    joint_drake2pin[1, 4] = 1
    joint_drake2pin[2, 6] = 1
    joint_drake2pin[3, 8] = 1
    joint_drake2pin[4, 10] = 1
    joint_drake2pin[5, 3] = 1
    joint_drake2pin[6, 5] = 1
    joint_drake2pin[7, 7] = 1
    joint_drake2pin[8, 9] = 1
    joint_drake2pin[9, 11] = 1
    joint_drake2pin[10, 0] = 1
    joint_drake2pin[11, 1] = 1
    return joint_drake2pin


class COMIDController(BasicController):
    """
    A standard QP-based inverse dynamics controller (to follow COM, trunk's ori,
    angular momentum and feet).

    Takes as input desired positions/velocities/accelerations of the
    feet and floating base and computes corresponding joint torques.
    """

    def __init__(self, plant, dt,  p_model, jump_running, use_lcm=False):
        BasicController.__init__(self, plant, dt, p_model, jump_running, use_lcm=use_lcm)

        # inputs from the trunk model are sent in a dictionary
        self.DeclareAbstractInputPort(
            "trunk_input",
            AbstractValue.Make({}))

        # Set the friction coefficient
        self.mu = 0.7
        self.x = 0.220 / 2.0
        self.y = 0.107 / 2.0
        # change according to actual env
        self.max_Fz_ = 1600

        # Choose a solver
        # self.solver = GurobiSolver()
        self.solver = OsqpSolver()

    def AddJacobianTypeCost(self, J, qdd, Jd_qd, xdd_des, weight=1.0):
        """
        Add a quadratic cost of the form
            weight*| J*qdd + Jd_qd - xdd_des |^2
        to the whole-body controller QP.
        """
        # Put in the form 1/2*qdd'*Q*qdd + c'*qdd for fast formulation
        Q = weight * np.dot(J.T, J)
        c = weight * (np.dot(Jd_qd.T, J) - np.dot(xdd_des.T, J)).T

        return self.mp.AddQuadraticCost(Q, c, qdd)

    def AddJacobianTypeCostFoot(self, J, qdd, Jd_qd, xdd_des, weight=1.0):
        """
        Add a quadratic cost of the form
            weight*| J*qdd + Jd_qd - xdd_des |^2
        to the whole-body controller QP.
        """
        # Put in the form 1/2*qdd'*Q*qdd + c'*qdd for fast formulation
        Q = weight * np.dot(J[:-3, :].T, J[:-3, :])
        c = weight * (np.dot(Jd_qd[:-3, :].T, J[:-3, :]) - np.dot(xdd_des.T,
                                                                  J[:-3, :])).T

        return self.mp.AddQuadraticCost(Q, c, qdd)

    def AddJacobianTypeConstraint(self, J, qdd, Jd_qd, xdd_des):
        """
        Add a linear constraint of the form
            J*qdd + Jd_qd == xdd_des
        to the whole-body controller QP.
        """
        A_eq = J  # A_eq*qdd == b_eq
        b_eq = xdd_des - Jd_qd
        return self.mp.AddLinearEqualityConstraint(A_eq, b_eq, qdd)

    def AddDynamicsConstraint(self, M, qdd, C, tau_g, S, tau, J_c, f_c):
        """
        Add a dynamics constraint of the form
            M*qdd + Cv + tau_g == S'*tau + sum(J_c[i]'*f_c[i])
        to the whole-body controller QP.
        """
        # We'll rewrite the constraints in the form A_eq*x == b_eq for speed
        A_eq = np.hstack([M, -S.T])
        x = np.vstack([qdd, tau])

        for j in range(len(J_c)):
            A_eq = np.hstack([A_eq, -J_c[j].T])
            x = np.vstack([x, f_c[j]])

        b_eq = -C - tau_g

        return self.mp.AddLinearEqualityConstraint(A_eq, b_eq, x)

    def AddFrictionPyramidConstraint(self, f_c, Rot_foot_left_mtx,
                                     Rot_foot_right_mtx, swing_feet):
        """
        Add a friction pyramid constraint for the given set of contact forces
        to the whole-body controller QP,
        "Stability of Surface Contacts for Humanoid Robots:
        Closed-Form Formulae of the Contact Wrench Cone
        for Rectangular Support Areas"(by Stéphane Caron).
        """
        num_contacts = len(f_c)

        # pyramid approximation of CWC for contact force f \in R^6
        A_i = np.asarray([[0, 0, 0, 0, 0, 1],  # 0.
                          [0, 0, 0, 1, 0, self.mu],  # 1. ----
                          [0, 0, 0, -1, 0, self.mu],  # 2.
                          [0, 0, 0, 0, 1, self.mu],  # 3.
                          [0, 0, 0, 0, -1, self.mu],  # 4.
                          [1, 0, 0, 0, 0, self.y],  # 5.
                          [-1, 0, 0, 0, 0, self.y],  # 6.
                          [0, 1, 0, 0, 0, self.x],  # 7.
                          [0, -1, 0, 0, 0, self.x],  # 8.
                          [-self.mu, -self.mu, 1, self.y, self.x,
                           (self.x + self.y) * self.mu],  # 9. tau ----
                          [-self.mu, self.mu, 1, self.y, -self.x,
                           (self.x + self.y) * self.mu],  # 10.
                          [self.mu, -self.mu, 1, -self.y, self.x,
                           (self.x + self.y) * self.mu],  # 11.
                          [self.mu, self.mu, 1, -self.y, -self.x,
                           (self.x + self.y) * self.mu],  # 12.
                          [-self.mu, -self.mu, -1, -self.y, -self.x,
                           (self.x + self.y) * self.mu],  # 13. -----
                          [-self.mu, self.mu, -1, -self.y, self.x,
                           (self.x + self.y) * self.mu],  # 14.
                          [self.mu, -self.mu, -1, self.y, -self.x,
                           (self.x + self.y) * self.mu],  # 15.
                          [self.mu, self.mu, -1, self.y, self.x,
                           (self.x + self.y) * self.mu],  # 16.
                          [0, 0, 0, 0, 0, -1]])  # 17.
        # We'll formulate as lb <= Ax <= ub, where x=[f_1',f_2',...]'

        Rot_foot_left = np.block(
            [[Rot_foot_left_mtx.transpose(), np.zeros((3, 3))],
             [np.zeros((3, 3)), Rot_foot_left_mtx.transpose()]])
        Rot_foot_right = np.block(
            [[Rot_foot_right_mtx.transpose(), np.zeros((3, 3))],
             [np.zeros((3, 3)), Rot_foot_right_mtx.transpose()]])
        A_left_f = A_i @ Rot_foot_left
        A_right_f = A_i @ Rot_foot_right
        if swing_feet[0] == False and swing_feet[1] == False:
            A = np.block([[A_left_f, np.zeros((18, 6))],
                          [np.zeros((18, 6)), A_right_f]])
            lb = np.zeros((18 * num_contacts, 1))
            ub = np.inf * np.ones((18 * num_contacts, 1))
            lb[17] = - self.max_Fz_
            lb[35] = - self.max_Fz_
            # contact variables
            x_ = np.vstack([f_c[j] for j in range(num_contacts)])
        elif swing_feet[0] == True and swing_feet[1] == True:
            pass
        elif swing_feet[0] == True and swing_feet[1] == False:
            A = A_right_f
            lb = np.zeros((18 * num_contacts, 1))
            ub = np.inf * np.ones((18 * num_contacts, 1))
            lb[17] = - self.max_Fz_
            # contact variables
            x_ = np.vstack([f_c[j] for j in range(num_contacts)])
        elif swing_feet[0] == False and swing_feet[1] == True:
            A = A_left_f
            lb = np.zeros((18 * num_contacts, 1))
            ub = np.inf * np.ones((18 * num_contacts, 1))
            lb[17] = - self.max_Fz_
            # contact variables
            x_ = np.vstack([f_c[j] for j in range(num_contacts)])

        return self.mp.AddLinearConstraint(A=A, lb=lb, ub=ub, vars=x_)

    def AddContactConstraint(self, J_c, vd, Jdv_c, v):
        """
        Add contact constraints with velocity damping

            J_c[j]*vd + Jdv_c[j] == -Kd*J_c[j]*v
        """
        Kd_angular = 1 * np.eye(3)
        Kd_linear = 10 * np.eye(3)
        Kd = np.block([[Kd_angular, np.zeros((3, 3))],
                       [np.zeros((3, 3)), Kd_linear]])

        num_contacts = len(J_c)
        # num_contacts = 1
        for j in range(num_contacts):
            j = 1
            pd = (J_c[j] @ v).reshape(6, 1)
            # print(f"{j}J_c[j, :, :]", J_c[j, :, :])
            pdd_des = -Kd @ pd
            print(f"{j}pd{pd}")
            # print(f"pdd_des{pdd_des}")
            # print("2Jdv_c", Jdv_c)
            constraint = self.AddJacobianTypeConstraint(J_c[j], vd, Jdv_c[j],
                                                        pdd_des)

    def ControlLaw(self, context, q, v):
        """
        A standard inverse dynamics whole-body QP:

           minimize:
               w_body* || J_body*vd + Jd_body*v - vd_body_des ||^2 +
               w_s_foot* || J_s*vd+ Jd_s*v - pdd_s_des ||^2+
               w_c_foot* || J_s*vd+ Jd_s*v - damping ||^2+
               w_hg* || Ag*vd+ Ag_dot*v - hg_des ||^2+
               w_shoulder_torque* || shoulder_taus ||^2+
               w_shoulder_pos* || shoulder_joints_pos_error ||^2
           subject to:
                M*vd + Cv + tau_g = S'*tau + sum(J'*f)
                f \in friction cones
                J_cj*vd + Jd_cj*v == 0

           since we don't have reference trajectories for shoulders,
           we add shoulder_torque and shoulder_pos in cost. Otherwise,
           shoulders will be out of control.

           For jumping, the standard QP-based inverse dynamics controller
           will follow: CoM acceleration,
                        trunk's ori acceleration (no trunk's linear part),
                        feet acceleration,
                        angular momentum.

        """

        ######### Tuning Parameters #########
        Kp_com_p = 40.0
        Kd_com_p = 4.0

        # Kp_body_p = 80.0
        # Kd_body_p = 10.5
        Kp_body_p = 40.0
        Kd_body_p = 4.

        # Kp_body_rpy = 0.2*Kp_body_p
        # Kd_body_rpy = 0.04*Kd_body_p
        Kp_body_rpy = 0.3 * Kp_body_p
        Kd_body_rpy = 0.3 * Kd_body_p

        Kp_foot = 40.0
        Kd_foot = 4.

        Kp_foot_rpy = Kp_foot
        Kd_foot_rpy = Kd_foot

        Kp_arm = 40
        Kd_arm = 4.01

        Kp_hg = 100.0
        Kd_hg = 100

        kk_ = 0.3  # pd setting of contact foot

        # w_body = 1.0
        # w_s_foot = 15.0

        w_com = 70.0  # CoM task weight
        w_body = 70.0  # body ori task weight
        w_s_foot = 5.0  # swing foot task weight
        w_c_foot = 17000.0  # contact foot task weight
        w_hg = 0.24  # angular momentum task weight

        w_shoulder_torque = 1  # shoulder joint torque weight
        w_shoulder_pos = 3  # Shoulder joint pos task weight
        #####################################

        # Compute Dynamics Quantities
        M, Cv, tau_g, S = self.CalcDynamics()

        # Get setpoint data from the trunk model
        trunk_data = self.EvalAbstractInput(context, 1).get_value()

        contact_feet = trunk_data[
            "contact_states"]  # Note: it may be better to determine
        swing_feet = [not foot for foot in
                      contact_feet]  # contact states from the actual robot rather than
        num_contact = sum(contact_feet)  # the planned trunk trajectory.
        num_swing = sum(swing_feet)

        # p: planned position; pd: planned velocity; pdd: planned acceleration
        p_com_nom = trunk_data["p_com"]  # CoM planned position
        pd_com_nom = trunk_data["pd_com"]  # CoM planned velocity
        pdd_com_nom = trunk_data["pdd_com"]  # CoM planned acceleration

        p_body_nom = trunk_data["p_body"]  # Trunk planned position
        pd_body_nom = trunk_data["pd_body"]  # Trunk planned velocity
        pdd_body_nom = trunk_data["pdd_body"]  # Trunk planned acceleration

        rpy_body_nom = trunk_data["rpy_body"]  # Trunk planned row,pitch,yaw
        rpyd_body_nom = trunk_data["rpyd_body"]  # Derivative of Trunk planned row,pitch,yaw
        rpydd_body_nom = trunk_data["rpydd_body"]  # Double derivative of Trunk planned row,pitch,yaw

        p_feet_nom = np.array(
            [trunk_data["p_lfoot"], trunk_data["p_rfoot"]])
        pd_feet_nom = np.array(
            [trunk_data["pd_lfoot"], trunk_data["pd_rfoot"]])
        pdd_feet_nom = np.array(
            [trunk_data["pdd_lfoot"], trunk_data["pdd_rfoot"]])

        rpy_foot_nom = np.array([trunk_data["rpy_lfoot"],
                                 trunk_data["rpy_rfoot"]])
        rpyd_foot_nom = np.array([trunk_data["rpyd_lfoot"],
                                  trunk_data["rpyd_rfoot"]])
        rpydd_foot_nom = np.array([trunk_data["rpydd_lfoot"],
                                   trunk_data["rpydd_rfoot"]])

        p_s_nom = p_feet_nom[swing_feet]
        pd_s_nom = pd_feet_nom[swing_feet]
        pdd_s_nom = pdd_feet_nom[swing_feet]
        rpy_foot_s_nom = rpy_foot_nom[swing_feet]
        rpyd_foot_s_nom = rpyd_foot_nom[swing_feet]
        rpydd_foot_s_nom = rpydd_foot_nom[swing_feet]

        # Get robot's actual com position, velocity, jacobian, jacobian bias

        p_com, J_com, Jdv_com = self.CalcComQuantities()
        pd_com = (J_com @ v)  # J_com has 3 rows

        # Get robot's actual task-space (body pose + foot pose) data
        X_body, J_body, Jdv_body = self.CalcFramePoseQuantities(
            self.body_frame)
        p_body = X_body.translation()
        pd_body = (J_body @ v)[3:]

        RPY_body = RollPitchYaw(
            X_body.rotation())  # RPY object helps convert between angular velocity and rpyd
        rpy_body = RPY_body.vector()
        omega_body = (J_body @ v)[:3]  # angular velocity of the body
        rpyd_body = RPY_body.CalcRpyDtFromAngularVelocityInParent(omega_body)
        # left foot
        X_lf, J_lf, Jdv_lf_T = self.CalcFramePoseQuantities(
            self.lfoot_frame)

        Jdv_lf = Jdv_lf_T.reshape(6, 1)

        p_lf = X_lf.translation()
        pd_lf = (J_lf @ v)[3:]

        RPY_lf = RollPitchYaw(X_lf.rotation())
        rpy_lf = RPY_lf.vector()
        omega_lf = (J_lf @ v)[:3]
        rpyd_lf = RPY_lf.CalcRpyDtFromAngularVelocityInParent(omega_lf)
        # right foot
        X_rf, J_rf, Jdv_rf_T = self.CalcFramePoseQuantities(
            self.rfoot_frame)
        Jdv_rf = Jdv_rf_T.reshape(6, 1)

        p_rf = X_rf.translation()
        pd_rf = (J_rf @ v)[3:]

        RPY_rf = RollPitchYaw(X_rf.rotation())
        rpy_rf = RPY_rf.vector()
        omega_rf = (J_rf @ v)[:3]
        rpyd_rf = RPY_rf.CalcRpyDtFromAngularVelocityInParent(omega_rf)

        # Swing foot actual ori and pos
        p_feet = np.array([p_lf, p_rf]).reshape(2, 3)
        pd_feet = np.array([pd_lf, pd_rf]).reshape(2, 3)  # 左右脚linear J*v
        rpy_feet = np.array([rpy_lf, rpy_rf]).reshape(2, 3)
        rpyd_feet = np.array([rpyd_lf, rpyd_rf]).reshape(2, 3)
        J_feet = np.array([J_lf, J_rf])
        Jdv_feet = np.array([Jdv_lf, Jdv_rf])

        p_s = p_feet[swing_feet]
        pd_s = pd_feet[swing_feet]
        rpy_foot_s = rpy_feet[swing_feet]
        rpyd_foot_s = rpyd_feet[swing_feet]

        J_c = J_feet[contact_feet]
        J_s = J_feet[swing_feet]
        Jdv_c = Jdv_feet[contact_feet]
        Jdv_s = Jdv_feet[swing_feet]

        # set com desired task-space accelerations
        pdd_com_des = pdd_com_nom - Kp_com_p * (p_com - p_com_nom) \
                      - Kd_com_p * (pd_com - pd_com_nom)

        # Set desired task-space accelerations
        pdd_body_des = pdd_body_nom - Kp_body_p * (p_body - p_body_nom) \
                       - Kd_body_p * (pd_body - pd_body_nom)

        # global pd_body_pre
        # if context.get_time() < 0.1:
        #     pd_body_pre = np.zeros(3)
        # pd_body_error = pd_body_pre-pd_body / self.dt
        # print(f"{pd_body_error}")
        # if context.get_time() > 0.1:
        #     pd_body_pre = pd_body

        # print(f"qdd{pd_body}")

        rpydd_body_des = rpydd_body_nom - Kp_body_rpy * (
            rpy_body - rpy_body_nom) \
                         - Kd_body_rpy * (rpyd_body - rpyd_body_nom)
        omegad_body_des = RPY_body.CalcAngularVelocityInParentFromRpyDt(
            rpydd_body_des)

        vd_body_des = np.hstack([omegad_body_des,
                                 pdd_body_des])  # desired spatial acceleration of the body
        # Ser swing foot task-space accelerations

        pdd_s_des = pdd_s_nom - Kp_foot * (p_s - p_s_nom) \
                    - Kd_foot * (pd_s - pd_s_nom)

        # print(f"pdd_s_nom{pdd_s_nom}\nKp_foot{Kp_foot}\np_s{p_s}\np_s_nom{p_s_nom}")
        # print(f"Kd_foot{Kd_foot}\npd_s{pd_s}\npd_s_nom{pd_s_nom}")
        # print(f"swing_feet{swing_feet}")
        # print(f"pdd_s_des{pdd_s_des}")
        rpydd_foot_des = rpydd_foot_s_nom - Kp_foot_rpy * (
            rpy_foot_s - rpy_foot_s_nom) - Kd_foot_rpy * (
                             rpyd_foot_s - rpyd_foot_s_nom)
        if swing_feet[0] == False and swing_feet[1] == False:
            pass
        elif swing_feet[0] == True and swing_feet[1] == True:
            omegad_lfrf_des = np.array([
                RPY_lf.CalcAngularVelocityInParentFromRpyDt(
                    rpydd_foot_des[0]),
                RPY_rf.CalcAngularVelocityInParentFromRpyDt(
                    rpydd_foot_des[1])])
            vd_foot_des = np.hstack([omegad_lfrf_des, pdd_s_des])
        elif swing_feet[0] == True and swing_feet[1] == False:
            omegad_lfrf_des = np.array([
                RPY_lf.CalcAngularVelocityInParentFromRpyDt(
                    rpydd_foot_des[0])])
            vd_foot_des = np.hstack([omegad_lfrf_des, pdd_s_des])
        elif swing_feet[0] == False and swing_feet[1] == True:
            omegad_lfrf_des = np.array([
                RPY_rf.CalcAngularVelocityInParentFromRpyDt(
                    rpydd_foot_des[0])])
            vd_foot_des = np.hstack([omegad_lfrf_des, pdd_s_des])

        ################################ pin Ag Agd ###########################

        # using pinocchio to compute Ag Ag_dot
        p_data = pinocchio.Data(self.p_model)
        p_q = np.array(q)
        v_raw_ = np.array(v)
        d2p = JointDrakeToPin()

        p_q[0] = q[4]
        p_q[1] = q[5]
        p_q[2] = q[6]
        p_q[3] = q[1]
        p_q[4] = q[2]
        p_q[5] = q[3]
        p_q[6] = q[0]
        p_q[7:] = d2p @ q[7:]
        T_world_to_body = self.plant.CalcRelativeTransform(self.context,
                                                           self.body_frame,
                                                           self.plant.world_frame())
        T_body = T_world_to_body.GetAsMatrix4()
        R_pw = T_body[0:3, 0:3]

        ad_T = np.zeros((6, 6))
        ad_T[0:3, 0:3] = R_pw
        ad_T[3:6, 3:6] = R_pw

        v_raw = ad_T @ v_raw_[0:6]

        p_v = np.zeros(18)
        p_v[0] = v_raw[3]
        p_v[1] = v_raw[4]
        p_v[2] = v_raw[5]
        p_v[3] = v_raw[0]
        p_v[4] = v_raw[1]
        p_v[5] = v_raw[2]
        p_v[6:] = d2p @ v_raw_[6:]

        computeCentroidalMapTimeVariation(model=self.p_model, data=p_data, q=p_q,
                                          v=p_v)
        hg_pin = p_data.Ag @ p_v
        # print(f"p_data.Ag{hg_pin}")
        # print(p_data.hg)

        # compare the centroidal momentums between pinocchio and drake
        hg_drake = self.plant.CalcSpatialMomentumInWorldAboutPoint(
            context=self.context,
            p_WoP_W=p_com)

        # print(f"hg_drake.translational(){hg_drake.translational()}")
        # print(f"hg_drake.rotational(){hg_drake.rotational()}")
        Ag_ang = p_data.Ag[3:, :]
        Agd_ang = p_data.dAg[3:, :]
        Agd_ang_v = Agd_ang @ p_v
        hg_ang = hg_pin[3:]

        time_start_takeoff = 4.0
        if context.get_time() <= time_start_takeoff:
            hgd_des = 0. - Kd_hg * (hg_ang - [0., 0., 0.])
        else:
            hgd_des = 0. - Kd_hg * (hg_ang - [-0., 0., 0.])
        # print(f"hg_ang{hg_ang}")
        # print(p_data.Ag)
        # print(f"dAg{p_data.dAg}")

        # robot = RobotWrapper.BuildFromURDF(URDF)
        pin_com = np.array(p_data.com[0])
        ######################### print names of joints #######################
        # iii = 0
        # while iii < 12:
        #     joint_ii = self.plant.get_joint(JointIndex(iii))
        #     print(f"joint{iii}is:  {joint_ii.name()}")
        #     iii = iii + 1
        #######################################################################
        # Set up the QP
        self.mp = MathematicalProgram()

        vd = self.mp.NewContinuousVariables(self.plant.num_velocities(), 1,
                                            'vd')
        tau = self.mp.NewContinuousVariables(self.plant.num_actuators(), 1,
                                             'tau')
        f_c = [self.mp.NewContinuousVariables(6, 1, 'f_%s' % j) for j in
               range(num_contact)]

        # min || J_com*vd + Jd_com*v - pdd_com_des ||^2
        self.AddJacobianTypeCost(J_com, vd, Jdv_com, pdd_com_des,
                                 weight=w_com)

        # min | Ag*vd + Agd*v - hgd_des ||^2
        self.AddJacobianTypeCost(Ag_ang, vd, Agd_ang_v, hgd_des,
                                 weight=w_hg)
        # # min || J_body*vd + Jd_body*v - pdd_body_des ||^2
        # self.AddJacobianTypeCost(J_body, vd, Jdv_body, vd_body_des,
        #                          weight=w_body)

        # min || J_body*vd + Jd_body*v - pdd_body_des ||^2
        J_body_ori = J_body[:3]
        Jdv_body_ori = Jdv_body[:3]
        vd_body_des_ori = vd_body_des[:3]
        self.AddJacobianTypeCost(J_body_ori, vd, Jdv_body_ori, vd_body_des_ori,
                                 weight=w_body)

        # min || J_s*vd+ Jd_s*v - pdd_s_des ||^2
        for i in range(num_swing):
            self.AddJacobianTypeCost(J_s[i], vd, Jdv_s[i], vd_foot_des[i],
                                     weight=w_s_foot)

        # TODO track angular momentum, pinocchio lib can generate Ag and Ag_dot (finished)

        # TODO track a complete jumping trajectory

        # s.t.  M*vd + Cv + tau_g = S'*tau + sum(J_c[j]'*f_c[j])
        self.AddDynamicsConstraint(M, vd, Cv, tau_g, S, tau, J_c, f_c)

        # rotation matrixs of feet
        Rot_foot_left_mtx, Rot_foot_right_mtx = self.CalcFeetRot()

        # min || J_cj*vd + Jd_cj*v == 0 ||^2 (+ some damping)
        for i in range(num_contact):

            # Kd_angular = kk_ * 0.2 * np.eye(3)
            Kd_angular = kk_ * 0.1 * np.eye(3)
            # Kd_linear = kk_ * 200 * np.eye(3)
            Kd_linear = kk_ * 30 * np.eye(3)
            Kd = np.block([[Kd_angular, np.zeros((3, 3))],
                           [np.zeros((3, 3)), Kd_linear]])
            num_contacts = len(J_c)
            for j in range(num_contacts):
                # j = 1
                pd = (J_c[j] @ v).reshape(6, 1)
                vd_c_foot_des = -Kd @ pd
                # print(f"{j}pd{pd}")
            self.AddJacobianTypeCost(J_c[i], vd, Jdv_c[i], vd_c_foot_des,
                                     weight=w_c_foot)

        # min || vd - kp*(q_arm_error) - kd*(qd_arm_error) ||^2
        arms_degree_setting = 20.
        q_arm_nom = np.array([arms_degree_setting * math.pi / 180, -arms_degree_setting * math.pi / 180])
        q_arm_cur = np.array(q[7:9])
        q_arm_error = q_arm_cur - q_arm_nom
        qd_arm_cur = np.array(v[6:8])
        qd_arm_error = qd_arm_cur - np.zeros(2)
        tau_arms = - (Kp_arm * q_arm_error) - (Kd_arm * qd_arm_error)
        Q_shoulder_pos = w_shoulder_pos * np.eye(2)
        c_shoulder_pos = -w_shoulder_pos * tau_arms
        shoulder_joints_pdd = np.array([vd[6], vd[7]])
        self.mp.AddQuadraticCost(Q_shoulder_pos, c_shoulder_pos, shoulder_joints_pdd)

        # min || tau_shoulders ||^2
        Q_shoulder_torque = w_shoulder_torque * np.eye(2)
        c_shoulder_torque = np.zeros([2, 1])
        vd_arms = np.array([vd[6], vd[7]])
        self.mp.AddQuadraticCost(Q_shoulder_torque, c_shoulder_torque, vd_arms)

        # s.t. f_c[j] in friction cones
        if any(contact_feet):
            self.AddFrictionPyramidConstraint(f_c, Rot_foot_left_mtx,
                                              Rot_foot_right_mtx, swing_feet)
            # print("ok")

        # s.t.  knee_left_joint > 0 knee_right_joint>0
        # lb_l_knee, ub_l_knee = self.CalcLbUbKneeL()  #
        # self.mp.AddBoundingBoxConstraint(lb_l_knee, ub_l_knee, vd[-6])
        # lb_r_knee, ub_r_knee = self.CalcLbUbKneeR()
        # self.mp.AddBoundingBoxConstraint(lb_r_knee, ub_r_knee, vd[-5])
        # self.mp.AddBoundingBoxConstraint(-200,200,tau)
        # self.mp.AddBoundingBoxConstraint(-1500,1500,f_c[0])
        # self.mp.AddBoundingBoxConstraint(-1500, 1500, f_c[1])

        # s.t.  shoulder_left_joint > 0 shoulder_right_joint>0
        lb_l_shoulder, ub_l_shoulder = self.CalcLbUbShoulderL()  #
        self.mp.AddBoundingBoxConstraint(lb_l_shoulder, ub_l_shoulder, vd[6])
        lb_r_shoulder, ub_r_shoulder = self.CalcLbUbShoulderR()
        self.mp.AddBoundingBoxConstraint(lb_r_shoulder, ub_r_shoulder, vd[7])

        result = self.solver.Solve(self.mp)
        assert result.is_success()
        tau = result.GetSolution(tau)

        # Set error for logging
        x_tilde = np.hstack([rpy_body - rpy_body_nom,
                             p_body - p_body_nom,
                             p_s.flatten() - p_s_nom.flatten()])
        self.err = x_tilde.T @ x_tilde
        self.res = result.get_solver_details().primal_res

        # q_arm_nom = np.array([-0*math.pi/180, 0*math.pi/180])
        # q_arm_cur = np.array(q[7:9])
        # q_arm_error = q_arm_cur - q_arm_nom
        # v_arms = np.array(v[6:8])
        # v_arm_error = v_arms - np.zeros(2)
        # tau[0:2] = - (Kp_arm * q_arm_error) - (Kd_arm * v_arm_error)
        # print(f"tau{tau}")

        # give the joint torque commands
        return tau

    def ControlLaw_PD(self, context, q, v):
        """
        This function is called by DoSetControlTorques, and consists of the main control
        code for the robot.
        """
        # Some dynamics computations
        M, C, tau_g, S = self.CalcDynamics()

        # Tuning parameters
        Kp = 1000.0 * np.eye(self.plant.num_velocities())
        time_landing = 4.8
        if context.get_time() <= time_landing:
            kp_ankle_set = 5.0
        else:
            kp_ankle_set = 5
        kp_ankle_set = 180.0
        Kp[6, 6] = kp_ankle_set
        Kp[7, 7] = kp_ankle_set
        Kp[14, 14] = kp_ankle_set
        Kp[15, 15] = kp_ankle_set
        Kp[16, 16] = kp_ankle_set
        Kp[17, 17] = kp_ankle_set

        Kd = 40 * np.eye(self.plant.num_velocities())

        kd_ankle_set = 0.0007*kp_ankle_set
        Kd[6, 6] = 40*kd_ankle_set
        Kd[7, 7] = 40*kd_ankle_set
        Kd[14, 14] = kd_ankle_set
        Kd[15, 15] = kd_ankle_set
        Kd[16, 16] = 0.001*kd_ankle_set
        Kd[17, 17] = 0.001*kd_ankle_set
        # Nominal joint angles
        # q_nom = np.asarray([0.0, 0.0, 0.0, 0.0,  # base orientation
        #                     0.0, 0.0, 0.0,  # base position
        #                     0.0, 0.0, 0.0, 0.0,  #
        #                     0.0, 0.0, 0.0, 0.0,  #
        #                     0.0, 0.0, 0.0, 0.0])  #
        q_nom = np.asarray(
            [9.84625570e-01, 8.47824762e-04, 1.74777949e-01, 2.29166405e-03,
             - 2.85259563e-02, - 6.34527837e-03, 7.07437164e-01,
             2.82725937e-02,
             - 5.62032345e-02, - 1.67686068e-02, - 7.90473043e-03,
             1.60650512e+00,
             - 1.60149489e+00, - 2.13936803e+00, 2.13786458e+00,
             - 8.88485533e-01,
             8.91950614e-01, 3.57152295e-02, - 3.57152295e-02])
        # shoulder joints setting
        arms_degree_setting = 20.
        q_nom[7] = arms_degree_setting*math.pi/180
        q_nom[8] = -arms_degree_setting*math.pi/180

        # q_nom[0:4] = np.array([0.996, 0., 0.087, 0.])
        # q_nom[11] = 80*math.pi/180
        # q_nom[12] = -80*math.pi/180
        # q_nom[15] = 0*math.pi/180
        # q_nom[16] = -0*math.pi/180
        # q_nom[17] = -0 * math.pi / 180
        # q_nom[18] = -0 * math.pi / 180

        # q_nom = np.asarray([1.0, 0.0, 0.0, 0.0,
        #                  0.0, 0.0, 1.044e+00,
        #                  -0.0e-01, 0.0e-01, -2.22e-03, 2.22e-03,
        #                  5.583e-01, -5.583e-01, -1.174e+00, 1.174e+00,
        #                  -6.24e-01, 6.24e-01, 2.25e-02, -2.25e-02])

        # Compute desired generalized forces
        q_err = self.plant.MapQDotToVelocity(self.context,
                                             q - q_nom)  # Need to use qd=N(q)*v here,
        # q_err_ = q - q_nom  # Need to use qd=N(q)*v here,
        # q_err = q_err_[1:]
        # since q and v have different sizes
        qd_err = v - np.zeros(self.plant.num_velocities())
        tau = - Kp @ q_err - Kd @ qd_err

        # Use actuation matrix to map generalized forces to control inputs
        u = S @ tau
        u = np.clip(u, -300, 300)

        return u

################## joint sequences and actuator sequences #####################

# joints states:
# joint0is:  shoulder_left
# joint1is:  shoulder_right
# joint2is:  hip_left_x
# joint3is:  hip_right_x
# joint4is:  hip_left_y
# joint5is:  hip_right_y
# joint6is:  knee_left
# joint7is:  knee_right
# joint8is:  ankle_left_y
# joint9is:  ankle_right_y
# joint10is:  ankle_left_x
# joint11is:  ankle_right_x

# actuators (torques):
# joint0is:  shoulder_left
# joint1is:  shoulder_right
# joint2is:  hip_left_x
# joint3is:  hip_left_y
# joint4is:  knee_left
# joint5is:  ankle_left_y
# joint6is:  ankle_left_x
# joint7is:  hip_right_x
# joint8is:  hip_right_y
# joint9is:  knee_right
# joint10is:  ankle_right_y
# joint11is:  ankle_right_x
