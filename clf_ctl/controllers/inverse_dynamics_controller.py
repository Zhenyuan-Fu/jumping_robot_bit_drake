import numpy as np
from pydrake.math import RollPitchYaw

from controllers.basic_controller import *


class IDController(BasicController):
    """
    A standard QP-based inverse dynamics controller.

    Takes as input desired positions/velocities/accelerations of the
    feet and floating base and computes corresponding joint torques.
    """

    def __init__(self, plant, dt, p_model, jump_running, use_lcm=False):
        BasicController.__init__(self, plant, dt, p_model, jump_running, use_lcm=use_lcm)

        # inputs from the trunk model are sent in a dictionary
        self.DeclareAbstractInputPort(
            "trunk_input",
            AbstractValue.Make({}))

        # Set the friction coefficient
        self.mu = 0.7
        self.x = 0.220 / 2.0
        self.y = 0.107 / 2.0

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
        # print(f"A_eq{A_eq}\n")
        # print(f"xdd_des{xdd_des}\n")
        # print(f"Jd_qd{Jd_qd}\n")
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
        to the whole-body controller QP.
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
        # print(f"Rot_foot_left{Rot_foot_left}")
        # print(f"Rot_foot_right{Rot_foot_right}")
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
               w_arm* || tau_arm ||^2
           subject to:
                M*vd + Cv + tau_g = S'*tau + sum(J'*f)
                f \in friction cones
                J_cj*vd + Jd_cj*v == 0
        """

        ######### Tuning Parameters #########
        # Kp_body_p = 80.0
        # Kd_body_p = 10.5
        Kp_body_p = 200.0
        Kd_body_p = 20.0

        # Kp_body_rpy = 0.2*Kp_body_p
        # Kd_body_rpy = 0.04*Kd_body_p
        Kp_body_rpy = 0.4 * Kp_body_p
        Kd_body_rpy = 0.4 * Kd_body_p

        Kp_foot = 200.0
        Kd_foot = 20.

        Kp_foot_rpy = Kp_foot
        Kd_foot_rpy = Kd_foot

        # w_body = 1.0
        # w_s_foot = 15.0
        w_body = 10.0
        w_s_foot = 5.0
        w_c_foot = 400.0
        w_tau = 100
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

        p_body_nom = trunk_data["p_body"]
        pd_body_nom = trunk_data["pd_body"]
        pdd_body_nom = trunk_data["pdd_body"]

        rpy_body_nom = trunk_data["rpy_body"]
        rpyd_body_nom = trunk_data["rpyd_body"]
        rpydd_body_nom = trunk_data["rpydd_body"]

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

        # Get robot's actual task-space (body pose + foot pose) data
        X_body, J_body, Jdv_body = self.CalcFramePoseQuantities(
            self.body_frame)

        p_body = X_body.translation()
        pd_body = (J_body @ v)[3:]

        RPY_body = RollPitchYaw(
            X_body.rotation())  # RPY object helps convert between angular velocity and rpyd
        rpy_body = RPY_body.vector()
        omega_body = (J_body @ v)[:3]  # angular velocity of the body 不用改
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

        # swing foot actual ori and pos
        p_feet = np.array([p_lf, p_rf]).reshape(2, 3)
        pd_feet = np.array([pd_lf, pd_rf]).reshape(2, 3)
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

        # Set up the QP
        self.mp = MathematicalProgram()

        vd = self.mp.NewContinuousVariables(self.plant.num_velocities(), 1,
                                            'vd')
        tau = self.mp.NewContinuousVariables(self.plant.num_actuators(), 1,
                                             'tau')
        f_c = [self.mp.NewContinuousVariables(6, 1, 'f_%s' % j) for j in
               range(num_contact)]

        # min || J_body*vd + Jd_body*v - pdd_body_des ||^2
        self.AddJacobianTypeCost(J_body, vd, Jdv_body, vd_body_des,
                                 weight=w_body)

        # min || J_s*vd+ Jd_s*v - pdd_s_des ||^2
        for i in range(num_swing):
            self.AddJacobianTypeCost(J_s[i], vd, Jdv_s[i], vd_foot_des[i],
                                     weight=w_s_foot)

        # TODO CalcJacobianCenterOfMassTranslationalVelocity 跟随质心 9_4-9_6

        # TODO 平衡角动量 9_7-9_9

        # TODO 找一个跳跃轨迹并完成跳跃 9_2-9_3 9_7-9_10

        # s.t.  M*vd + Cv + tau_g = S'*tau + sum(J_c[j]'*f_c[j])
        self.AddDynamicsConstraint(M, vd, Cv, tau_g, S, tau, J_c, f_c)

        # rotation matrix of foot
        Rot_foot_left_mtx, Rot_foot_right_mtx = self.CalcFeetRot()
        # min || J_cj*vd + Jd_cj*v == 0 ||^2 (+ some daming)
        for i in range(num_contact):
            kk_ = 1
            # Kd_angular = kk_ * 0.2 * np.eye(3)
            Kd_angular = kk_ * 0.1 * np.eye(3)
            # Kd_linear = kk_ * 200 * np.eye(3)
            Kd_linear = kk_ * 20 * np.eye(3)
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

        # Q_arm = w_tau * np.eye(11)
        # Q_arm[0, 0] = w_tau * 2
        # Q_arm[1, 1] = w_tau * 2
        # c_arm = np.zeros([11, 1])
        # vd_arms = np.array([tau[0],tau[1],tau[2],tau[3],tau[4],tau[5],
        #                     tau[6],tau[7],tau[8],tau[9],tau[10]])
        # self.mp.AddQuadraticCost(Q_arm, c_arm, vd_arms)

        Q_arm = w_tau * np.eye(2)
        c_arm = np.zeros([2, 1])
        vd_arms = np.array([tau[0], tau[1]])
        self.mp.AddQuadraticCost(Q_arm, c_arm, vd_arms)

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

        result = self.solver.Solve(self.mp)
        assert result.is_success()
        tau = result.GetSolution(tau)

        # Set error for logging
        x_tilde = np.hstack([rpy_body - rpy_body_nom,
                             p_body - p_body_nom,
                             p_s.flatten() - p_s_nom.flatten()])
        self.err = x_tilde.T @ x_tilde
        self.res = result.get_solver_details().primal_res

        return tau
