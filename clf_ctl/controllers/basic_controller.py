import math

import numpy as np
from pydrake.all import *

import lcm
from lcm_types.jumping_robot_lcm import jumping_robot_state_control_lcmt
from helpers import jacobian2


class BasicController(LeafSystem):
    """
    A simple PD controller for a quadruped robot.

                  -----------------------
                  |                     |
      [q,qd] ---> |   BasicController   | ---> tau
                  |                     |
                  -----------------------

    Includes some basic dynamics computations, so other
    more complex controllers can inherit from this class.
    """

    def __init__(self, plant, dt,  p_model, jump_running, use_lcm=False):
        LeafSystem.__init__(self)

        self.dt = dt
        self.plant = plant
        self.context = self.plant.CreateDefaultContext()  # stores q,qd
        self.p_model = p_model
        self.jump_running = jump_running

        # AutoDiff plant and context for values that require automatic differentiation
        self.plant_autodiff = plant.ToAutoDiffXd()
        self.context_autodiff = self.plant_autodiff.CreateDefaultContext()

        # Declare input and output ports
        self.DeclareVectorInputPort(
            "biped_state",
            BasicVector(
                self.plant.num_positions() + self.plant.num_velocities()))

        self.DeclareVectorOutputPort(
            "biped_torques",
            BasicVector(self.plant.num_actuators()),
            self.DoSetControlTorques)

        # Declare output port for logging
        self.V = 0
        self.err = 0
        self.res = 0
        self.Vdot = 0
        self.DeclareVectorOutputPort(
            "output_metrics",
            BasicVector(4),
            self.SetLoggingOutputs)

        # Handle whether or not we're communicating with a real robot and/or simulator
        # over LCM.
        self.use_lcm = use_lcm
        if self.use_lcm:
            self.lc = lcm.LCM()

            # LCM subscriber gives us estimates of q, qd, tau
            self.q = np.zeros(self.plant.num_positions())
            self.v = np.zeros(self.plant.num_velocities())
            subscription = self.lc.subscribe("robot_current_state",
                                             self.lcm_callback)

        # Relevant frames for the CoM and each foot
        self.world_frame = self.plant.world_frame()
        self.body_frame = self.plant.GetFrameByName(
            "trunk")  # "body" for bit_jumping_robot

        self.lfoot_frame = self.plant.GetFrameByName("foot_left")  # left foot
        self.rfoot_frame = self.plant.GetFrameByName(
            "foot_right")  # right foot

        self.world_frame_autodiff = self.plant_autodiff.world_frame()
        self.body_frame_autodiff = self.plant_autodiff.GetFrameByName("trunk")
        self.lfoot_frame_autodiff = self.plant_autodiff.GetFrameByName(
            "foot_left")
        self.rfoot_frame_autodiff = self.plant_autodiff.GetFrameByName(
            "foot_right")

    def lcm_callback(self, channel, data):
        """
        Handle data coming over LCM from another simulator or a real robot.

        Sets self.q, self.qd according to latest estimates of robot state.
        """
        msg = jumping_robot_state_control_lcmt.decode(data)
        self.q = np.asarray(msg.q)
        self.v = np.asarray(msg.v)

    def UpdateStoredContext(self, context):
        """
        Use the data in the given input context to update self.context.
        This should be called at the beginning of each timestep.
        """
        state = self.EvalVectorInput(context, 0).get_value()
        q = state[:self.plant.num_positions()]
        v = state[-self.plant.num_velocities():]

        self.plant.SetPositions(self.context, q)
        self.plant.SetVelocities(self.context, v)

    def CalcDynamics(self):
        """
        Compute dynamics quantities, M, Cv, tau_g, and S such that the
        robot's dynamics are given by

            M(q)vd + C(q,v)v + tau_g = S'u + tau_ext.

        Assumes that self.context has been set properly.
        """
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.context)
        Cv = self.plant.CalcBiasTerm(self.context)
        tau_g = -self.plant.CalcGravityGeneralizedForces(self.context)
        S = self.plant.MakeActuationMatrix().T

        return M, Cv, tau_g, S

    def CalcCoriolisMatrix(self):
        """
        Compute the coriolis matrix C(q,qd) using autodiff.

        Assumes that self.context has been set properly.
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        def Cv_fcn(v):
            self.plant_autodiff.SetPositions(self.context_autodiff, q)
            self.plant_autodiff.SetVelocities(self.context_autodiff, v)
            return self.plant_autodiff.CalcBiasTerm(self.context_autodiff)

        C = 0.5 * jacobian(Cv_fcn, v)
        return C

    def CalcComQuantities(self):
        """
        Compute the position (p), jacobian (J) and
        jacobian-time-derivative-times-v (Jdv) for the center-of-mass.

        Assumes that self.context has been set properly.
        """
        p = self.plant.CalcCenterOfMassPositionInWorld(self.context)
        J = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
            self.context,
            JacobianWrtVariable.kV,
            self.world_frame,
            self.world_frame)
        Jdv = self.plant.CalcBiasCenterOfMassTranslationalAcceleration(
            self.context,
            JacobianWrtVariable.kV,
            self.world_frame,
            self.world_frame)
        return p, J, Jdv

    def CalcComJacobianDot(self):
        """
        Compute the time derivative of the center-of-mass Jacobian (Jd)
        directly using autodiff.

        Assumes that self.context has been set properly.
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        def J_fcn(q):
            self.plant_autodiff.SetPositions(self.context_autodiff, q)
            self.plant_autodiff.SetVelocities(self.context_autodiff, v)
            return self.plant_autodiff.CalcJacobianCenterOfMassTranslationalVelocity(
                self.context_autodiff,
                JacobianWrtVariable.kV,
                self.world_frame_autodiff,
                self.world_frame_autodiff)

        Jd = jacobian(J_fcn, q) @ self.plant.MapVelocityToQDot(self.context, v)

        return Jd

    def CalcFramePositionQuantities(self, frame):
        """
        Compute the position (p), jacobian (J) and
        jacobian-time-derivative-times-v (Jdv) for the given frame

        Assumes that self.context has been set properly.
        """
        p = self.plant.CalcPointsPositions(self.context,
                                           frame,
                                           np.array([0, 0, 0]),
                                           self.world_frame)
        J = self.plant.CalcJacobianSpatialVelocity(self.context,
                                                   JacobianWrtVariable.kV,
                                                   frame,
                                                   np.array([0, 0, 0]),
                                                   self.world_frame,
                                                   self.world_frame)
        Jdv_spatial = self.plant.CalcBiasSpatialAcceleration(self.context,
                                                     JacobianWrtVariable.kV,
                                                     frame,
                                                     np.array(
                                                         [0.0, 0, 0.0]),
                                                     self.world_frame,
                                                     self.world_frame)

        Jdv_ = np.hstack((Jdv_spatial.rotational(), Jdv_spatial.translational()))
        Jdv = np.array([[Jdv_[0]],
                        [Jdv_[1]],
                        [Jdv_[2]],
                        [Jdv_[3]],
                        [Jdv_[4]],
                        [Jdv_[5]]])
        return p, J, Jdv

    def CalcFeetRot(self):
        """
        Compute the rotation mtx for feet
        """
        foot_left = self.plant.GetFrameByName("foot_left")
        foot_right = self.plant.GetFrameByName("foot_right")
        world_frame = self.plant.world_frame()

        X_W_foot_left = self.plant.CalcRelativeTransform(
            context=self.context, frame_A=world_frame, frame_B=foot_left)
        X_W_foot_right = self.plant.CalcRelativeTransform(
            context=self.context, frame_A=world_frame, frame_B=foot_right)
        Rot_foot_left_mtx = X_W_foot_left.rotation().matrix()
        Rot_foot_right_mtx = X_W_foot_right.rotation().matrix()
        return Rot_foot_left_mtx, Rot_foot_right_mtx

    def CalcLbUbKneeL(self):
        """
        Compute the rotation mtx for feet
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        q_min = -120*math.pi/180
        q_max = 120*math.pi/180
        dt = self.dt
        # print(f"q{q}")
        lb_l_knee = (q_min-q[-6]-v[-6]*self.dt)/(0.5 * dt * dt)
        ub_l_knee = (q_max - q[-6] - v[-6] * self.dt) / (0.5 * dt * dt)
        return lb_l_knee, ub_l_knee

    def CalcLbUbKneeR(self):
        """
        Compute the rotation mtx for feet
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        q_min = -120*math.pi/180
        q_max = 120*math.pi/180
        dt = self.dt
        lb_r_knee = (q_min - q[-5]-v[-5]*self.dt)/(0.5 * dt * dt)
        ub_r_knee = (q_max - q[-5] - v[-5] * self.dt) / (0.5 * dt * dt)
        return lb_r_knee, ub_r_knee

    def CalcLbUbShoulderL(self):
        """
        Compute the rotation mtx for feet
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        q_min = -130*math.pi/180
        q_max = 130*math.pi/180
        dt = self.dt
        # print(f"q{q}")
        lb_l_shoulder = (q_min-q[7]-v[6]*self.dt)/(0.5 * dt * dt)
        ub_l_shoulder = (q_max - q[7] - v[6] * self.dt) / (0.5 * dt * dt)
        return lb_l_shoulder, ub_l_shoulder

    def CalcLbUbShoulderR(self):
        """
        Compute the rotation mtx for feet
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        q_min = -130*math.pi/180
        q_max = 130*math.pi/180
        dt = self.dt
        lb_r_shoulder = (q_min - q[8]-v[7]*self.dt)/(0.5 * dt * dt)
        ub_r_shoulder = (q_max - q[8] - v[7] * self.dt) / (0.5 * dt * dt)
        return lb_r_shoulder, ub_r_shoulder

    def CalcFrameJacobianDot(self, frame):
        """
        Compute the time derivative of the given frame's position Jacobian (Jd)
        directly using autodiff.

        Note that `frame` must be an autodiff type frame.

        Assumes that self.context has been set properly.
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        def J_fcn(q):
            self.plant_autodiff.SetPositions(self.context_autodiff, q)
            self.plant_autodiff.SetVelocities(self.context_autodiff, v)
            return self.plant_autodiff.CalcJacobianTranslationalVelocity(
                self.context_autodiff,
                JacobianWrtVariable.kV,
                frame,
                np.zeros(3, ),
                self.world_frame_autodiff,
                self.world_frame_autodiff)

        Jd = jacobian2(J_fcn, q) @ self.plant.MapVelocityToQDot(self.context,
                                                                v)
        return Jd

    def CalcFramePoseJacobianDot(self, frame, position):
        """
        Compute the time derivative of the given frame's pose Jacobian (Jd)
        directly using autodiff.

        Note that `frame` must be an autodiff type frame.

        Assumes that self.context has been set properly.
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        def J_fcn(q):
            self.plant_autodiff.SetPositions(self.context_autodiff, q)
            self.plant_autodiff.SetVelocities(self.context_autodiff, v)
            return self.plant_autodiff.CalcJacobianSpatialVelocity(
                self.context_autodiff,
                JacobianWrtVariable.kV,
                frame,
                position,
                self.world_frame_autodiff,
                self.world_frame_autodiff)

        Jd = jacobian2(J_fcn, q) @ self.plant.MapVelocityToQDot(self.context,
                                                                v)
        return Jd

    def CalcFramePoseQuantities(self, frame):
        """
        Compute the pose (position + orientation), spatial jacobian (J) and,
        spatial jacobian-time-derivative-times-v (Jdv) for the given frame.

        Assumes that self.context has been set properly.
        """
        pose = self.plant.CalcRelativeTransform(self.context,
                                                self.world_frame,
                                                frame)
        J = self.plant.CalcJacobianSpatialVelocity(self.context,
                                                   JacobianWrtVariable.kV,
                                                   frame,
                                                   np.array([0, 0, 0]),
                                                   self.world_frame,
                                                   self.world_frame)
        Jdv = self.plant.CalcBiasSpatialAcceleration(self.context,
                                                     JacobianWrtVariable.kV,
                                                     frame,
                                                     np.array([0, 0, 0]),
                                                     self.world_frame,
                                                     self.world_frame)

        return pose, J, Jdv.get_coeffs()

    # def CalcWorldFrameCoMQuantities(self, frame, com_position):
    #     """
    #     Compute the position (p), jacobian (J) and
    #     jacobian-time-derivative-times-v (Jdv) for the given frame
    #
    #     Assumes that self.context has been set properly.
    #     """
    #     # p = self.plant.CalcPointsPositions(self.context,
    #     #                                    frame,
    #     #                                    np.array([0, 0, 0]),
    #     #                                    self.world_frame)
    #     J = self.plant.CalcJacobianTranslationalVelocity(self.context,
    #                                                      JacobianWrtVariable.kV,
    #                                                      frame,
    #                                                      com_position,
    #                                                      self.world_frame,
    #                                                      self.world_frame)
    #     Jdv = self.plant.CalcBiasTranslationalAcceleration(self.context,
    #                                                        JacobianWrtVariable.kV,
    #                                                        frame,
    #                                                        com_position,
    #                                                        self.world_frame,
    #                                                        self.world_frame)
    #     # print("1Jdv_c[j]", J)
    #     # print("2Jdv_c", Jdv)
    #     return J, Jdv

    def SetLoggingOutputs(self, context, output):
        """
        Set outputs for logging, namely a vector consisting of
        the current simulation function

            V = 1/2 qd_tilde'*M*qd_tilde + pd_tilde'*Kp*pd_tilde

        and the current output error

            pd_tilde'*pd_tilde.

        """
        output.SetFromVector(
            np.asarray([self.V, self.err, self.res, self.Vdot]))

    def DoSetControlTorques(self, context, output):
        """
        This function gets called at every timestep and sends output torques to
        the simulator (and/or over LCM, if needed).
        """
        if self.use_lcm:
            # Get robot's current state (q,v) from LCM
            self.lc.handle()
            q = self.q
            v = self.v
            self.plant.SetPositions(self.context, self.q)
            self.plant.SetVelocities(self.context, self.v)
        else:
            # Get robot's current state (q,v) from Drake
            self.UpdateStoredContext(context)
            q = self.plant.GetPositions(self.context)
            v = self.plant.GetVelocities(self.context)

        # Compute controls to apply
        if self.jump_running:
            time_takeoff = 4.4
            if context.get_time() <= time_takeoff:
                u = self.ControlLaw(context, q, v)
            else:
                u = self.ControlLaw_PD(context, q, v)
        else:
            u = self.ControlLaw(context, q, v)


        if self.use_lcm:
            # Send control outputs over LCM
            msg = jumping_robot_state_control_lcmt()
            S = self.plant.MakeActuationMatrix().T
            msg.tau = (S.T @ u)[
                      -self.plant.num_actuators():]  # The mini cheetah controller assumes
            # control torques are in the same order as
            # v, but drake uses a different (custom) mapping.
            self.lc.publish("robot_control_input", msg.encode())

            # We'll just send zero control inputs to the drake simulator
            output.SetFromVector(np.zeros(self.plant.num_actuators()))
        else:
            # Send control outputs to drake
            output.SetFromVector(u)

    def ControlLaw(self, context, q, v):
        """
        This function is called by DoSetControlTorques, and consists of the main control
        code for the robot.
        """
        # Some dynamics computations
        M, C, tau_g, S = self.CalcDynamics()

        # Tuning parameters
        Kp = 300.0 * np.eye(self.plant.num_velocities())
        Kd = 0.35 * np.eye(self.plant.num_velocities())

        # Nominal joint angles
        q_nom = np.asarray([0.0, 0.0, 0.0, 0.0,  # base orientation
                            0.0, 0.0, 0.0,  # base position
                            0.0, 0.0, 0.0, 0.0,  #
                            0.0, 0.0, 0.0, 0.0,  #
                            0.0, 0.0, 0.0, 0.0])  #

        # Compute desired generalized forces
        q_err = self.plant.MapQDotToVelocity(self.context,
                                             q - q_nom)  # Need to use qd=N(q)*v here,
        # since q and v have different sizes
        qd_err = v - np.zeros(self.plant.num_velocities())
        tau = - Kp @ q_err - Kd @ qd_err

        # Use actuation matrix to map generalized forces to control inputs
        u = S @ tau
        u = np.clip(u, -300, 300)

        return u
