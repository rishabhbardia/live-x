import numpy as np
import pyceres
import cv2

# ============================================================================
# Pinhole projection and analytical Jacobians — no distortion
# ============================================================================
#
# Parameters:
#   Xw    : world point                         (3,)
#   Cw    : camera center in world coordinates  (3,)
#   omega : angle-axis rotation vector          (3,)
#   f     : shared horizontal/vertical focal length, in pixels
#   cx, cy: principal point, in pixels
#
# Convention:
#   R = Rodrigues(omega) maps world directions to camera directions.
#
#   Xcam = R @ Xw + t
#   t    = -R @ Cw
#   Xcam = R @ (Xw - Cw)
#
# Define:
#   d = Xw - Cw
#   Xcam = [X, Y, Z]
#
# Prediction:
#   pred = [u, v]
#   u = f * X / Z + cx
#   v = f * Y / Z + cy
#
# All derivatives below assume Z > EPS.
# Each partial derivative holds the other independent parameters fixed.
# Jacobian rows correspond to outputs; columns correspond to parameters.
#
#
# ----------------------------------------------------------------------------
# 1. Jpred = d(pred) / d(Xcam)                                  shape: (2, 3)
# ----------------------------------------------------------------------------
#
# Differentiate u = f * X / Z + cx:
#
#   du/dX = f / Z
#   du/dY = 0
#   du/dZ = f * X * d(Z^-1)/dZ = -f * X / Z^2
#
# Differentiate v = f * Y / Z + cy:
#
#   dv/dX = 0
#   dv/dY = f / Z
#   dv/dZ = f * Y * d(Z^-1)/dZ = -f * Y / Z^2
#
# Therefore:
#
#   Jpred = [[f/Z,   0, -f*X/Z^2],
#            [  0, f/Z, -f*Y/Z^2]]
#
#
# ----------------------------------------------------------------------------
# 2. Jintr = d(pred) / d([f, cx, cy])                           shape: (2, 3)
# ----------------------------------------------------------------------------
#
# Hold Xcam fixed.
#
#   du/df  = X / Z       dv/df  = Y / Z
#   du/dcx = 1           dv/dcx = 0
#   du/dcy = 0           dv/dcy = 1
#
# Therefore:
#
#   Jintr = [[X/Z, 1, 0],
#            [Y/Z, 0, 1]]
#
#
# ----------------------------------------------------------------------------
# 3. J_Cw = d(pred) / d(Cw)                                    shape: (2, 3)
# ----------------------------------------------------------------------------
#
# Hold Xw and omega fixed.
#
#   Xcam = R @ (Xw - Cw)
#   d(Xw - Cw)/dCw = -I
#   dXcam/dCw = R @ (-I) = -R
#
# By the chain rule:
#
#   J_Cw = d(pred)/dXcam @ dXcam/dCw
#        = Jpred @ (-R)
#
# Component form, for world coordinate j:
#
#   du/dCw[j] = -(f/Z) * R[0,j] + (f*X/Z^2) * R[2,j]
#   dv/dCw[j] = -(f/Z) * R[1,j] + (f*Y/Z^2) * R[2,j]
#
#
# ----------------------------------------------------------------------------
# 4. J_Xw = d(pred) / d(Xw)                                    shape: (2, 3)
# ----------------------------------------------------------------------------
#
# Hold Cw and omega fixed.
#
#   Xcam = R @ (Xw - Cw)
#   d(Xw - Cw)/dXw = I
#   dXcam/dXw = R
#
# By the chain rule:
#
#   J_Xw = d(pred)/dXcam @ dXcam/dXw
#        = Jpred @ R
#
# Component form, for world coordinate j:
#
#   du/dXw[j] = (f/Z) * R[0,j] - (f*X/Z^2) * R[2,j]
#   dv/dXw[j] = (f/Z) * R[1,j] - (f*Y/Z^2) * R[2,j]
#
# Hence:
#
#   J_Cw = -J_Xw
#
# Translating the point and camera center by the same world displacement
# leaves their difference, and therefore the prediction, unchanged.
#
#
# ----------------------------------------------------------------------------
# 5. J_omega = d(pred) / d(omega)                              shape: (2, 3)
# ----------------------------------------------------------------------------
#
# Hold Xw and Cw fixed, so d = Xw - Cw is constant.
#
#   Xcam = R(omega) @ d
#
# For rotation-vector component omega[m]:
#
#   dXcam/domega[m] = (dR/domega[m]) @ d
#
# Construct the intermediate (3, 3) Jacobian column by column:
#
#   J_Xcam_omega[:, m] = (dR/domega[m]) @ d
#
# Then apply the chain rule:
#
#   J_omega = Jpred @ J_Xcam_omega
#
# Component form:
#
#   du/domega[m] = (f/Z) * J_Xcam_omega[0,m]
#                 - (f*X/Z^2) * J_Xcam_omega[2,m]
#
#   dv/domega[m] = (f/Z) * J_Xcam_omega[1,m]
#                 - (f*Y/Z^2) * J_Xcam_omega[2,m]
#
#
# OpenCV supplies the analytical derivative of R:
#
#   R, D = cv2.Rodrigues(omega)
#
# For angle-axis INPUT:
#   R.shape == (3, 3)
#   D.shape == (3, 9)
#
# Each row of D contains one flattened matrix derivative:
#
#   D[m].reshape(3, 3) = dR/domega[m]
#
# Equivalently:
#
#   D[m, 3*k + j] = d(R[k,j])/d(omega[m])
#
# Transpose and reshape to put the parameter axis last:
#
#   A = D.T.reshape(3, 3, 3)
#   A[k,j,m] = d(R[k,j])/d(omega[m])
#
# Here:
#   k = output camera-coordinate index
#   j = input world-coordinate index
#   m = rotation-vector parameter index
#
# Since:
#
#   Xcam[k] = sum_j R[k,j] * d[j]
#
# Differentiation gives:
#
#   d(Xcam[k])/d(omega[m]) = sum_j A[k,j,m] * d[j]
#
# This contraction is:
#
#   J_Xcam_omega = np.einsum("kjm,j->km", A, d)
#
# The j axis is summed out; k and m remain.
#
# Full calculation:
#
#   R, D = cv2.Rodrigues(omega)
#   A = D.T.reshape(3, 3, 3)
#   d = Xw - Cw
#   J_omega = Jpred @ np.einsum("kjm,j->km", A, d)
#
# Equivalent calculation without einsum:
#
#   J_Xcam_omega = np.column_stack([
#       D[m].reshape(3, 3) @ d
#       for m in range(3)
#   ])
#   J_omega = Jpred @ J_Xcam_omega
#
# These derivatives correspond to additive updates of the stored angle-axis
# vector: omega_new = omega + delta_omega.
# A local rotation update, such as R_new = Exp(delta_omega) @ R, uses a
# different rotation Jacobian.
#
# Cw is fixed in this derivation, so t = -R @ Cw changes with omega.
# If using independent parameters (omega, t) and holding t fixed instead,
# dXcam/domega[m] would be (dR/domega[m]) @ Xw.
#
#
# ----------------------------------------------------------------------------
# 6. Prediction Jacobians versus residual Jacobians
# ----------------------------------------------------------------------------
#
# For a fixed observed pixel obs:
#
#   residual = pred - obs  -> use the Jacobians above.
#   residual = obs - pred  -> negate the Jacobians above.
#
# If residuals are divided by a fixed scalar standard deviation sigma_px,
# divide their Jacobians by sigma_px as well.
#
# For parameter order [f, cx, cy, Cw(3), Xw(3), omega(3)]:
#
#   J = np.hstack([Jintr, J_Cw, J_Xw, J_omega])     # shape: (2, 12)
#
# ============================================================================

def _project_jac(Xw, intr, C, om):
    """
        Everything that a cost function needs at one world point.

        One projection `X_cam = R (X_world - C_world)` then `u = f * X/Z + cx` and `v = f * Y/Z + cy`
        and every analystica derivative. `R` maps WORLD into CAMERA.

        Parameters
        ----------

        Xw : float(3,)
             The world point being projected in meters in the solved world frame. Built
             by the calling cost from parameter blocks it owns.
        intr : float(3,)
               `[f, cx, cy]` in pixels. Assembled by the called from its 2 ceres
               blocks -- `focal(1)` and `pp(2)` which are separate so we can do the zoom lock
               and keep free f per-frame while principal point stays shared. Square pixels, no skew and no distortion.
        C : float (3,)
            The camera centre in world meters, the `center` block. Subtracted from
            `Xw` before rotating, so it is a centre, not a translation `t`.
        om : float (3,)
            The frame's rotation as an axis-angle (Rodrigues) vector, radians --
            the `omega` block, and the form every file on disk stores. Reshaped to
            (3, 1) internally, so any shape with 3 elements is accepted.
            `cv2.Rodrigues` is asked for `dR` alongside `R` here, which is the whole
            reason this function exists rather than calling the shared `project`.  
    """
    f, cx, cy = intr
    om = np.asarray(om, np.float64).reshape(3, 1)
    R, dR = cv2.Rodrigues(om)                 # dR is (3,9); its transpose is dvec(R)/dom
    d = np.asarray(Xw, np.float64) - np.asarray(C, np.float64)
    x, y, z = R @ d
    zi = 1.0 / (z if abs(z) > 1e-9 else 1e-9)

    J_Xcam_omega = np.empty((3, 3), dtype=np.float64)
    pred = np.array([f * x * zi + cx, f * y * zi + cy])
    J_pred = np.array([[f * zi, 0.0, -f * x * zi * zi],
                   [0.0, f * zi, -f * y * zi * zi]])          # d(pred)/d(X_cam)
    J_intr = np.array([[x * zi, 1.0, 0.0],
                       [y * zi, 0.0, 1.0]])
    J_Cw = J_pred @ (-R)

    for m in range(3):
        dR_domega_m = dR[m].reshape(3, 3)
        J_Xcam_omega[:, m] = dR_domega_m @ d

    J_omega = J_pred @ J_Xcam_omega

    return pred, J_intr, J_Cw, J_omega, J_pred @ R, z

class PillarCost(pyceres.CostFunction):
    """
        One pillar end point touches : 
            Blocks : [focal(1), pp(2), C(3), omega(3), theta(1), delta_radius(1), delta_height(1)]
        Returns False when the point falls at or behind the camera. Ceres reads that
        as "this step is invalid", shrinks the trust region and backs off, which is the only thing keeping
        the solver out of mirrored configuration. A point behind the camera projects to a finite pixel
        , so its residual looks reasonable and nothing else in the cost would object.

    """

    def __init__(self, obs_uv: np.ndarray, is_top: bool, sigma: float, radius: float, pillar_height: float):
        super().__init__()
        self.obs = np.asarray(obs_uv, np.float64)
        self.is_top = bool(is_top)
        self.sigma = float(sigma) # need to scale the Jacobian remember from the NLLS series
        self.radius = float(radius)
        self.pillar_height = float(pillar_height)
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([1, 2, 3, 3, 1, 1, 1])


    def Evaluate(self, parameters, residuals, jacobians):
        foc, pp, C, om, theta, dR, dH = (np.asarray(p, np.float64) for p in parameters)
        intr = np.array([foc[0], pp[0], pp[1]])
        th, dr, dh = float(theta[0]), float(dR[0]), float(dH[0])
        r = self.radius + dr
        ct, st = np.cos(th), np.sin(th)

        Xw = np.array(
            [r * ct, r * st,
            self.pillar_height + dh if self.is_top else 0.0]
        )
        pred, J_intr, J_Cw, J_om, J_Xw, z = _project_jac(Xw, intr, C, om)

        if z <= 1e-9:
            return False # cheirality
        
        residuals[:] = (pred - self.obs) / self.sigma # scaled residual

        if jacobians is None:
            return True

        for k, J in ((0, J_intr[:, :1]), (1, J_intr[:, 1:]), (2, J_Cw), (3, J_om)):
            if jacobians[k] is not None:
                jacobians[k][:] = (J / self.sigma).ravel()

        # dPred/dTheta = d_pred/d_Xw * d_Xw/d_theta
        if jacobians[4] is not None:
            jacobians[4][:] = (J_Xw @ np.array([-r * st, r * ct, 0.0]) / self.sigma).ravel()

        # dPred/d_dR = d_pred/d_Xw * d_Xw/d_dR
        if jacobians[5] is not None:
            jacobians[5][:] = (J_Xw @ np.array([ct, st, 0.0]) / self.sigma).ravel()

        # dPred/d_dH = d_pred/d_Xw * d_Xw/d_dH only the top points
        if jacobians[6] is not None:
            jacobians[6][:] = ((J_Xw[:, 2]  / self.sigma) if self.is_top else np.zeros(2)).ravel()
        
        
        return True


class LandmarkCost(pyceres.CostFunction):
    """
        One canvas landmark sighting. The point is tracked across the frames.
        Blocks: [focal(1), pp(2), C(3), omega(3), Q(2)]

        Here we are actually optimizing the point points as well
    
    """

    def __init__(self, obs_uv, sigma):
        super().__init__()
        self.obs = np.asarray(obs_uv, np.float64)
        self.sigma = float(sigma)
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([1, 2, 3, 3, 2])

    def Evaluate(self, parameters, residuals, jacobians):
        foc, pp, C, om, Q = (np.asarray(p, np.float64) for p in parameters)
        intr = np.array([foc[0], pp[0], pp[1]])
        Xw = np.array([Q[0], Q[1], 0.0])
        pred, J_intr, J_Cw, J_om, J_Xw, z = _project_jac(Xw, intr, C, om)
        if z <= 1e-9:
            return False        # cheirality -- see PillarCost
        residuals[:] = (pred - self.obs) / self.sigma
        if jacobians is None:
            return True
        for k, J in ((0, J_intr[:, :1]), (1, J_intr[:, 1:]), (2, J_Cw), (3, J_om)):
            if jacobians[k] is not None:
                jacobians[k][:] = (J / self.sigma).ravel()
        if jacobians[4] is not None:                      # Z = 0, so only X and Y
            jacobians[4][:] = (J_Xw[:, :2] / self.sigma).ravel()
        return True

class PrincipalPointPrior(pyceres.CostFunction):
    """
        Weak pull of (cx, cy) toward the image centre. Block: [pp(2)].
    """

    def __init__(self, w, h, sigma):
        super().__init__()
        self.c = np.array([w / 2.0, h / 2.0])
        self.sigma = float(sigma)
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([2])

    def Evaluate(self, parameters, residuals, jacobians):
        pp = np.asarray(parameters[0], np.float64)
        residuals[:] = (pp - self.c) / self.sigma
        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][:] = (np.eye(2) / self.sigma).ravel()
        return True

class DeltaPrior(pyceres.CostFunction):
    """
        Weak pull of the radius offset toward zero. Block: [delta(1)].
    """

    def __init__(self, sigma):
        super().__init__()
        self.sigma = float(sigma)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(self, parameters, residuals, jacobians):
        residuals[0] = float(parameters[0][0]) / self.sigma
        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][:] = np.array([1.0 / self.sigma])
        return True

class HeightPrior(pyceres.CostFunction):
    """
        Weak pull of the pillar height offset toward zero. Block: [dh(1)].
    """

    def __init__(self, sigma):
        super().__init__()
        self.sigma = float(sigma)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(self, parameters, residuals, jacobians):
        residuals[0] = float(parameters[0][0]) / self.sigma
        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][:] = np.array([1.0 / self.sigma])
        return True


class StaticPointCost(pyceres.CostFunction):
    """
        Reprojection of a frozen world point. Blocks : [intr(3), C(3), omega(3)]

        The world point is baked in rather than being a parameter as we have fixed
        this point from calibration hence `StaticPointCost`.
        It also means canvas landmarks (Z = 0) and pillar points
        (Z = 0 or H_p) go through one identical path -- `LandmarkCost` could
        not, because it hard-codes Z = 0.

        Returns False behind the camera, so Ceres backs off instead of fitting a
        mirrored pose.
    """

    def __init__(self, X, obs_uv, sigma):
        super().__init__()
        self.X = np.asarray(X, np.float64)
        self.obs = np.asarray(obs_uv, np.float64)
        self.s = float(sigma)
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([3, 3, 3])

    def Evaluate(self, parameters, residuals, jacobians):
        intr, C, om = (np.asarray(p, np.float64) for p in parameters)
        pred, J_intr, J_C, J_om, _, z = _project_jac(self.X, intr, C, om)
        if z <= 1e-9:
            return False
        residuals[:] = (pred - self.obs) / self.s
        if jacobians is None:
            return True
        for k, J in ((0, J_intr), (1, J_C), (2, J_om)):
            if jacobians[k] is not None:
                jacobians[k][:] = (J / self.s).ravel()
        return True

    
class InvDepthCost(pyceres.CostFunction):
    """
       Fit geometric inverse depth from network disparity.

        Parameter block: [alpha, beta, gamma]

            q = alpha * d**gamma + beta
            residual = (q - 1 / z_geom) / sigma

        sigma is the residual standard deviation in inverse-depth units (1/m
        when z_geom is in metres).

        Fitting depth directly would use 1/q - z_geom, whose parameter
        derivative is -(1/q**2) * dq/dparams. This becomes unstable near q=0.
        Small inverse-depth errors also produce depth errors proportional to
        Z**2, giving distant points greater influence in a depth-space fit.

        This model is affine in transformed disparity d**gamma. It is affine
        in raw disparity only when gamma=1.
    """



    def __init__(self, d, z_geom, sigma):
        super().__init__()
        self.d = float(np.asarray(d).item())
        self.inv_z = 1.0 / float(np.asarray(z_geom).item())
        self.sigma = float(sigma)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([3])
        return

    def Evaluate(self, parameters, residuals, jacobians):
        alpha, beta, gamma = (float(v) for v in parameters[0])
        d = max(self.d, 1e-6)
        dg = d ** gamma
        residuals[0] = (alpha * dg + beta - self.inv_z) / self.sigma
        if jacobians is not None and jacobians[0] is not None:
            # d/dalpha = d^gamma ; d/dbeta = 1 ; d/dgamma = alpha*d^gamma*ln d
            jacobians[0][:] = np.array(
                [dg, 1.0, alpha * dg * np.log(d)]) / self.sigma
        return True