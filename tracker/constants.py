
import cv2

BAND_TOP_FRAC: float = 1.0
BAND_HEIGHT_FRAC: float = 1.0 / 6.0

LK = dict(winSize=(21, 21),
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
DEFAULT_FB_TOL = 1.0      # px round-trip error above which a track is dropped
DEFAULT_LEVELS = 3        # pyramid levels; 0 = no pyramid
DEFAULT_DILATE = 12       # px to grow the person mask by

WIN = 21 # LK window
MIN_POSE_PTS = 4  # fewer than this and the pose is not solved at all
SIGMA_MSER = 6.0               # px, replenished (MSER-discovered) observations

PROMOTE_FRAMES = 5             # consecutive good frames before PROVISIONAL->ACTIVE

REPLENISH_TOL = 1.5            # px reprojection to keep aging a PROVISIONAL point

REACQ_WINDOW = 30      # px half-width around the POSE-PREDICTED position
REACQ_NCC = 0.70       # min correlation of the plane-warped reference patch
REACQ_PATCH = 12       # reference patch half-size
PROMOTE_TOL = 3.0      # px reprojection after refit, to promote TENTATIVE


REPLENISH_CANVAS_TARGET = 4    # target ACTIVE+PROVISIONAL canvas points
REPLENISH_POST_TARGET = 6      # target ACTIVE+PROVISIONAL points per post face
REPLENISH_BAND_TARGET = 8      # ... per band panel. The band is never
                               # occluded by fighters and spans the frame
                               # width, so it is the one surface whose
                               # population does not collapse mid-clip.
REPLENISH_MIN_SEP = 20.0       # px; do not reseed near an already-tracked point
REPLENISH_MAX_NEW = 10         # cap new detections per surface per frame