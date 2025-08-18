import os
import pandas as pd
import numpy as np
from projectaria_tools.core import data_provider

import time
from math import atan, sqrt

import numpy as np
import requests
from PIL import Image
from scipy.spatial.transform import Rotation
from iopath.common.file_io import PathManager
import math
path_manager = PathManager()
import json
import pandas as pd
from typing import Any, Dict, List


class CropInfo:
    """
    This class represents the crop information of an image.
    """

    min_corner = np.zeros(2)
    scale = 1

    # scale is assumed to be performed first, i.e., min_corner is
    # in the reference frame of the scaled image
    def __init__(self, min_corner=np.zeros(2), scale=1):
        self.min_corner = min_corner
        self.scale = scale

    # convert the coordinates p=[x,y] from an uncropped image into coordinates that work on the cropped image
    def project(self, p):
        x, y = p
        x = x * self.scale
        y = y * self.scale
        x -= self.min_corner[0]
        y -= self.min_corner[1]
        return [x, y]

    # convert the coordinates p=[x,y] from a cropped image into coordinates in the uncropped image
    def unproject(self, p):
        x, y = p
        x += self.min_corner[0]
        y += self.min_corner[1]
        x = x / self.scale
        y = y / self.scale
        return [x, y]


class CameraCalibrationFisheye624:
    """
    This class is a wrapper around the FisheyeRadTanThinPrism class.
    It adds the following:
    1. A constructor that takes in a json file and initializes the class with the parameters
    2. A method to project a 3D point into the image plane
    3. A method to unproject a 2D point into the 3D space
    4. A method to get the translation and rotation of the camera
    5. A method to transform a 3D point from camera to CPF frame
    Default:
        translation=np.zeros(3),
        rotation=Rotation.identity(),
        cropInfo=CropInfo(),
    """

    # translation and rotation describe the transformation that transforms a 3D point in camera frame to a 3D point in CPF
    def __init__(
        self,
        params,
        translation,
        rotation,
        cropInfo,
    ):
        self.projection = FisheyeRadTanThinPrism(
            numK=6, useTangential=True, useThinPrism=True, useSingleFocalLength=True
        )
        self.params = params
        self.translation = translation
        self.rotation = rotation
        self.cropInfo = cropInfo

    def project(self, pointOptical):
        return self.cropInfo.project(self.projection.project(pointOptical, self.params))

    def unproject(self, pointInImage):
        return self.projection.unproject(
            self.cropInfo.unproject(pointInImage), self.params
        )

    def get_translation(self):
        return self.translation

    def get_rotation(self):
        return self.rotation

    def transform(self, point3D):
        translated_point = point3D - self.translation
        rotated_point = self.rotation.inv().apply(translated_point)
        return rotated_point

    def inverseTransform(self, point3D):
        rotated_point = self.rotation.apply(point3D)
        translated_point = rotated_point + self.translation
        return translated_point


class FisheyeRadTanThinPrism:
    """
    This class implements the fisheye camera model.
    The model is described by the following equation:
    x = f_u * r_u * (r_cos_theta + r_sin_theta) + c_u
    y = f_v * r_v * (r_cos_theta - r_sin_theta) + c_v
    where r_u, r_v are the radial distortion parameters,
    r_cos_theta, r_sin_theta are the tangential distortion
    """

    def __init__(self, numK, useTangential, useThinPrism, useSingleFocalLength):
        self.numK = numK
        self.useTangential = useTangential
        self.useThinPrism = useThinPrism
        self.useSingleFocalLength = useSingleFocalLength
        self.kNumParams = (
            (4 - useSingleFocalLength) + numK + 2 * useTangential + 4 * useThinPrism
        )
        self.kNumDistortionParams = numK + 2 * useTangential + 4 * useThinPrism
        self.kFocalXIdx = 0
        self.kFocalYIdx = 1 - useSingleFocalLength
        self.kPrincipalPointColIdx = 2 - useSingleFocalLength
        self.kPrincipalPointRowIdx = 3 - useSingleFocalLength
        self.kIsFisheye = True
        self.kHasAnalyticalProjection = True
        self.startK = self.kPrincipalPointRowIdx + 1
        self.startP = self.startK + numK
        self.startS = self.startP + 2 * useTangential

    def project(self, pointOptical, params):
        # make sure we have nonnegative orders:
        assert self.numK >= 0, "nonnegative number required"

        # the parameter vector has the following interpretation:
        # params = [f_u {f_v} c_u c_v [k_0: k_{numK-1}]  {p_0 p_1} {s_0 s_1 s_2 s_3}]

        # projection computations begin
        # compute [a;b] = [x/z; y/z]
        # make sure the point is not on the image plane

        inv_z = 1.0 / pointOptical[2]
        ab = pointOptical[:2] * inv_z
        # compute the squares of the elements of ab
        ab_squared = np.square(ab)
        # these will be used in multiple computations
        r_sq = ab_squared[0] + ab_squared[1]
        r = sqrt(r_sq)
        th = atan(r)
        thetaSq = th * th

        th_radial = 1.0
        # compute the theta polynomial
        theta2is = thetaSq
        for i in range(self.numK):
            th_radial += theta2is * params[self.startK + i]
            theta2is *= thetaSq

        # compute th/r, using the limit for small values
        th_divr = 1.0 if r < np.finfo(float).eps else th / r

        # the distorted coordinates -- except for focal length and principal point
        # start with the radial term:
        xr_yr = (th_radial * th_divr) * ab
        xr_yr_squaredNorm = np.linalg.norm(xr_yr) ** 2

        # start computing the output: first the radially-distorted terms, then add more as needed
        uvDistorted = xr_yr

        if self.useTangential:
            temp = 2 * np.dot(xr_yr, params[self.startP : self.startP + 2])
            uvDistorted += (
                temp * xr_yr + xr_yr_squaredNorm * params[self.startP : self.startP + 2]
            )

        if self.useThinPrism:
            radialPowers2And4 = np.array([xr_yr_squaredNorm, xr_yr_squaredNorm**2])
            uvDistorted[0] += np.dot(
                params[self.startS : self.startS + 2], radialPowers2And4
            )
            uvDistorted[1] += np.dot(
                params[self.startS + 2 : self.startS + 4], radialPowers2And4
            )

        # compute the return value
        if self.useSingleFocalLength:
            return (
                params[0] * uvDistorted
                + params[self.kPrincipalPointColIdx : self.kPrincipalPointColIdx + 2]
            )
        else:
            return (
                uvDistorted * params[:2]
                + params[self.kPrincipalPointColIdx : self.kPrincipalPointColIdx + 2]
            )

    def scaleParams(self, s, params):
        scale = s
        params[self.kFocalXIdx] *= scale
        if not self.useSingleFocalLength:
            params[self.kFocalYIdx] *= scale
        params[self.kPrincipalPointColIdx] = (
            scale * (params[self.kPrincipalPointColIdx] + 0.5) - 0.5
        )
        params[self.kPrincipalPointRowIdx] = (
            scale * (params[self.kPrincipalPointRowIdx] + 0.5) - 0.5
        )
        # Distortion coefficients aren't affected by scale

    def subtractFromOrigin(self, u, v, params):
        params[self.kPrincipalPointColIdx] -= u
        params[self.kPrincipalPointRowIdx] -= v

    def unproject(self, p, params):
        # get uvDistorted:
        if self.useSingleFocalLength:
            uvDistorted = (
                p - params[self.kPrincipalPointColIdx : self.kPrincipalPointColIdx + 2]
            ) / params[0]
        else:
            uvDistorted = (
                p - params[self.kPrincipalPointColIdx : self.kPrincipalPointColIdx + 2]
            ) / params[:2]

        # get xr_yr from uvDistorted
        xr_yr = self.compute_xr_yr_from_uvDistorted(uvDistorted, params)

        # early exit if point is in the center of the image
        xr_yrNorm = np.linalg.norm(xr_yr)
        if xr_yrNorm == 0.0:
            return np.array([0.0, 0.0, 1.0])

        # otherwise, find theta
        theta = self.getThetaFromNorm_xr_yr(xr_yrNorm, params)

        # get the point coordinates:
        point3dEst = np.zeros(3)
        point3dEst[:2] = np.tan(theta) / xr_yrNorm * xr_yr
        point3dEst[2] = 1.0

        return point3dEst

    def compute_xr_yr_from_uvDistorted(self, uvDistorted, params):
        # early exit if we're not using any tangential/ thin prism distortions
        if not self.useTangential and not self.useThinPrism:
            return uvDistorted

        # initial guess:
        xr_yr = uvDistorted

        # do Newton iterations to find xr_yr
        for _ in range(10):  # 10 is a placeholder for the maximum number of iterations
            # compute the estimated uvDistorted
            uvDistorted_est = xr_yr
            xr_yr_squaredNorm = np.linalg.norm(xr_yr) ** 2

            if self.useTangential:
                temp = 2 * np.dot(xr_yr, params[self.startP : self.startP + 2])
                uvDistorted_est += (
                    temp * xr_yr
                    + xr_yr_squaredNorm * params[self.startP : self.startP + 2]
                )

            if self.useThinPrism:
                radialPowers2And4 = np.array([xr_yr_squaredNorm, xr_yr_squaredNorm**2])
                uvDistorted_est[0] += np.dot(
                    params[self.startS : self.startS + 2], radialPowers2And4
                )
                uvDistorted_est[1] += np.dot(
                    params[self.startS + 2 : self.startS + 4], radialPowers2And4
                )

            # compute the derivative of uvDistorted wrt xr_yr
            duvDistorted_dxryr = np.zeros((2, 2))
            self.compute_duvDistorted_dxryr(
                xr_yr, xr_yr_squaredNorm, params, duvDistorted_dxryr
            )

            # compute correction:
            correction = np.linalg.pinv(duvDistorted_dxryr).dot(
                uvDistorted - uvDistorted_est
            )

            xr_yr += correction

            if (
                np.linalg.norm(correction) < 1e-6
            ):  # 1e-6 is a placeholder for the convergence threshold
                break

        return xr_yr

    def getThetaFromNorm_xr_yr(self, th_radialDesired, params):
        # initial guess
        th = th_radialDesired

        for _ in range(10):  # 10 is a placeholder for the maximum number of iterations
            thetaSq = th * th

            th_radial = 1
            dthD_dth = 1
            # compute the theta polynomial and its derivative wrt theta
            theta2is = thetaSq
            for i in range(self.numK):
                th_radial += theta2is * params[self.startK + i]
                dthD_dth += (2 * i + 3) * params[self.startK + i] * theta2is
                theta2is *= thetaSq
            th_radial *= th

            # compute the correction:
            if abs(dthD_dth) > 1e-6:  # 1e-6 is a placeholder for the epsilon value
                step = (th_radialDesired - th_radial) / dthD_dth
            else:
                step = (
                    (th_radialDesired - th_radial) * dthD_dth * 10
                    if (th_radialDesired - th_radial) * dthD_dth > 0.0
                    else -(th_radialDesired - th_radial) * dthD_dth * 10
                )

            # apply correction
            th += step

            if abs(step) < 1e-6:  # 1e-6 is a placeholder for the convergence threshold
                break

            # revert to within 180 degrees FOV to avoid numerical overflow
            if abs(th) >= np.pi / 2.0:
                th = 0.999 * np.pi / 2.0

        return th

    def compute_duvDistorted_dxryr(
        self, xr_yr, xr_yr_squaredNorm, params, duvDistorted_dxryr
    ):
        if self.useTangential:
            duvDistorted_dxryr[0, 0] = (
                1
                + 6 * xr_yr[0] * params[self.startP]
                + 2 * xr_yr[1] * params[self.startP + 1]
            )
            offdiag = 2 * (
                xr_yr[0] * params[self.startP + 1] + xr_yr[1] * params[self.startP]
            )
            duvDistorted_dxryr[0, 1] = offdiag
            duvDistorted_dxryr[1, 0] = offdiag
            duvDistorted_dxryr[1, 1] = (
                1
                + 6 * xr_yr[1] * params[self.startP + 1]
                + 2 * xr_yr[0] * params[self.startP]
            )
        else:
            duvDistorted_dxryr = np.eye(2)

        if self.useThinPrism:
            temp1 = 2 * (
                params[self.startS] + 2 * params[self.startS + 1] * xr_yr_squaredNorm
            )
            duvDistorted_dxryr[0, 0] += xr_yr[0] * temp1
            duvDistorted_dxryr[0, 1] += xr_yr[1] * temp1

            temp2 = 2 * (
                params[self.startS + 2]
                + 2 * params[self.startS + 3] * xr_yr_squaredNorm
            )
            duvDistorted_dxryr[1, 0] += xr_yr[0] * temp2
            duvDistorted_dxryr[1, 1] += xr_yr[1] * temp2

def load_camera_config_from_json(camera_config_file):
    with path_manager.open(camera_config_file, "r") as json_file:
        loaded_data = json.load(json_file)
    camera_config = {}
    camera_config["overall_transform"] = np.array(loaded_data["overall_transform"])
    camera_config["cpf_transform"] = np.array(loaded_data["cpf_transform"])

    min_corner = np.array(loaded_data["min_corner"])
    scale = loaded_data["scale"]
    cropAria = CropInfo(min_corner, scale)  # default
    paramsAria = np.array(loaded_data["paramsAria"])
    camTranslationAria = np.array(loaded_data["camTranslationAria"])
    quatAria = np.array(loaded_data["quatAria"])
    camRotationAria = Rotation.from_quat(quatAria)
    camCalibration = CameraCalibrationFisheye624(
        paramsAria, camTranslationAria, camRotationAria, cropAria
    )

    camera_config["camera_calibration"] = camCalibration

    return camera_config
    
def get_default_camera_config():
    loaded_data = {'overall_transform': [[ 0.0114827 , -0.99071809, -0.13504478,  0.00812104],
                                        [ 0.99961912,  0.0083608 ,  0.02357818, -0.0580445 ],
                                        [-0.02222674, -0.13528985,  0.99050637, -0.01013661]],
                    'cpf_transform': [[-0.03186013725398906, 0.7933534010715845, 0.6079270619591683, 0.0702907945], 
                                            [-0.9986295320622568, 1.7481548431064198e-09, -0.05233600761538715, 0.002372], 
                                            [-0.04152095070292633, -0.608761349798219, 0.79226613561642, 0.00882276712], 
                                            [0.0, 0.0, 0.0, 1.0]], 
                    'min_corner': [0, 0], 'scale': 1, 
                    'paramsAria': [609.616187, 713.198294, 708.488070, 0.389290834, 
                                -0.359538964, -0.212498504, 1.67088963, -2.05240150,
                                    0.738013179,  1.66051788e-04, -1.70460771e-04, -4.53193147e-04,
                                    6.52964246e-05,  3.72170495e-04, -2.47378464e-04],
                    'camTranslationAria': [-0.00453888,-0.01222385,-0.00497373], 
                    'quatAria': [0.94228508, 0.3307318,  0.03789971, 0.03515514]}
    camera_config = {}
    camera_config["overall_transform"] = np.array(loaded_data["overall_transform"])
    camera_config["cpf_transform"] = np.array(loaded_data["cpf_transform"])

    min_corner = np.array(loaded_data["min_corner"])
    scale = loaded_data["scale"]
    cropAria = CropInfo(min_corner, scale)  # default
    paramsAria = np.array(loaded_data["paramsAria"])
    camTranslationAria = np.array(loaded_data["camTranslationAria"])
    quatAria = np.array(loaded_data["quatAria"])
    camRotationAria = Rotation.from_quat(quatAria)
    camCalibration = CameraCalibrationFisheye624(
        paramsAria, camTranslationAria, camRotationAria, cropAria
    )

    camera_config["camera_calibration"] = camCalibration

    return camera_config


def binocular_yaw_pitch_to_vector(DF):
    """
    Convert yaw and pitch to gaze vector
    DF: dataframe with yaw and pitch of left and right eyes

    return dataframe with gaze point position in 3D
    """

    DF["transformed_gaze_x"] = np.nan
    DF["transformed_gaze_y"] = np.nan
    DF["transformed_gaze_z"] = np.nan

    ipd = 63  # in mm
    half_ipd = ipd / 2.0
    tan_left_rad = np.tan(DF.left_yaw_rads_cpf)
    tan_right_rad = np.tan(DF.right_yaw_rads_cpf)

    for i in np.arange(len(DF)):
        # in case there is no intersection, we still want a valid direction vector
        tan_right_minus_left = max(tan_right_rad[i] - tan_left_rad[i], 1e-6)
        intersection_x = (
            half_ipd * (tan_left_rad[i] + tan_right_rad[i]) / tan_right_minus_left
        )
        intersection_z = ipd / tan_right_minus_left
        intersection_y = intersection_z * np.tan(DF.pitch_rads_cpf[i])

        v = np.array([intersection_x, intersection_y, intersection_z]) / np.linalg.norm(
            np.array([intersection_x, intersection_y, intersection_z])
        )
        DF.at[i, "transformed_gaze_x"] = v[0]
        DF.at[i, "transformed_gaze_y"] = v[1]
        DF.at[i, "transformed_gaze_z"] = v[2]

    return DF

def yaw_pitch_to_vector(DF: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yaw and pitch to gaze vector
    DF: dataframe with yaw and pitch
    """

    # for i in np.arange(len(DF)):

    DF["transformed_gaze_x"] = np.nan
    DF["transformed_gaze_y"] = np.nan
    DF["transformed_gaze_z"] = np.nan

    for i in np.arange(len(DF)):
        yaw = DF.yaw_rads_cpf[i]
        pitch = DF.pitch_rads_cpf[i]
        x = math.tan(yaw)
        y = math.tan(pitch)
        z = 1
        v = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
        DF.at[i, "transformed_gaze_x"] = v[0]
        DF.at[i, "transformed_gaze_y"] = v[1]
        DF.at[i, "transformed_gaze_z"] = v[2]
    return DF



class GazeRay:
    """
    This class represents a gaze ray.
    """

    def __init__(self, timestamp, origin, direction, depth):
        self.timestamp = timestamp
        self.origin = origin
        self.direction = direction
        self.depth = depth

    def __repr__(self):
        return (
            "time: "
            + str(self.timestamp)
            + ", origin: "
            + str(self.origin)
            + ", dir: "
            + str(self.direction)
        )


class GazeRays:
    """
    This class is a wrapper around a list of gaze rays.
    It adds the following:
    1. A method to get a subset of the gaze rays based on a time range
    2. A method to head correct the gaze rays
    3. A method to read gaze rays from a csv file
    4. A method to write gaze rays to a csv file
    """

    def __init__(self, rays):

        self.rays = rays

    def get_subset(self, from_time, to_time):
        subset_rays = [
            ray for ray in self.rays if from_time <= ray.timestamp <= to_time
        ]
        return GazeRays(subset_rays)

    def head_correction(self, head_transforms, target_timestamp, transformation_to_cpf):
        target_transform = head_transforms.get_by_time(target_timestamp)
        inverse_target_transform = np.linalg.inv(target_transform.matrix)
        transformed_gazes = []

        transformation_to_device = np.linalg.inv(transformation_to_cpf)

        # print("change4")
        for gaze in self.rays:
            # find the closest head pose
            head_pose = head_transforms.get_by_time(gaze.timestamp)
            # head pose differential
            head_pose_diff = inverse_target_transform @ head_pose.matrix

            head_pose_diff = (
                transformation_to_cpf @ head_pose_diff @ transformation_to_device
            )

            # apply head pose
            origin_4d = np.append(gaze.origin, 1)
            transformed_origin = np.dot(head_pose_diff, origin_4d)[:3]
            direction_4d = np.append(gaze.direction, 0)
            transformed_direction = np.dot(head_pose_diff, direction_4d)[:3]

            transformed_gazes.append(
                GazeRay(
                    gaze.timestamp,
                    transformed_origin,
                    transformed_direction,
                    gaze.depth,
                )
            )
        return GazeRays(transformed_gazes)

    @classmethod
    def load_from_dataframe(cls, df):
        gaze_data = []
        for _, row in df.iterrows():
            timestamp = row["tracking_timestamp_us"]  # need update
            direction = np.array(
                [
                    row["transformed_gaze_x"],
                    row["transformed_gaze_y"],
                    row["transformed_gaze_z"],
                ]
            )
            origin = np.array([0, 0, 0])
            if "depth_m" in row:
                depth = row["depth_m"]
            else:
                depth = float("nan")
            gaze_data.append(GazeRay(timestamp, origin, direction, depth))
        return GazeRays(gaze_data)

    @classmethod
    def read_from_file(cls, filename):
        df = pd.read_csv(filename)
        return cls.load_from_dataframe(df)



def gazeray_projection(
    eye_gaze_df: pd.DataFrame,
    head_transform_df: pd.DataFrame,
    camera_config: Dict[str, Any],
    timestamp_of_projection: float,
    timestamp_start: float,
    timestamp_end: float,
    depth_estimation: float,
):
    """
    Project gaze rays to RGB camera
    Inputs:
        eye_gaze_df: dataframe with eye gaze data
        head_transform_df: dataframe with head transform data
        camera_config: dictionary with rgb camera calibration
        timestamp_of_projection: the timestamp of the frame to be projected in seconds
        timestamp_start: start time of the recording in seconds
        timestamp_end: end time of the recording in seconds
        depth_estimation: estimated depth of the gaze point, usually set to 1 meter
    Outputs:
        A pandas dataframe that contains the 2D projected gaze points for the assigned frame
    """
    #timestamp_of_projection_us = int(timestamp_of_projection * 1e6)
    #timestamp_start_us = int(timestamp_start * 1e6)
    #timestamp_end_us = int(timestamp_end * 1e6)
    #eye_gaze_df = eye_gaze_df.loc[
    #    (eye_gaze_df["tracking_timestamp_us"] >= timestamp_start_us)
    #    & (eye_gaze_df["tracking_timestamp_us"] <= timestamp_end_us)
    #]

    # process eye gaze
    eye_gaze_df = eye_gaze_df.reset_index(drop=True)
    if not ("transformed_gaze_x" in eye_gaze_df.keys()):
        if "left_yaw_rads_cpf" in eye_gaze_df.keys():
            eye_gaze_df = binocular_yaw_pitch_to_vector(eye_gaze_df)
        elif "yaw_rads_cpf" in eye_gaze_df.keys():
            eye_gaze_df = yaw_pitch_to_vector(eye_gaze_df)
        else:
            print("invalid gaze dataframe")
    if len(eye_gaze_df) == 0:
        raise IndexError("Invalid time window for gaze data")

    # process head transform
    #head_trans_df = head_transform_df.loc[
    #    (head_transform_df["tracking_timestamp_us"] >= timestamp_start_us - 1e6)
    #    & (head_transform_df["tracking_timestamp_us"] <= timestamp_end_us + 1e6)
    #]
    #head_trans_df = head_trans_df.reset_index(drop=True)

    #if len(head_trans_df) == 0:
    #    raise IndexError("Invalid time window for head pose data")

    # projection
    #head_transforms = HeadTransforms.load_from_dataframe(head_trans_df)
    gazes_transformed = GazeRays.load_from_dataframe(eye_gaze_df)
    #gazes_transformed = gaze_rays.head_correction(
    #    head_transforms, timestamp_of_projection_us, camera_config["cpf_transform"]
    #)

    gaze_points_2d = []

    for gazeray in gazes_transformed.rays:
        p = project_gaze_point_to_rgb(
            gazeray.origin, gazeray.direction, gazeray.depth, camera_config
        )  # if replace depth_m with gazeray.depth, then vergence depth is used!
        gaze_points_2d.append(p)

    gaze_points_2d = np.array(gaze_points_2d)
    gaze_points_2d_x = gaze_points_2d[:, 0].tolist()
    gaze_points_2d_y = gaze_points_2d[:, 1].tolist()
    eye_gaze_df = eye_gaze_df.assign(projected_point_2d_x=gaze_points_2d_x)
    eye_gaze_df = eye_gaze_df.assign(projected_point_2d_y=gaze_points_2d_y)

    return eye_gaze_df

def project_gaze_point_to_rgb(origin, direction, est_depth, camera_config):
    """
    Project a 3D gaze point to RGB camera
    Inputs:
        origin: 3D gaze point
        direction: gaze direction
        est_depth: estimated depth
        camera_config: camera config CameraCalibrationFisheye624
    Outputs:
        p1: x coordinate in RGB camera
        p2: y coordinate in RGB camera
    """
    gaze_target_in_cpf = direction * est_depth + origin
    g = gaze_target_in_cpf
    gaze_target_in_camera_rgb = camera_config["overall_transform"] @ np.array(
        [g[0], g[1], g[2], 1]
    )
    ptInRgb = camera_config["camera_calibration"].project(gaze_target_in_camera_rgb)
    p1 = 1408 - ptInRgb[1]  # rotate -90 degrees
    p2 = ptInRgb[0]
    p1 = 1408 - p1
    return [p1, p2]


def get_calibs(provider):
    device_calib = provider.get_device_calibration()
    # print(device_calib.get_device_subtype())
    label = "camera-rgb"
    #transform_device_sensor = device_calib.get_transform_device_sensor(label)
    #transform_device_cpf = device_calib.get_transform_device_cpf()
    transform_cpf_rgb = device_calib.get_transform_cpf_sensor(label, get_cad_value=False)
    rt = transform_cpf_rgb.inverse()
    rt = rt.to_matrix()
    # returns None if the calibration label does not exist
    cam_calib = device_calib.get_camera_calib(label)
    return cam_calib, rt


def compute_depth_and3rdeye(preds):
    """
    preds: row x column. Columns: left yaw, right yaw, pitch
    """
    ipd = 0.063
    d = ipd / 2
    # if isinstance(preds, np.ndarray):
    tan = np.tan
    atan = np.arctan
    lan = np.linalg.norm
    cos = np.cos
    thirdeye = np.zeros((preds.shape[0], 2))
    intersection_xyz = np.zeros((preds.shape[0], 3))
    intersection_xyz[:, 0] = (
        d
        * (tan(preds[:, 0]) + tan(preds[:, 1]))
        / (tan(preds[:, 1]) - tan(preds[:, 0]))
    )  # x
    intersection_xyz[:, 2] = 2 * d / (tan(preds[:, 1]) - tan(preds[:, 0]))  # z
    intersection_xyz[:, 1] = intersection_xyz[:, 2] * tan(preds[:, 2])  # y
    #print(intersection_xyz)
    r = lan(intersection_xyz, axis=1)

    thirdeye[:, 0] = atan(intersection_xyz[:, 0] / intersection_xyz[:, 2])
    thirdeye[:, 1] = preds[:, 2]
    return r, thirdeye



def polar_to_xyz(polar: np.ndarray, z=None) -> np.ndarray:
    """Converts polar coordinate np arrays to xyz dimensions. Undoing xyz_to_polar
    to compute degree error.

    Turning [arctan(x/z), arctan(y/z)] -> [x, y, z], assuming z is large, since
    we only care about angle
    Args:
        polar: Nx2 polar coordinate vector or Nx3 for 3rd column as Z
        z: scaling factor (z coordinate) to resacle the output results
    returns:
        Nx3 xyz coordiante vector
    """
    # print("polar input", polar)
    # if isinstance(polar, np.ndarray):
    xyz = np.empty((polar.shape[0], 3), dtype=polar.dtype)
    tan = np.tan
    mul = np.multiply

    # Since we are only interested in angle, we can lower prediction error by
    # ignoring the 3rd dimension (which specifies distance)
    # We normalize the vector to fixed z. Assume z is 10 meters, if not given
    z = z if z else 10
    z = z if polar.shape[1] < 3 else polar[:, 2]
    xyz[:, 2] = z
    xyz[:, 0] = mul(tan(polar[:, 0]), z)
    xyz[:, 1] = mul(tan(polar[:, 1]), z)
    return xyz


def polar_to_unit_vector(p_polar):
    p_xyz = polar_to_xyz(np.array(p_polar).reshape(1, -1), 100)
    p_xyz_unitVec = p_xyz / np.linalg.norm(p_xyz)
    return p_xyz_unitVec


def get_eyegaze_point_at_depth(yaw, pitch, depth):
    """
    yaw: left yaw, right yaw, pitch
    depth: depth in meters
    """
    point3d = polar_to_unit_vector(np.array([yaw, pitch])) * depth
    return point3d


def get_proj(cpf_3d, cam_calib, RT):
    cpf_3d = np.append(cpf_3d, 1)
    cpf_sensor = np.dot(RT, cpf_3d)
    proj = cam_calib.project(cpf_sensor[:3])
    return proj

def get_projections_et(
    latest_et_df: pd.DataFrame,
    cam_calib,
    cpf_to_rgb_T,
):
    
    cpf_origin = np.array([0, 0, 0])
    projections = []
    directions = []
    
    for row in latest_et_df.iterrows():
        V_cpf = get_eyegaze_point_at_depth(
            row[1]["yaw_rads_cpf"],
            row[1]["pitch_rads_cpf"],
            row[1]["depth_m"],
        )
        direction = V_cpf.flatten() - cpf_origin        
        V_proj = get_proj(V_cpf, cam_calib, cpf_to_rgb_T)
        
        if V_proj is None:
            if len(projections) > 0:
                V_proj = projections[-1]
                direction = directions[-1]
            else: 
                V_proj = np.array([704,704], dtype=np.float64)

        projections.append((V_proj[0], V_proj[1]))
        directions.append(direction.flatten())

    return projections, directions







def project_gaze(gaze_path, config_path=None, vrs_path=None):
    """
    Input:
    gaze_path: path to either general or personalized eye gaze .csv file
    and either:
        config_path: path to "camera_config.json" 
        vrs_path: path to .vrs file    
    if neither config_path or vrs_path is provided, then a default camera config is used (this is a simple mean of all parameters)

    Output:
    A dataframe with
    cols = ["tracking_timestamp_us", "projected_point_2d_x", "projected_point_2d_y", "transformed_gaze_x", "transformed_gaze_y", "transformed_gaze_z", "depth_m"]
    """
    if isinstance(gaze_path, str):
        gaze = pd.read_csv(gaze_path, engine='python')
    else:
        gaze = gaze_path #use gaze df

    #solution 1: using vrs
    if vrs_path:
        provider = data_provider.create_vrs_data_provider(vrs_path)
        cam_calib, cpf_to_rgb_T = get_calibs(provider)
        cols = ["left_yaw_rads_cpf", "right_yaw_rads_cpf", "pitch_rads_cpf"]
        _, third_eye = compute_depth_and3rdeye(gaze.loc[:, cols].to_numpy())
        gaze["yaw_rads_cpf"] = third_eye[:, 0]
        gaze["pitch_rads_cpf"] = third_eye[:, 1]
        projections, directions = get_projections_et(latest_et_df=gaze, cam_calib=cam_calib, cpf_to_rgb_T=cpf_to_rgb_T)
        projections = np.array(projections)
        directions = np.array(directions)  / np.linalg.norm(directions, axis=-1, keepdims=True)
        data = {
            "tracking_timestamp_us": gaze["tracking_timestamp_us"],
            "projected_point_2d_x": projections[:, 1],
            "projected_point_2d_y": projections[:, 0],
            "transformed_gaze_x": directions[:, 0],
            "transformed_gaze_y": directions[:, 1],
            "transformed_gaze_z": directions[:, 2],
            "depth_m": gaze["depth_m"]
        }
        # Create a DataFrame from the dictionary
        out = pd.DataFrame(data)

    #solution 2: use config
    else:
        #use default config if not provided
        camera_config = load_camera_config_from_json(config_path) if config_path else get_default_camera_config()
        processed_gaze_df = gazeray_projection(gaze, None, camera_config, None, None, None, None)
        cols = ["tracking_timestamp_us", "projected_point_2d_x", "projected_point_2d_y", "transformed_gaze_x", "transformed_gaze_y", "transformed_gaze_z", "depth_m"]
        out = processed_gaze_df.loc[:, cols]

    return out

if __name__ == "__main__":
    recordings_dir = "/source_1a/data/contextual_ai/reading_itw_intern/debug_data/1110820370024666/" 
    gaze_path = os.path.join(recordings_dir, "EyeGaze", "personalized_eye_gaze.csv")
    config_path = os.path.join(recordings_dir, "vrs_videos", "camera_config.json") #some have no calib

    #solution 1: using camera_config   
    gaze = project_gaze(gaze_path, config_path=config_path)
    print(gaze)
    #solution 2: using vrs
    vrs_path = os.path.join(recordings_dir, "vrs_videos", "_33287c9e-0b89-408c-901b-8893ba1a4542_Ariane_ResearchScene.vrs")
    gaze = project_gaze(gaze_path, vrs_path=vrs_path)
    print(gaze)
    #should return the same results
    