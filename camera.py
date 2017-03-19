##########################################################################
# Basic camera class port from Sascha Willems - www.saschawillems.de
#
# Copyright (C) 2017 by Shi Chi
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
##########################################################################

import math

import numpy as np

import glm


class _CameraType(object):

    def __init__(self):
        pass

    @property
    def lookat(self):
        return 0

    @property
    def firstperson(self):
        return 1
CameraType = _CameraType()

class _Matrices(object):

    def __init__(self):
        self.view = np.identity(4, np.float32)
        self.perspective = np.identity(4, np.float32)


class _Keys(object):

    def __init__(self):
        self.left = False
        self.right = False
        self.up = False
        self.down = False


class Camera(object):

    def __init__(self):
        self.type = CameraType.lookat

        self.rotation = np.array([0, 0, 0], np.float32)
        self.position = np.array([0, 0, 0], np.float32)

        self.rotationSpeed = 1.0
        self.movementSpeed = 1.0

        self.matrices = _Matrices()
        self.keys = _Keys()

        self.__fov = 0.0
        self.__znear = 0.0
        self.__zfar = 0.0

    def __updateViewMatrix(self):
        rotM = np.identity(4, np.float32)
        rotM = glm.rotate(rotM, self.rotation[0], 1.0, 0.0, 0.0)
        rotM = glm.rotate(rotM, self.rotation[1], 0.0, 1.0, 0.0)
        rotM = glm.rotate(rotM, self.rotation[2], 0.0, 0.0, 1.0)

        transM = glm.translate(np.identity(4, np.float32), self.position[0], self.position[1], self.position[2])

        if self.type == CameraType.firstperson:
            self.matrices.view = rotM * transM
        else:
            self.matrices.view = transM * rotM

    def moving(self):
        return self.keys.left or self.keys.right or self.keys.up or self.keys.down

    def setPrespective(self, fov, aspect, znear, zfar):
        self.__fov = fov
        self.__znear = znear
        self.__zfar = zfar
        self.matrices.perspective = glm.perspective(self.__fov, aspect, self.__znear, self.__zfar)

    def updateAspectRatio(self, aspect):
        self.matrices.perspective = glm.perspective(self.__fov, aspect, self.__znear, self.__zfar)

    def setRotation(self, rotation):
        self.rotation = rotation
        self.__updateViewMatrix()

    def rotate(self, delta):
        self.rotation += delta
        self.__updateViewMatrix()

    def setTranslation(self, translation):
        self.position = translation
        self.__updateViewMatrix()

    def translate(self, delta):
        self.position += delta
        self.__updateViewMatrix()

    def update(self, deltaTime):
        if self.type == CameraType.firstperson:
            if self.moving():
                x = -math.cos(self.rotation[0]) * math.sin(self.rotation[1])
                y = math.sin(self.rotation[0])
                z = math.cos(self.rotation[0]) * math.cos(self.rotation[1])
                camFront = glm.normalize(np.array([x, y, z], np.float32))

                moveSpeed = deltaTime * self.movementSpeed

                if self.keys.up:
                    self.position += camFront * moveSpeed
                if self.keys.down:
                    self.position -= camFront * moveSpeed
                if self.keys.left:
                    self.position -= glm.normalize(np.cross(camFront, np.array([0, 1, 0], np.float32))) * moveSpeed
                if self.keys.right:
                    self.position += glm.normalize(np.cross(camFront, np.array([0, 1, 0], np.float32))) * moveSpeed

                self.__updateViewMatrix()

    def updatePad(self, axisLeft, axisRight, deltaTime):
        retVal = False

        if self.type == CameraType.firstperson:

            deadZone = 0.0015
            range_ = 1.0 - deadZone

            x = -math.cos(self.rotation[0]) * math.sin(self.rotation[1])
            y = math.sin(self.rotation[0])
            z = math.cos(self.rotation[0]) * math.cos(self.rotation[1])
            camFront = glm.normalize(np.array([x, y, z], np.float32))

            moveSpeed = deltaTime * self.movementSpeed * 2.0
            rotSpeed = deltaTime * self.rotationSpeed * 50.0

            if math.fabs(axisLeft[1]) > deadZone:
                pos = (math.fabs(axisLeft[1]) - deadZone) / range_
                xx = -1.0 if axisLeft[1] < 0.0 else 1.0
                self.position -= camFront * pos * xx * moveSpeed
                retVal = True
            if math.fabs(axisLeft[0]) > deadZone:
                pos = (math.fabs(axisLeft[1]) - deadZone) / range_
                xx = -1.0 if axisLeft[0] < 0.0 else 1.0
                self.position += glm.normalize(np.cross(camFront, np.array([0, 1, 0], np.float32)) * pos * xx) * moveSpeed
                retVal = True

            if math.fabs(axisRight[0]) > deadZone:
                pos = (math.fabs(axisRight[0]) - deadZone) / range_
                xx = -1.0 if axisRight[0] < 0.0 else 1.0
                self.rotation[1] += pos * xx * rotSpeed
                retVal = True
            if math.fabs(axisRight[1]) > deadZone:
                pos = (math.fabs(axisRight[1]) - deadZone) / range_
                xx = -1.0 if axisRight[1] < 0.0 else 1.0
                self.rotation[0] -= pos * xx * rotSpeed
                retVal = True
        else:
            pass

        if retVal:
            self.__updateViewMatrix()

        return retVal

