##########################################################################
# Mesh loader for creating Vulkan resources from models loaded
#                                - port from Sascha Willems - www.saschawillems.de
#
# Copyright (C) 2017 by Shi Chi
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
##########################################################################


from pyVulkan import *
import pyassimp as assimp
import numpy as np

import vulkanbuffer as vks

class VertexLayout(object):

    VERTEX_COMPONENT_POSITION = 0x0
    VERTEX_COMPONENT_NORMAL = 0x1
    VERTEX_COMPONENT_COLOR = 0x2
    VERTEX_COMPONENT_UV = 0x3
    VERTEX_COMPONENT_TANGENT = 0x4
    VERTEX_COMPONENT_BITANGENT = 0x5
    VERTEX_COMPONENT_DUMMY_FLOAT = 0x6
    VERTEX_COMPONENT_DUMMY_VEC4 = 0x7

    def __init__(self, components):
        self.components = components

    def stride(self):
        res = 0
        for component in self.components:
            if component == VertexLayout.VERTEX_COMPONENT_UV:
                res += 2 * 4
            elif component == VertexLayout.VERTEX_COMPONENT_DUMMY_FLOAT:
                res += 4
            elif component == VertexLayout.VERTEX_COMPONENT_DUMMY_VEC4:
                res += 4 * 4
            else:
                res += 3 * 4

        return res

class ModelCreateInfo(object):

    def __init__(self, center=[0, 0, 0], scale=[0, 0, 0], uvscale=[0, 0, 0]):
        self.center = center
        self.scale = scale
        self.uvscale = uvscale
        
class ModelPart(object):
    def __init__(self):
        self.vertexBase = 0
        self.vertexCount = 0
        self.indexBase = 0
        self.indexCount = 0

class Dimension(object):
    FLT_MAX = 3.402823466e+38

    def __init__(self):
        self.min = [Dimension.FLT_MAX, Dimension.FLT_MAX, Dimension.FLT_MAX]
        self.max = [-Dimension.FLT_MAX, -Dimension.FLT_MAX, -Dimension.FLT_MAX]
        self.size = 0

# Simple mesh class for getting all the necessary stuff from models loaded via ASSIMP
class Model(object):

    def __init__(self, vulkanDevice=None):
        self.device = vulkanDevice
        self.vertices = vks.Buffer()
        self.indices = vks.Buffer()
        self.indexCount = 0
        self.vertexCount = 0

        self.parts = []
        self.dim = Dimension()

        self.scene = None

    def __del__(self):
        self.parts = []
        if self.scene:
            assimp.release(self.scene)

        if self.vertices.buffer:
            vkDestroyBuffer(self.device, self.vertices.buffer, None)
        if self.vertices.memory:
            vkFreeMemory(self.device, self.vertices.memory, None)

        if self.indices.buffer != VK_NULL_HANDLE:
            vkDestroyBuffer(self.device, self.indices.buffer, None)
            if self.indices.memory:
                vkFreeMemory(self.device, self.indices.memory, None)

    @property
    def defaultFlags(self):
        return assimp.postprocess.aiProcess_FlipWindingOrder | assimp.postprocess.aiProcess_Triangulate | assimp.postprocess.aiProcess_PreTransformVertices | assimp.postprocess.aiProcess_CalcTangentSpace | assimp.postprocess.aiProcess_GenSmoothNormals
        # return assimp.postprocess.aiProcess_Triangulate | assimp.postprocess.aiProcess_CalcTangentSpace | assimp.postprocess.aiProcess_FlipUVs

    def loadFromFile(self, filename, layouts, scale, device, copyQueue, createInfo=None, flags=None):
        '''Load a scene from a supported 3D file format'''
        self.device = device.logicalDevice
        self.vertices.device = self.device
        self.indices.device = self.device

        _scale = [1.0, 1.0, 1.0]
        uvscale = [1.0, 1.0]
        center = [0.0, 0.0, 0.0]
        if createInfo is None:

            if scale:
                createInfo = ModelCreateInfo([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], scale)
            else:
                createInfo = ModelCreateInfo(center, _scale, uvscale)
        else:
            scale = createInfo.scale
            uvscale = createInfo.uvscale
            center = createInfo.center

        if flags is None:
            flags = self.defaultFlags

        self.scene = assimp.load(filename, processing=flags)

        if self.scene:
            self.parts = []

            vertexBuffer = []
            indexBuffer = []
            dimMax = []
            dimMin = []
            for i, mesh in enumerate(self.scene.meshes):
                part = ModelPart()
                part.vertexBase = self.vertexCount
                part.indexBase = self.indexCount

                numVertice = len(mesh.vertices)
                self.vertexCount += numVertice
                part.vertexCount = numVertice

                diffuse = mesh.material.properties[('diffuse', 0)][:3]
                pos = mesh.vertices
                normal = mesh.normals
                texCoord = np.compress([True, True, False], mesh.texturecoords[0],
                                       axis=1) if np.any(mesh.texturecoords) else np.zeros((numVertice, 2), mesh.vertices.dtype).reshape(numVertice, 2)
                tangent = mesh.tangents if np.any(mesh.tangents) else np.zeros_like(mesh.vertices, mesh.vertices.dtype).reshape(numVertice, 3)
                biTangent = mesh.bitangents if np.any(mesh.bitangents) else np.zeros_like(mesh.vertices, mesh.vertices.dtype).reshape(numVertice, 3)
                color = mesh.colors if np.any(mesh.colors) else np.array(diffuse * numVertice, mesh.vertices.dtype).reshape((numVertice, len(diffuse)))

                for i in xrange(len(pos)):
                    for component in layouts.components:
                        if component == VertexLayout.VERTEX_COMPONENT_POSITION:
                            # vertexBuffer += np.ravel(pos*scale*[1, -1, 1] + center).tolist()
                            # vertexBuffer += (pos[i]*scale*[1, -1, 1] + center).flatten().tolist()
                            vertexBuffer += (pos[i]*scale + center).flatten().tolist()
                        elif component == VertexLayout.VERTEX_COMPONENT_NORMAL:
                            # vertexBuffer += np.ravel(normal*[1, -1, 1]).tolist()
                            vertexBuffer += (normal[i]*[1, -1, 1]).flatten().tolist()
                            # vertexBuffer += normal[i].flatten().tolist()
                        elif component == VertexLayout.VERTEX_COMPONENT_UV:
                            # vertexBuffer += np.ravel(texCoord*uvscale).tolist()
                            vertexBuffer += (texCoord[i]*uvscale).flatten().tolist()
                        elif component == VertexLayout.VERTEX_COMPONENT_COLOR:
                            # vertexBuffer += np.ravel(color).tolist()
                            vertexBuffer += color[i].flatten().tolist()
                        elif component == VertexLayout.VERTEX_COMPONENT_TANGENT:
                            # vertexBuffer += np.ravel(tangent).tolist()
                            vertexBuffer += tangent[i].flatten().tolist()
                        elif component == VertexLayout.VERTEX_COMPONENT_BITANGENT:
                            # vertexBuffer += np.ravel(biTangent).tolist()
                            vertexBuffer += biTangent[i].flatten().tolist()
                        elif component == VertexLayout.VERTEX_COMPONENT_DUMMY_FLOAT:
                            vertexBuffer.append(0.0)
                        elif component == VertexLayout.VERTEX_COMPONENT_DUMMY_VEC4:
                            vertexBuffer += [0.0, 0.0, 0.0, 0.0]

                dimMax.append(np.amax(mesh.vertices, 0))
                dimMin.append(np.amin(mesh.vertices, 0))

                indexBase = len(indexBuffer)
                # indexBuffer += np.ravel(mesh.faces + indexBase).tolist()
                indexBuffer += (mesh.faces + indexBase).flatten().tolist()
                count = len(indexBuffer)# * 3
                # self.indexCount += len(mesh.faces)
                part.indexCount += count
                self.parts.append(part)

            self.dim.max = np.amax(dimMax, 0)
            self.dim.min = np.amin(dimMin, 0)
            self.dim.size = self.dim.max - self.dim.min

            self.indexCount = len(indexBuffer)

            vertexBuffer = np.array(vertexBuffer, np.float32)
            indexBuffer = np.array(indexBuffer, np.uint32)
            vBufferSize = vertexBuffer.nbytes
            iBufferSize = indexBuffer.nbytes

            # Use staging buffer to move vertex and index buffer to device local memory
            # Create staging buffers
            vertexStaging = vks.Buffer(self.device)
            indexStaging = vks.Buffer(self.device)

            # vertex buffer
            device.createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                vBufferSize,
                ffi.cast('float*', vertexBuffer.ctypes.data),
                vertexStaging
            )

            # index buffer
            device.createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                iBufferSize,
                ffi.cast('uint32_t*', indexBuffer.ctypes.data),
                indexStaging
            )

            # Create device local target buffers
            # vertex buffer
            device.createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                vBufferSize,
                None,
                self.vertices
            )

            # index buffer
            device.createBuffer(
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                iBufferSize,
                None,
                self.indices
            )

            # copy from staging buffers
            copyCmd = device.createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, True)

            copyRegion = VkBufferCopy(size=vBufferSize, srcOffset=0, dstOffset=0)
            vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, self.vertices.buffer, 1, [copyRegion])
            copyRegion.size = iBufferSize
            vkCmdCopyBuffer(copyCmd, indexStaging.buffer, self.indices.buffer, 1, [copyRegion])

            device.flushCommandBuffer(copyCmd, copyQueue)

            del vertexStaging, indexStaging
            return True
        return False