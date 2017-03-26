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


class VertexLayout(object):

    VERTEX_LAYOUT_POSITION = 0x0
    VERTEX_LAYOUT_NORMAL = 0x1
    VERTEX_LAYOUT_COLOR = 0x2
    VERTEX_LAYOUT_UV = 0x3
    VERTEX_LAYOUT_TANGENT = 0x4
    VERTEX_LAYOUT_BITANGENT = 0x5
    VERTEX_LAYOUT_DUMMY_FLOAT = 0x6
    VERTEX_LAYOUT_DUMMY_VEC4 = 0x7

class MeshBufferInfo(object):

    def __init__(self):
        self.buf = VK_NULL_HANDLE
        self.mem = VK_NULL_HANDLE
        self.size = 0

class MeshDescriptor(object):

    def __init__(self):
        self.vertexCount = None
        self.indexBase = None
        self.indexCount = None

class MeshBuffer(object):

    def __init__(self):
        self.meshDescriptors = []
        self.vertices = MeshBufferInfo()
        self.indices = MeshBufferInfo()
        self.indexCount = None
        self.dim = [0, 0, 0]

class MeshCreateInfo(object):

    def __init__(self):
        self.center = [0, 0, 0]
        self.scale = [0, 0, 0]
        self.uvscale = [0, 0, 0]


def vertexSize(layouts):
    vSize = 0
    for layoutDetail in layouts:
        if layoutDetail == VertexLayout.VERTEX_LAYOUT_UV:
            vSize += 2 * 4
        else:
            vSize += 3 * 4
    return vSize

def getVertexInputAttributeDescriptions(layouts, binding):
    offset = 0
    attributeDescriptions = []
    for i, layoutDetail in enumerate(layouts):
        if layoutDetail == VertexLayout.VERTEX_LAYOUT_UV:
            offset += 2 * 4
            inputAttribDescription = VkVertexInputAttributeDescription(
                binding=binding,
                location=i,
                offset=offset,
                format=VK_FORMAT_R32G32_SFLOAT
            )
        else:
            offset += 3 * 4
            inputAttribDescription = VkVertexInputAttributeDescription(
                binding=binding,
                location=i,
                offset=offset,
                format=VK_FORMAT_R32G32B32_SFLOAT
            )

        attributeDescriptions.append(inputAttribDescription)

    return attributeDescriptions


class Mesh(object):

    def __init__(self):
        self.buffers = None
        self.pipelineLayout = VK_NULL_HANDLE
        self.pipeline = VK_NULL_HANDLE
        self.descriptorSet = VK_NULL_HANDLE

        self.vertexBufferBinding = 0

        self.vertexInputState = None
        self.bindingDescription = None
        self.attributeDescriptions = None

    def setupVertexInputState(self, layouts):
        self.bindingDescription = VkVertexInputBindingDescription(
            binding=self.vertexBufferBinding,
            stride=vertexSize(layouts),
            inputRate=VK_VERTEX_INPUT_RATE_VERTEX
        )

        self.attributeDescriptions = []
        offset = 0
        for i, layoutDetail in enumerate(layouts):
            format = VK_FORMAT_R32G32_SFLOAT if layoutDetail == VertexLayout.VERTEX_LAYOUT_UV else VK_FORMAT_R32G32B32_SFLOAT
            vid = VkVertexInputAttributeDescription(
                location=self.vertexBufferBinding,
                binding=i,
                format=format,
                offset=offset
            )

            offset += 2 * 4 if layoutDetail == VertexLayout.VERTEX_LAYOUT_UV else 3 * 4

        self.vertexInputState = VkPipelineVertexInputStateCreateInfo(
            vertexBindingDescriptionCount=1,
            pVertexBindingDescriptions=[self.bindingDescription],
            vertexAttributeDescriptionCount=len(self.attributeDescriptions),
            pVertexAttributeDescriptions=self.attributeDescriptions
        )

    def drawIndexed(self, cmdBuffer):
        offsets = ffi.new('uint64_t[]', [0])
        if self.pipeline != VK_NULL_HANDLE:
            vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline)
        if self.pipelineLayout != VK_NULL_HANDLE and self.descriptorSet != VK_NULL_HANDLE:
            vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipelineLayout,
                                    0, 1, [self.descriptorSet], 0, ffi.NULL)
        vkCmdBindVertexBuffers(cmdBuffer, self.vertexBufferBinding, 1, [self.buffers.vertices.buf], offsets)
        vkCmdBindIndexBuffer(cmdBuffer, self.buffers.indices.buf, 0, VK_INDEX_TYPE_UINT32)
        vkCmdDrawIndexed(cmdBuffer, self.buffers.indexCount, 1, 0, 0, 0)

    @staticmethod
    def freeMeshBufferResources(device, meshBuffer):
        vkDestroyBuffer(device, meshBuffer)
        vkFreeMemory(device, meshBuffer.vertices.mem)
        if meshBuffer.indices.buf != VK_NULL_HANDLE:
            vkDestroyBuffer(device, meshBuffer.indices.buf)
            vkFreeMemory(device, meshBuffer.indices.mem)

class _Vertex(object):

    def __init__(self, pos, tex, normal, color, tangent, bitangent):
        self.pos = pos
        self.tex = tex
        self.normal = normal
        self.color = color
        self.tangent = tangent
        self.binormal = bitangent

class _MeshEntry(object):

    def __init__(self):
        self.numIndices = None
        self.materialIndex = None
        self.vertexBase = None
        self.vertices = None
        self.indices = []

class Dimension(object):
    FLT_MAX = 3.402823466e+38

    def __init__(self):
        self.min = [Dimension.FLT_MAX, Dimension.FLT_MAX, Dimension.FLT_MAX]
        self.max = [-Dimension.FLT_MAX, -Dimension.FLT_MAX, -Dimension.FLT_MAX]
        self.size = 0

# Simple mesh class for getting all the necessary stuff from models loaded via ASSIMP
class VulkanMeshLoader(object):

    def __init__(self, vulkanDevice):
        self.__vulkanDevice = vulkanDevice

        self.mEntries = []
        self.dim = Dimension()

        self.numVertices = 0

        self.scene = None

    def __del__(self):
        self.mEntries = []
        if self.scene:
            assimp.release(self.scene)

    @property
    def defaultFlags(self):
        return assimp.postprocess.aiProcess_FlipWindingOrder | assimp.postprocess.aiProcess_Triangulate | assimp.postprocess.aiProcess_PreTransformVertices | assimp.postprocess.aiProcess_CalcTangentSpace | assimp.postprocess.aiProcess_GenSmoothNormals

    def loadMesh(self, filename, flags=None):
        '''Load a scene from a supported 3D file format'''
        if flags is None:
            flags = VulkanMeshLoader.defaultFlags

        self.scene = assimp.load(filename, processing=flags)

        if self.scene:
            dimMax = []
            dimMin = []
            for i, mesh in enumerate(self.scene.meshes):
                entry = _MeshEntry()
                entry.vertexBase = self.numVertices
                self.numVertices += len(mesh.vertices)
                self.initMesh(entry, mesh)
                self.mEntries.append(entry)
                dimMax.append(np.amax(mesh.vertices, 0))
                dimMin.append(np.amin(mesh.vertices, 0))

            self.dim.max = np.amax(dimMax, 0)
            self.dim.min = np.amin(dimMin, 0)
            self.dim.size = self.dim.max - self.dim.min
            return True
        return False

    def initMesh(self, meshEntry, mesh):
        '''Read mesh data from ASSIMP mesh to an internal mesh representation that can be used to generate Vulkan buffers'''
        # meshEntry = _MeshEntry()
        meshEntry.materialIndex = mesh.materialindex
        diffuse = mesh.material.properties[('diffuse', 0)]

        numVertice = len(mesh.vertices)
        pos = mesh.vertices
        normal = mesh.normals
        texCoord = np.compress([True, True, False], mesh.texturecoords[0], axis=1) if mesh.texturecoords[0] else np.zeros((numVertice, 2*numVertice), mesh.vertices.dtype)
        tangent = mesh.tangents if mesh.tangents else np.zeros_like(mesh.vertices, mesh.vertices.dtype)
        biTangent = mesh.bitangents if mesh.bitangents else np.zeros_like(mesh.vertices, mesh.vertices.dtype)
        color = mesh.colors if mesh.colors else np.array(diffuse*numVertice, mesh.vertices.dtype)

        v = _Vertex(pos, texCoord, normal, color, tangent, biTangent)

        meshEntry.vertices = v

        indexBase = len(meshEntry.indices)
        meshEntry.indices = indexBase + mesh.faces

    def createBuffers(self, layouts, createInfo, useStaging, copyCmd, copyQueue):
        '''Create Vulkan buffers for the index and vertex buffer using a vertex layout'''
        meshBuffer = MeshBuffer()
        scale = [1.0, 1.0, 1.0]
        uvscale = [1.0, 1.0, 1.0]
        center = [0.0, 0.0, 0.0]
        if createInfo:
            scale = createInfo.scale
            uvscale = createInfo.uvscale
            center = createInfo.center

        vertexBuffer = []
        for entry in self.mEntries:
            # vertex data depending on layout
            for layout in layouts:
                # position
                if layout == VertexLayout.VERTEX_LAYOUT_POSITION:
                    vertexBuffer += np.ravel(center + scale * entry.vertices.pos)
                # normal
                if layout == VertexLayout.VERTEX_LAYOUT_NORMAL:
                    vertexBuffer += np.ravel([1, -1, 1] * entry.vertices.normal)
                # texture coordinates
                if layout == VertexLayout.VERTEX_LAYOUT_UV:
                    vertexBuffer += np.ravel(uvscale * entry.vertices.tex)
                # color
                if layout == VertexLayout.VERTEX_LAYOUT_COLOR:
                    vertexBuffer += np.ravel(entry.vertices.color)
                # tangent
                if layout == VertexLayout.VERTEX_LAYOUT_TANGENT:
                    vertexBuffer += np.ravel(entry.vertices.tangent)
                # bitangent
                if layout == VertexLayout.VERTEX_LAYOUT_BITANGENT:
                    vertexBuffer += np.ravel(entry.vertices.binormal)
                # dummy layout components for padding
                if layout == VertexLayout.VERTEX_LAYOUT_DUMMY_FLOAT:
                    vertexBuffer.append(0.0)
                if layout == VertexLayout.VERTEX_LAYOUT_DUMMY_VEC4:
                    vertexBuffer += [0.0, 0.0, 0.0, 0.0]

        vertexBuffer = np.array(vertexBuffer, np.float32)
        meshBuffer.vertices.size = vertexBuffer.nbytes

        self.dim.min *= scale
        self.dim.max *= scale
        self.dim.size *= scale

        indexBuffer = []
        for entry in self.mEntries:
            indexBase = len(indexBuffer)
            indexBuffer += np.ravel(entry.indices + indexBase)

            descriptor = MeshDescriptor()
            descriptor.indexBase = indexBase
            descriptor.indexCount = len(entry.indices)
            descriptor.vertexCount = len(entry.vertices)
            meshBuffer.meshDescriptors.append(descriptor)
        indexBuffer = np.array(indexBuffer, np.uint32)
        meshBuffer.indices.size = indexBuffer.nbytes
        meshBuffer.indexCount = len(indexBuffer)

        # Use staging buffer to move vertex and index buffer to device local memory
        if useStaging and copyQueue != VK_NULL_HANDLE and copyCmd != VK_NULL_HANDLE:
            # vertex buffer
            vertexStagingBuf, vertexStagingMem = self.__vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                meshBuffer.vertices.size,
                ffi.cast('float*', vertexBuffer.ctypes.data)
            )

            # index buffer
            indexStagingBuf, indexStagingMem = self.__vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                meshBuffer.indices.size,
                ffi.cast('uint32_t*', indexBuffer.ctypes.data)
            )

            # create device local target buffers
            # vertex buffer
            meshBuffer.vertices.buf, meshBuffer.vertices.mem = self.__vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                meshBuffer.vertices.size
            )

            # index buffer
            meshBuffer.indices.buf, meshBuffer.indices.mem = self.__vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                meshBuffer.indices.size
            )

            # copy from staging buffers
            cmdBufInfo = VkCommandBufferBeginInfo()
            vkBeginCommandBuffer(copyCmd, cmdBufInfo)

            copyRegion = VkBufferCopy(
                size=vertexBuffer.nbytes
            )
            vkCmdCopyBuffer(
                copyCmd,
                vertexStagingBuf,
                meshBuffer.vertices.buf,
                1,
                [copyRegion]
            )

            copyRegion.size = meshBuffer.indices.size
            vkCmdCopyBuffer(
                copyCmd,
                indexStagingBuf,
                meshBuffer.indices.buf,
                1,
                [copyRegion]
            )

            vkEndCommandBuffer(copyCmd)

            submitInfo = VkSubmitInfo(
                commandBufferCount=1,
                pCommandBuffers=[copyCmd]
            )

            vkQueueSubmit(copyQueue, 1, [submitInfo], VK_NULL_HANDLE)
            vkQueueWaitIdle(copyQueue)

            vkDestroyBuffer(self.__vulkanDevice.logicalDevice, vertexStagingBuf, None)
            vkFreeMemory(self.__vulkanDevice.logicalDevice, vertexStagingMem, None)
            vkDestroyBuffer(self.__vulkanDevice.logicalDevice, indexStagingBuf, None)
            vkFreeMemory(self.__vulkanDevice.logicalDevice, indexStagingMem, None)
        else:
            # generate vertex buffer
            meshBuffer.vertices.buf, meshBuffer.vertices.mem = self.__vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                meshBuffer.vertices.size,
                ffi.cast('float*', vertexBuffer.ctypes.data)
            )

            # generate index buffer
            meshBuffer.indices.buf, meshBuffer.indices.mem = self.__vulkanDevice.createBuffer(
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                meshBuffer.indices.size,
                ffi.cast('uint32_t*', indexBuffer.ctypes.data)
            )

        return meshBuffer



