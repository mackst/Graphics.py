##########################################################################
# Vulkan Example - Basic indexed triangle rendering
#                             - port from Sascha Willems - www.saschawillems.de
#
# Copyright (C) 2017 by Shi Chi
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
##########################################################################


import logging

from PySide import (QtCore, QtGui)
from pyVulkan import *
import numpy as np

import glm
import camera
import vulkanExampleBase
import vulkantools

VERTEX_BUFFER_BIND_ID = 0
ENABLE_VALIDATION = False
USE_STAGING = True

UINT64_MAX = 0xffffffffffffffff

logger = logging.getLogger('triangle')
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


class Vertice(object):

    def __init__(self):
        self.memory = None
        self.buffer = None
        self.inputState = VkPipelineVertexInputStateCreateInfo()
        self.inputBinding = VkVertexInputBindingDescription()
        self.inputAttributes = []

class Indices(object):

    def __init__(self):
        self.memory = None
        self.buffer = None
        self.count = 0

class Uniform(object):

    def __init__(self):
        self.memory = None
        self.buffer = None
        self.descriptor = VkDescriptorBufferInfo()

class UBO_VS(object):

    def __init__(self):
        self.projectionMatrix = np.identity(4, np.float32)
        self.modelMatrix = np.identity(4, np.float32)
        self.viewMatrix = np.identity(4, np.float32)

    @property
    def nbytes(self):
        return self.modelMatrix.nbytes + self.viewMatrix.nbytes + self.projectionMatrix.nbytes

    @property
    def c_ptr(self):
        a = np.concatenate((self.modelMatrix, self.viewMatrix, self.projectionMatrix))

        return ffi.cast('float*', a.ctypes.data)


class VulkanExample(vulkanExampleBase.VulkanExampleBase):

    def __init__(self):
        super(VulkanExample, self).__init__(ENABLE_VALIDATION, None)
        self.vertices = Vertice()
        self.indices = Indices()
        self.uniformDataVS = Uniform()
        self.uboVS = UBO_VS()
        self.pipelineLayout = None
        self.pipeline = None
        self.descriptorSetLayout = None
        self.descriptorSet = None

        self.presentCompleteSemaphore = None
        self.renderCompleteSemaphore = None

        self.waitFences = []

        self.width = 1280
        self.height = 720
        self.zoom = -2.5
        self.title = "Vulkan Example - Basic indexed triangle"

    def __del__(self):

        if self.pipeline:
            vkDestroyPipeline(self._device, self.pipeline, None)

        if self.pipelineLayout:
            vkDestroyPipelineLayout(self._device, self.pipelineLayout, None)

        if self.descriptorSetLayout:
            vkDestroyDescriptorSetLayout(self._device, self.descriptorSetLayout, None)

        if self.vertices.buffer:
            vkDestroyBuffer(self._device, self.vertices.buffer, None)
            vkFreeMemory(self._device, self.vertices.memory, None)

        if self.indices.buffer:
            vkDestroyBuffer(self._device, self.indices.buffer, None)
            vkFreeMemory(self._device, self.indices.memory, None)

        if self.uniformDataVS.buffer:
            vkDestroyBuffer(self._device, self.uniformDataVS.buffer, None)
            vkFreeMemory(self._device, self.uniformDataVS.memory, None)

        if self.presentCompleteSemaphore:
            vkDestroySemaphore(self._device, self.presentCompleteSemaphore, None)

        if self.renderCompleteSemaphore:
            vkDestroySemaphore(self._device, self.renderCompleteSemaphore, None)

        [vkDestroyFence(self._device, fence, None) for fence in self.waitFences]

    def getMemoryTypeIndex(self, typeBits, properties):
        for i, memType in enumerate(self._deviceMemoryProperties.memoryTypes):
            if (typeBits & 1) == 1:
                if (memType.propertyFlags & properties) == properties:
                    return i

            typeBits >>= 1

        return 0

    def prepareSynchronizationPrimitives(self):
        logger.debug('begin prepareSynchronizationPrimitives')
        # Semaphores (Used for correct command ordering)
        semaphoreCreateInfo = VkSemaphoreCreateInfo()
        # Semaphore used to ensures that image presentation is complete before starting to submit again
        self.presentCompleteSemaphore = vkCreateSemaphore(self._device, semaphoreCreateInfo, None)
        # Semaphore used to ensures that all commands submitted have been finished before submitting the image to the queue
        self.renderCompleteSemaphore = vkCreateSemaphore(self._device, semaphoreCreateInfo, None)

        # Fences (Used to check draw command buffer completion)
        # Create in signaled state so we don't wait on first render of each command buffer
        fenceCreateInfo = VkFenceCreateInfo(flags=VK_FENCE_CREATE_SIGNALED_BIT)
        self.waitFences = [vkCreateFence(self._device, fenceCreateInfo, None) for dbuf in self._drawCmdBuffers]

        logger.debug('end of prepareSynchronizationPrimitives')

    def getCommandBuffer(self, begin):
        logger.debug('begin getCommandBuffer')
        cmdBuffer = None

        cmdBufAllocateInfo = VkCommandBufferAllocateInfo(
            commandPool=self._cmdPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        cmdBuffer = vkAllocateCommandBuffers(self._device, cmdBufAllocateInfo)[0]

        # If requested, also start the new command buffer
        if begin:
            cmdBufInfo = VkCommandBufferBeginInfo()
            vkBeginCommandBuffer(cmdBuffer, cmdBufInfo)

        logger.debug('end of getCommandBuffer')
        return cmdBuffer

    def flushCommandBuffer(self, commandBuffer):
        logger.debug('begin flushCommandBuffer')
        assert commandBuffer != VK_NULL_HANDLE

        vkEndCommandBuffer(commandBuffer)

        submitInfo = VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[commandBuffer]
        )

        # Create fence to ensure that the command buffer has finished executing
        fencCreateInfo = VkFenceCreateInfo(flags=0)
        fence = vkCreateFence(self._device, fencCreateInfo, None)

        # submit to the queue
        vkQueueSubmit(self._queue, 1, [submitInfo], fence)
        # Wait for the fence to signal that command buffer has finished executing
        vkWaitForFences(self._device, 1, [fence], VK_TRUE, vulkantools.DEFAULT_FENCE_TIMEOUT)

        vkDestroyFence(self._device, fence, None)
        vkFreeCommandBuffers(self._device, self._cmdPool, 1, [commandBuffer])
        logger.debug('end of flushCommandBuffer')

    def buildCommandBuffers(self):
        logger.debug('begin buildCommandBuffers')
        cmdBufInfo = VkCommandBufferBeginInfo()

        for i, drawBuf in enumerate(self._drawCmdBuffers):
            # Set clear values for all framebuffer attachments with loadOp set to clear
            # We use two attachments (color and depth) that are cleared at the start of the subpass and as such we need to set clear values for both
            # clearValues = [VkClearValue(color=[[0.0, 0.0, 0.2, 1.0]]),
            #                VkClearValue(depthStencil=[1.0, 0])]
            clearValues = ffi.new('VkClearValue[]', 2)
            clearValues[0].color = [[0.0, 0.0, 0.2, 1.0]]
            clearValues[1].depthStencil = [1.0, 0]

            renderArea = VkRect2D(offset=[0, 0], extent=[self.width, self.height])
            renderPassBeginInfo = VkRenderPassBeginInfo(
                renderPass=self._renderPass,
                renderArea=renderArea,
                clearValueCount=len(clearValues),
                pClearValues=clearValues,
                framebuffer=self._frameBuffers[i]
            )

            vkBeginCommandBuffer(drawBuf, cmdBufInfo)

            # Start the first sub pass specified in our default render pass setup by the base class
            # This will clear the color and depth attachment
            vkCmdBeginRenderPass(drawBuf, renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE)

            # Update dynamic viewport state
            viewport = VkViewport(
                height=float(self.height),
                width=float(self.width),
                minDepth=0.0,
                maxDepth=1.0
            )
            vkCmdSetViewport(drawBuf, 0, 1, [viewport])

            # Update dynamic scissor state
            scissor = VkRect2D([0, 0], [self.width, self.height])
            vkCmdSetScissor(drawBuf, 0, 1, [scissor])

            # Bind the rendering pipeline
            vkCmdBindPipeline(drawBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline)

            # The pipeline (state object) contains all states of the rendering pipeline, binding it will set all the states specified at pipeline creation time
            offsets = ffi.new('uint64_t[]', [0, ])
            vertexBuffers = ffi.new('VkBuffer[]', [self.vertices.buffer])
            vkCmdBindVertexBuffers(drawBuf, VERTEX_BUFFER_BIND_ID, 1, vertexBuffers, offsets)

            # Bind triangle index buffer
            vkCmdBindIndexBuffer(drawBuf, self.indices.buffer, 0, VK_INDEX_TYPE_UINT32)

            # Bind descriptor sets describing shader binding points
            vkCmdBindDescriptorSets(drawBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipelineLayout, 0, 1,
                                    [self.descriptorSet], 0, ffi.NULL)

            # Draw indexed triangle
            vkCmdDrawIndexed(drawBuf, self.indices.count, 1, 0, 0, 1)

            vkCmdEndRenderPass(drawBuf)

            # Ending the render pass will add an implicit barrier transitioning the frame buffer color attachment to
            # VK_IMAGE_LAYOUT_PRESENT_SRC_KHR for presenting it to the windowing system
            vkEndCommandBuffer(drawBuf)

        logger.debug('end of buildCommandBuffers')

    def draw(self):
        logger.debug('begin draw')
        # Get next image in the swap chain (back/front buffer)
        self._currentBuffer = self._swapChain.acquireNextImage(self.presentCompleteSemaphore)

        # Use a fence to wait until the command buffer has finished execution before using it again
        vkWaitForFences(self._device, 1, [self.waitFences[self._currentBuffer]], VK_TRUE, UINT64_MAX)
        vkResetFences(self._device, 1, [self.waitFences[self._currentBuffer]])

        # Pipeline stage at which the queue submission will wait (via pWaitSemaphores)
        waitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        waitSemaphores = ffi.new('VkSemaphore[]', [self.presentCompleteSemaphore])
        signalSemaphores = ffi.new('VkSemaphore[]', [self.renderCompleteSemaphore])
        # The submit info structure specifices a command buffer queue submission batch
        submitInfo = VkSubmitInfo(pWaitDstStageMask=waitStageMask,
                                  pWaitSemaphores=[self.presentCompleteSemaphore],
                                  waitSemaphoreCount=1,
                                  pSignalSemaphores=[self.renderCompleteSemaphore],
                                  signalSemaphoreCount=1,
                                  pCommandBuffers=[self._drawCmdBuffers[self._currentBuffer]],
                                  commandBufferCount=1)

        # Submit to the graphics queue passing a wait fence
        vkQueueSubmit(self._queue, 1, submitInfo, self.waitFences[self._currentBuffer])

        # Present the current buffer to the swap chain
        # Pass the semaphore signaled by the command buffer submission from the submit info as the wait semaphore for swap chain presentation
        # This ensures that the image is not presented to the windowing system until all commands have been submitted
        self._swapChain.queuePresent(self._queue, self._currentBuffer, self.renderCompleteSemaphore)


    def prepareVertices(self, useStagingBuffers=True):
        logger.debug('begin prepareVertices')
        '''Prepare vertex and index buffers for an indexed triangle
        Also uploads them to device local memory using staging and initializes vertex input and attribute binding to match the vertex shader
        '''
        # A note on memory management in Vulkan in general:
        # This is a very complex topic and while it's fine for an example application to to small individual memory allocations that is not
        # what should be done a real-world application, where you should allocate large chunkgs of memory at once isntead.

        # setup vertices
        vertexBuffer = np.array([[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                                 [[-1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                                 [[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]]], np.float32)

        # setup indices
        indexBuffer = np.array([0, 1, 2], np.uint32)
        self.indices.count = len(indexBuffer)

        memAlloc = VkMemoryAllocateInfo()
        memReqs = VkMemoryRequirements()

        if useStagingBuffers:
            # Static data like vertex and index buffer should be stored on the device memory
            # for optimal (and fastest) access by the GPU
            # To achieve this we use so-called "staging buffers" :
            # - Create a buffer that's visible to the host (and can be mapped)
            # - Copy the data to this buffer
            # - Create another buffer that's local on the device (VRAM) with the same size
            # - Copy the data from the host to the device using a command buffer
            # - Delete the host visible (staging) buffer
            # - Use the device local buffers for rendering

            class StagingBuffer(object):
                def __init__(self):
                    self.memory = None
                    self.buffer = None

            class StagingBuffers(object):
                def __init__(self):
                    self.vertices = StagingBuffer()
                    self.indices = StagingBuffer()

            stagingBuffers = StagingBuffers()

            # Vertex buffer
            vertexBufferInfo = VkBufferCreateInfo(
                size=vertexBuffer.nbytes,
                usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            )

            # Create a host-visible buffer to copy the vertex data to (staging buffer)
            stagingBuffers.vertices.buffer = vkCreateBuffer(self._device, vertexBufferInfo, None)
            memReqs = vkGetBufferMemoryRequirements(self._device, stagingBuffers.vertices.buffer)
            memAlloc.allocationSize = memReqs.size
            # Request a host visible memory type that can be used to copy our data do
            # Also request it to be coherent, so that writes are visible to the GPU right after unmapping the buffer
            memAlloc.memoryTypeIndex = self.getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            stagingBuffers.vertices.memory = vkAllocateMemory(self._device, memAlloc, None)
            # Map and copy
            data = vkMapMemory(self._device, stagingBuffers.vertices.memory, 0, memAlloc.allocationSize, 0)
            vertex_c_ptr = ffi.cast('float*', vertexBuffer.ctypes.data)
            ffi.memmove(data, vertex_c_ptr, vertexBuffer.nbytes)
            vkUnmapMemory(self._device, stagingBuffers.vertices.memory)
            vkBindBufferMemory(self._device, stagingBuffers.vertices.buffer, stagingBuffers.vertices.memory, 0)

            # Create a device local buffer to which the (host local) vertex data will be copied and which will be used for rendering
            vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            self.vertices.buffer = vkCreateBuffer(self._device, vertexBufferInfo, None)
            memReqs = vkGetBufferMemoryRequirements(self._device, self.vertices.buffer)
            memAlloc.allocationSize = memReqs.size
            memAlloc.memoryTypeIndex = self.getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            self.vertices.memory = vkAllocateMemory(self._device, memAlloc, None)
            vkBindBufferMemory(self._device, self.vertices.buffer, self.vertices.memory, 0)

            # index buffer
            indexbufferInfo = VkBufferCreateInfo(
                size=indexBuffer.nbytes,
                usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            )
            # Copy index data to a buffer visible to the host (staging buffer)
            stagingBuffers.indices.buffer = vkCreateBuffer(self._device, indexbufferInfo, None)
            memReqs = vkGetBufferMemoryRequirements(self._device, stagingBuffers.indices.buffer)
            memAlloc.allocationSize = memReqs.size
            memAlloc.memoryTypeIndex = self.getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            stagingBuffers.indices.memory = vkAllocateMemory(self._device, memAlloc, None)
            data = vkMapMemory(self._device, stagingBuffers.indices.memory, 0, indexBuffer.nbytes, 0)
            index_c_ptr = ffi.cast('uint32_t*', indexBuffer.ctypes.data)
            ffi.memmove(data, index_c_ptr, indexBuffer.nbytes)
            vkUnmapMemory(self._device, stagingBuffers.indices.memory)
            vkBindBufferMemory(self._device, stagingBuffers.indices.buffer, stagingBuffers.indices.memory, 0)

            # Create destination buffer with device only visibility
            indexbufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            self.indices.buffer = vkCreateBuffer(self._device, indexbufferInfo, None)
            memReqs = vkGetBufferMemoryRequirements(self._device, self.indices.buffer)
            memAlloc.allocationSize = memReqs.size
            memAlloc.memoryTypeIndex = self.getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            self.indices.memory = vkAllocateMemory(self._device, memAlloc, None)
            vkBindBufferMemory(self._device, self.indices.buffer, self.indices.memory, 0)

            cmdBufferBeginInfo = VkCommandBufferBeginInfo()

            # Buffer copies have to be submitted to a queue, so we need a command buffer for them
            # Note: Some devices offer a dedicated transfer queue (with only the transfer bit set) that may be faster when doing lots of copies
            copyCmd = self.getCommandBuffer(True)

            # Put buffer region copies into command buffer
            copyRegion = VkBufferCopy(
                size=vertexBuffer.nbytes
            )
            # Vertex buffer
            vkCmdCopyBuffer(copyCmd, stagingBuffers.vertices.buffer, self.vertices.buffer, 1, [copyRegion])
            # Index buffer
            copyRegion.size = indexBuffer.nbytes
            vkCmdCopyBuffer(copyCmd, stagingBuffers.indices.buffer, self.indices.buffer, 1, [copyRegion])

            # Flushing the command buffer will also submit it to the queue and uses a fence to ensure that all commands have been executed before returning
            self.flushCommandBuffer(copyCmd)

            # Destroy staging buffers
            # Note: Staging buffer must not be deleted before the copies have been submitted and executed
            vkDestroyBuffer(self._device, stagingBuffers.vertices.buffer, None)
            vkFreeMemory(self._device, stagingBuffers.vertices.memory, None)
            vkDestroyBuffer(self._device, stagingBuffers.indices.buffer, None)
            vkFreeMemory(self._device, stagingBuffers.indices.memory, None)
        else:
            # Don't use staging
            # Create host-visible buffers only and use these for rendering. This is not advised and will usually result in lower rendering performance

            # Vertex buffer
            vertexBufferInfo = VkBufferCreateInfo(
                size=vertexBuffer.nbytes,
                usage=VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
            )

            # Copy vertex data to a buffer visible to the host
            self.vertices.buffer = vkCreateBuffer(self._device, vertexBufferInfo, None)
            memReqs = vkGetBufferMemoryRequirements(self._device, self.vertices.buffer)
            memAlloc.allocationSize = memReqs.size
            memAlloc.memoryTypeIndex = self.getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            self.vertices.memory = vkAllocateMemory(self._device, memAlloc, None)
            data = vkMapMemory(self._device, self.vertices.memory, 0, memAlloc.allocationSize, 0)
            vertex_c_ptr = ffi.cast('float*', vertexBuffer.ctypes.data)
            ffi.memmove(data, vertex_c_ptr, vertexBuffer.nbytes)
            vkUnmapMemory(self._device, self.vertices.memory)
            vkBindBufferMemory(self._device, self.vertices.buffer, self.vertices.memory, 0)

            # Index buffer
            indexbufferInfo = VkBufferCreateInfo(
                size=indexBuffer.nbytes,
                usage=VK_BUFFER_USAGE_INDEX_BUFFER_BIT
            )

            # Copy index data to a buffer visible to the host
            self.indices.buffer = vkCreateBuffer(self._device, indexbufferInfo, None)
            memReqs = vkGetBufferMemoryRequirements(self._device, self.indices.buffer)
            memAlloc.allocationSize = memReqs.size
            memAlloc.memoryTypeIndex = self.getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            self.indices.memory = vkAllocateMemory(self._device, memAlloc, None)
            data = vkMapMemory(self._device, self.indices.memory, 0, indexBuffer.nbytes, 0)
            index_c_ptr = ffi.cast('uint32_t*', indexBuffer.ctypes.data)
            ffi.memmove(data, index_c_ptr, indexBuffer.nbytes)
            vkUnmapMemory(self._device, self.indices.memory)
            vkBindBufferMemory(self._device, self.indices.buffer, self.indices.memory, 0)

        # Vertex input binding
        self.vertices.inputBinding.binding = VERTEX_BUFFER_BIND_ID
        self.vertices.inputBinding.stride = 6 * vertexBuffer.itemsize
        self.vertices.inputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX

        # Inpute attribute binding describe shader attribute locations and memory layouts
        # These match the following shader layout (see triangle.vert):
        # layout (location = 0) in vec3 inPos;
        # layout (location = 1) in vec3 inColor;
        # Attribute location 0: Position
        inputAttributes = []
        inputAttribute = VkVertexInputAttributeDescription(
            binding=VERTEX_BUFFER_BIND_ID,
            location=0,
            format=VK_FORMAT_R32G32B32_SFLOAT,
            offset=0
        )
        inputAttributes.append(inputAttribute)
        # Attribute location 1: Color
        inputAttribute = VkVertexInputAttributeDescription(
            binding=VERTEX_BUFFER_BIND_ID,
            location=1,
            format=VK_FORMAT_R32G32B32_SFLOAT,
            offset=3 * vertexBuffer.itemsize
        )
        inputAttributes.append(inputAttribute)
        self.vertices.inputAttributes = ffi.new('VkVertexInputAttributeDescription[]', inputAttributes)

        # Assign to the vertex input state used for pipeline creation
        self.vertices.inputState.flags = 0
        self.vertices.inputState.vertexBindingDescriptionCount = 1
        self.vertices.inputState.pVertexBindingDescriptions = ffi.addressof(self.vertices.inputBinding)
        self.vertices.inputState.vertexAttributeDescriptionCount = len(self.vertices.inputAttributes)
        self.vertices.inputState.pVertexAttributeDescriptions = self.vertices.inputAttributes

        logger.debug('end of prepareVertices')

    def setupDescriptorPool(self):
        logger.debug('begin setupDescriptorPool')
        # We need to tell the API the number of max. requested descriptors per type
        typeCounts = [VkDescriptorPoolSize()]
        # This example only uses one descriptor type (uniform buffer) and only requests one descriptor of this type
        typeCounts[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        typeCounts[0].descriptorCount = 1
        # For additional types you need to add new entries in the type count list
        #E.g. for two combined image samplers :
        # typeCounts[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
        # typeCounts[1].descriptorCount = 2

        # Create the global descriptor pool
        # All descriptors used in this example are allocated from this pool
        descriptorPoolInfo = VkDescriptorPoolCreateInfo(
            poolSizeCount=1,
            pPoolSizes=typeCounts,
            maxSets=1
        )

        self._descriptorPool = vkCreateDescriptorPool(self._device, descriptorPoolInfo, None)
        logger.debug('end of setupDescriptorPool')

    def setupDescriptorSetLayout(self):
        logger.debug('begin setupDescriptorSetLayout')
        # Setup layout of descriptors used in this example
        # Basically connects the different shader stages to descriptors for binding uniform buffers, image samplers, etc.
        # So every shader binding should map to one descriptor set layout binding

        # Binding 0: Uniform buffer (Vertex shader)
        layoutBinding = VkDescriptorSetLayoutBinding(
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=VK_SHADER_STAGE_VERTEX_BIT
        )

        descriptorLayout = VkDescriptorSetLayoutCreateInfo(
            bindingCount=1,
            pBindings=[layoutBinding]
        )

        self.descriptorSetLayout = vkCreateDescriptorSetLayout(self._device, descriptorLayout, None)

        # Create the pipeline layout that is used to generate the rendering pipelines that are based on this descriptor set layout
        # In a more complex scenario you would have different pipeline layouts for different descriptor set layouts that could be reused
        pPipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo(
            setLayoutCount=1,
            pSetLayouts=[self.descriptorSetLayout]
        )
        self.pipelineLayout = vkCreatePipelineLayout(self._device, pPipelineLayoutCreateInfo, None)
        logger.debug('end of setupDescriptorSetLayout')

    def setupDescriptorSet(self):
        logger.debug('begin setupDescriptorSet')
        # Allocate a new descriptor set from the global descriptor pool
        allocInfo = VkDescriptorSetAllocateInfo(
            descriptorPool=self._descriptorPool,
            descriptorSetCount=1,
            pSetLayouts=[self.descriptorSetLayout]
        )

        self.descriptorSet = vkAllocateDescriptorSets(self._device, allocInfo)[0]

        # Update the descriptor set determining the shader binding points
        # For every binding point used in a shader there needs to be one
        # descriptor set matching that binding point
        writeDescriptorSet = VkWriteDescriptorSet(
            dstSet=self.descriptorSet,
            descriptorCount=1,
            descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            pBufferInfo=self.uniformDataVS.descriptor,
            dstBinding=0
        )

        vkUpdateDescriptorSets(self._device, 1, [writeDescriptorSet], 0, ffi.NULL)
        logger.debug('end of setupDescriptorSet')

    def setupDepthStencil(self):
        logger.debug('begin setupDepthStencil')
        '''Create the depth (and stencil) buffer attachments used by our framebuffers
        Note: Override of virtual function in the base class and called from within VulkanExampleBase.prepare'''
        # Create an optimal image used as the depth stencil attachment
        image = VkImageCreateInfo(
            imageType=VK_IMAGE_TYPE_2D,
            format=self._depthFormat,
            extent=[self.width, self.height, 1],
            mipLevels=1,
            arrayLayers=1,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=VK_IMAGE_TILING_OPTIMAL,
            usage=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED
        )
        self.depthStencil.image = vkCreateImage(self._device, image, None)

        # Allocate memory for the image (device local) and bind it to our image
        memAlloc = VkMemoryAllocateInfo()
        memReqs = vkGetImageMemoryRequirements(self._device, self.depthStencil.image)
        memAlloc.allocationSize = memReqs.size
        memAlloc.memoryTypeIndex = self.getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.depthStencil.mem = vkAllocateMemory(self._device, memAlloc, None)
        vkBindImageMemory(self._device, self.depthStencil.image, self.depthStencil.mem, 0)

        # Create a view for the depth stencil image
        # Images aren't directly accessed in Vulkan, but rather through views described by a subresource range
        # This allows for multiple views of one image with differing ranges (e.g. for different layers)
        depthStencilView = VkImageViewCreateInfo(
            viewType=VK_IMAGE_VIEW_TYPE_2D,
            format=self._depthFormat,
            subresourceRange=[VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1],
            image=self.depthStencil.image
        )
        self.depthStencil.view = vkCreateImageView(self._device, depthStencilView, None)
        logger.debug('end of setupDepthStencil')

    def setupFrameBuffer(self):
        logger.debug('begin setupFrameBuffer')
        '''Create a frame buffer for each swap chain image'''
        # Create a frame buffer for every image in the swapchain
        self._frameBuffers = []
        for buf in self._swapChain.buffers:
            attachments = [buf.view, self.depthStencil.view]

            frameBufferCreateInfo = VkFramebufferCreateInfo(
                renderPass=self._renderPass,
                attachmentCount=len(attachments),
                pAttachments=attachments,
                width=self.width,
                height=self.height,
                layers=1
            )
            # Create the framebuffer
            frameBuf = vkCreateFramebuffer(self._device, frameBufferCreateInfo, None)
            self._frameBuffers.append(frameBuf)

        logger.debug('end of setupFrameBuffer')

    def setupRenderPass(self):
        '''Render pass setup
        Render passes are a new concept in Vulkan. They describe the attachments used during rendering and may contain multiple subpasses with attachment dependencies
        This allows the driver to know up-front what the rendering will look like and is a good opportunity to optimize especially on tile-based renderers (with multiple subpasses)
        Using sub pass dependencies also adds implicit layout transitions for the attachment used, so we don't need to add explicit image memory barriers to transform them'''
        logger.debug('begin setupRenderPass')
        # This example will use a single render pass with one subpass

        # Descriptors for the attachments used by this renderpass
        attachments = [VkAttachmentDescription(), VkAttachmentDescription()]

        # Color attachment
        attachments[0].format = self._colorformat
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR

        # Depth attachment
        attachments[1].format = self._depthFormat
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL

        # Setup attachment references
        colorReference = VkAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        depthReference = VkAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)

        # Setup a single subpass reference
        subpassDescription = VkSubpassDescription(
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[colorReference],
            pDepthStencilAttachment=[depthReference],
            inputAttachmentCount=0,
            preserveAttachmentCount=0
        )

        # Setup subpass dependencies
        # These will add the implicit ttachment layout transitionss specified by the attachment descriptions
        # The actual usage layout is preserved through the layout specified in the attachment reference
        # Each subpass dependency will introduce a memory and execution dependency between the source and dest subpass described by
        # srcStageMask, dstStageMask, srcAccessMask, dstAccessMask (and dependencyFlags is set)
        # Note: VK_SUBPASS_EXTERNAL is a special constant that refers to all commands executed outside of the actual renderpass)
        dependencies = [VkSubpassDependency(), VkSubpassDependency()]

        # First dependency at the start of the renderpass
        # Does the transition from final to initial layout
        dependencies[0].srcSubpass = ffi.cast('uint32_t', VK_SUBPASS_EXTERNAL)
        dependencies[0].dstSubpass = 0
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT

        # Second dependency at the end the renderpass
        # Does the transition from the initial to the final layout
        dependencies[0].srcSubpass = 0
        dependencies[0].dstSubpass = ffi.cast('uint32_t', VK_SUBPASS_EXTERNAL)
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
        dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        dependencies[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT

        # Create the actual renderpass
        renderPassInfo = VkRenderPassCreateInfo(
            attachmentCount=len(attachments),
            pAttachments=attachments,
            subpassCount=1,
            pSubpasses=[subpassDescription],
            dependencyCount=len(dependencies),
            pDependencies=dependencies
        )
        self._renderPass = vkCreateRenderPass(self._device, renderPassInfo, None)
        logger.debug('end of setupRenderPass')

    def preparePipelines(self):
        logger.debug('begin preparePipelines')
        # Create the graphics pipeline used in this example
        # Vulkan uses the concept of rendering pipelines to encapsulate fixed states, replacing OpenGL's complex state machine
        # A pipeline is then stored and hashed on the GPU making pipeline changes very fast
        # Note: There are still a few dynamic states that are not directly part of the pipeline (but the info that they are used is)

        # Construct the differnent states making up the pipeline

        # Input assembly state describes how primitives are assembled
        # This pipeline will assemble vertex data as a triangle lists (though we only use one triangle)
        inputAssemblyState = VkPipelineInputAssemblyStateCreateInfo(
            topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        )
        # Rasterization state
        rasterizationState = VkPipelineRasterizationStateCreateInfo(
            polygonMode=VK_POLYGON_MODE_FILL,
            cullMode=VK_CULL_MODE_NONE,
            frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depthClampEnable=VK_FALSE,
            rasterizerDiscardEnable=VK_FALSE,
            lineWidth=1.0
        )

        # Color blend state describes how blend factors are calculated (if used)
        # We need one blend attachment state per color attachment (even if blending is not used
        blendAttachmentState = VkPipelineColorBlendAttachmentState(
            colorWriteMask=0xf,
            blendEnable=VK_FALSE
        )
        colorBlendState = VkPipelineColorBlendStateCreateInfo(
            attachmentCount=1,
            pAttachments=[blendAttachmentState]
        )

        # Viewport state sets the number of viewports and scissor used in this pipeline
        # Note: This is actually overriden by the dynamic states (see below)
        viewportState = VkPipelineViewportStateCreateInfo(
            viewportCount=1,
            scissorCount=1
        )

        # Enable dynamic states
        # Most states are baked into the pipeline, but there are still a few dynamic states that can be changed within a command buffer
        # To be able to change these we need do specify which dynamic states will be changed using this pipeline. Their actual states are set later on in the command buffer.
        # For this example we will set the viewport and scissor using dynamic states
        dynamicStateEnables = [VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR]
        dynamicState = VkPipelineDynamicStateCreateInfo(
            pDynamicStates=dynamicStateEnables,
            dynamicStateCount=len(dynamicStateEnables)
        )

        # Depth and stencil state containing depth and stencil compare and test operations
        # We only use depth tests and want depth tests and writes to be enabled and compare with less or equal
        depthStencilState = VkPipelineDepthStencilStateCreateInfo(
            depthTestEnable=VK_TRUE,
            depthWriteEnable=VK_TRUE,
            depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL,
            depthBoundsTestEnable=VK_FALSE,
            back=VkStencilOpState(failOp=VK_STENCIL_OP_KEEP,
                                  passOp=VK_STENCIL_OP_KEEP,
                                  compareOp=VK_COMPARE_OP_ALWAYS),
            stencilTestEnable=VK_FALSE,
            front=VkStencilOpState(failOp=VK_STENCIL_OP_KEEP,
                                  passOp=VK_STENCIL_OP_KEEP,
                                  compareOp=VK_COMPARE_OP_ALWAYS)
        )

        # Multi sampling state
        # This example does not make use fo multi sampling (for anti-aliasing), the state must still be set and passed to the pipeline
        multisampleState = VkPipelineMultisampleStateCreateInfo(
            rasterizationSamples=VK_SAMPLE_COUNT_1_BIT
        )

        # Load shaders
        # Vulkan loads it's shaders from an immediate binary representation called SPIR-V
        # Shaders are compiled offline from e.g. GLSL using the reference glslang compiler
        ssv, vertShader = self.loadShader('shaders/triangle.vert.spv', VK_SHADER_STAGE_VERTEX_BIT)
        ssf, fragShader = self.loadShader('shaders/triangle.frag.spv', VK_SHADER_STAGE_FRAGMENT_BIT)
        shaderStages = [ssv, ssf]
        self._shaderModules.append(vertShader)
        self._shaderModules.append(fragShader)

        pipelineCreateInfo = VkGraphicsPipelineCreateInfo(
            layout=self.pipelineLayout,
            renderPass=self._renderPass,
            stageCount=len(shaderStages),
            pStages=shaderStages,
            pVertexInputState=self.vertices.inputState,
            pInputAssemblyState=inputAssemblyState,
            pRasterizationState=rasterizationState,
            pColorBlendState=colorBlendState,
            pMultisampleState=multisampleState,
            pViewportState=viewportState,
            pDepthStencilState=depthStencilState,
            pDynamicState=dynamicState,
            subpass=0,
            basePipelineHandle=VK_NULL_HANDLE
        )
        # Create rendering pipeline using the specified states
        self.pipeline = vkCreateGraphicsPipelines(self._device, self._pipelineCache, 1, pipelineCreateInfo, ffi.NULL)[0]
        # self.pipeline = vkCreateGraphicsPipelines(self._device, VK_NULL_HANDLE, 1, pipelineCreateInfo, ffi.NULL)[0]
        # vkDestroyShaderModule(self._device, vertShader, ffi.NULL)
        # vkDestroyShaderModule(self._device, fragShader, ffi.NULL)
        logger.debug('end of preparePipelines')

    def prepareUniformBuffers(self):
        logger.debug('begin prepareUniformBuffers')
        # Prepare and initialize a uniform buffer block containing shader uniforms
        # Single uniforms like in OpenGL are no longer present in Vulkan. All Shader uniforms are passed via uniform buffer blocks

        # Vertex shader uniform buffer block
        allocInfo = VkMemoryAllocateInfo(
            allocationSize=0,
            memoryTypeIndex=0
        )

        bufferInfo = VkBufferCreateInfo(
            size=self.uboVS.nbytes,
            usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
        )
        # Create a new buffer
        self.uniformDataVS.buffer = vkCreateBuffer(self._device, bufferInfo, None)
        # Get memory requirements including size, alignment and memory type
        memReqs = vkGetBufferMemoryRequirements(self._device, self.uniformDataVS.buffer)
        allocInfo.allocationSize = memReqs.size
        # Get the memory type index that supports host visibile memory access
        # Most implementations offer multiple memory types and selecting the correct one to allocate memory from is crucial
        # We also want the buffer to be host coherent so we don't have to flush (or sync after every update.
        # Note: This may affect performance so you might not want to do this in a real world application that updates buffers on a regular base
        allocInfo.memoryTypeIndex = self.getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        # Allocate memory for the uniform buffer
        self.uniformDataVS.memory = vkAllocateMemory(self._device, allocInfo, None)
        # Bind memory to buffer
        vkBindBufferMemory(self._device, self.uniformDataVS.buffer, self.uniformDataVS.memory, 0)

        # Store information in the uniform's descriptor that is used by the descriptor set
        self.uniformDataVS.descriptor.buffer = self.uniformDataVS.buffer
        self.uniformDataVS.descriptor.offset = 0
        self.uniformDataVS.descriptor.range = self.uboVS.nbytes

        self.updateUniformBuffers()
        logger.debug('end of prepareUniformBuffers')

    def updateUniformBuffers(self):
        logger.debug('begin updateUniformBuffers')
        # Update matrices
        self.uboVS.projectionMatrix = glm.perspective(60.0, float(self.width) / self.height, 0.1, 256.0)

        self.uboVS.viewMatrix = glm.translate(self.uboVS.viewMatrix, 0.0, 0.0, self.zoom)

        self.uboVS.modelMatrix = glm.rotate(self.uboVS.modelMatrix, self.rotation[0], 1.0, 0.0, 0.0)
        self.uboVS.modelMatrix = glm.rotate(self.uboVS.modelMatrix, self.rotation[1], 0.0, 1.0, 0.0)
        self.uboVS.modelMatrix = glm.rotate(self.uboVS.modelMatrix, self.rotation[2], 0.0, 0.0, 1.0)

        # Map uniform buffer and update it
        data = vkMapMemory(self._device, self.uniformDataVS.memory, 0, self.uboVS.nbytes, 0)
        ffi.memmove(data, self.uboVS.c_ptr, self.uboVS.nbytes)
        # Unmap after data has been copied
        # Note: Since we requested a host coherent memory type for the uniform buffer, the write is instantly visible to the GPU
        vkUnmapMemory(self._device, self.uniformDataVS.memory)
        logger.debug('end of updateUniformBuffers')

    def prepare(self):
        super(VulkanExample, self).prepare()
        logger.debug('prepareSynchronizationPrimitives')
        self.prepareSynchronizationPrimitives()
        self.prepareVertices(USE_STAGING)
        self.prepareUniformBuffers()
        self.setupDescriptorSetLayout()
        self.preparePipelines()
        self.setupDescriptorPool()
        self.setupDescriptorSet()
        self.buildCommandBuffers()
        self.prepared = True

    def render(self):
        if not self.prepared:
            return
        self.draw()

    def viewChanged(self):
        # This function is called by the base example class each time the view is changed by user input
        self.updateUniformBuffers()

if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    vulkanExample = VulkanExample()
    logger.debug('setup window')
    vulkanExample.setupWindow()
    # vulkanExample.show()
    logger.debug('init swapchain')
    vulkanExample.initSwapchain()
    logger.debug('prepare')
    vulkanExample.prepare()
    vulkanExample.renderLoop()

    sys.exit(app.exec_())