from pyVulkan import *

import vulkantools as vkTools

class _QueueFamilyIndices(object):

    def __init__(self):
        self.graphics = 0
        self.compute = 0
        self.transfer = 0


class VulkanDevice(object):

    def __init__(self, physicalDevice):
        self.physicalDevice = physicalDevice
        self.logicalDevice = None
        self.memoryProperties = None
        self.queueFamilyProperties = None
        self.commandPool = VK_NULL_HANDLE
        self.enableDebugMarkers = False
        self.createFlags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        self.queueFamilyIndices = _QueueFamilyIndices()

        # Store Properties features, limits and properties of the physical device for later use
        # Device properties also contain limits and sparse properties
        self.properties = vkGetPhysicalDeviceProperties(self.physicalDevice)
        # Features should be checked by the examples before using them
        self.features = vkGetPhysicalDeviceFeatures(self.physicalDevice)
        # Memory properties are used regularly for creating all kinds of buffer
        self.memoryProperties = vkGetPhysicalDeviceMemoryProperties(self.physicalDevice)
        # Queue family properties, used for setting up requested queues upon device creation
        self.queueFamilyProperties = vkGetPhysicalDeviceQueueFamilyProperties(self.physicalDevice)

    def __del__(self):
        if self.commandPool:
            vkDestroyCommandPool(self.logicalDevice, self.commandPool, None)

        if self.logicalDevice:
            vkDestroyDevice(self.logicalDevice, None)

    # Get the index of a memory type that has all the requested property bits set
    #
    # @param typeBits Bitmask with bits set for each memory type supported by the resource to request for (from VkMemoryRequirements)
    # @param properties Bitmask of properties for the memory type to request
    # @param (Optional) memTypeFound Pointer to a bool that is set to true if a matching memory type has been found
    #
    # @return Index of the requested memory type
    def getMemoryType(self, typeBits, properties):
        for i, memType in enumerate(self.memoryProperties.memoryTypes):
            if (typeBits & 1) == 1:
                if (memType.propertyFlags & properties) == properties:
                    return i

            typeBits >>= 1

    # Get the index of a queue family that supports the requested queue flags
    #
    # @param queueFlags Queue flags to find a queue family index for
    #
    # @return Index of the queue family index that matches the flags
    def getQueueFamiliyIndex(self, queueFlags):
        # Dedicated queue for compute
        # Try to find a queue family index that supports compute but not graphics
        if queueFlags & VK_QUEUE_COMPUTE_BIT:
            for i, queueProp in enumerate(self.queueFamilyProperties):
                if queueProp.queueFlags & queueFlags and (queueProp.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0:
                    return i

        # Dedicated queue for transfer
        # Try to find a queue family index that supports transfer but not graphics and compute
        if queueFlags & VK_QUEUE_TRANSFER_BIT:
            for i, queueProp in enumerate(self.queueFamilyProperties):
                if queueProp.queueFlags & queueFlags and (queueProp.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0 and (queueProp.queueFlags & VK_QUEUE_TRANSFER_BIT) == 0:
                    return i

        # For other queue types or if no separate compute queue is present, return the first one to support the requested flags
        for i, queueProp in enumerate(self.queueFamilyProperties):
            if queueProp.queueFlags & queueFlags:
                return i

    # Create the logical device based on the assigned physical device, also gets default queue family indices
    #
    # @param enabledFeatures Can be used to enable certain features upon device creation
    # @param useSwapChain Set to false for headless rendering to omit the swapchain device extensions
    # @param requestedQueueTypes Bit flags specifying the queue types to be requested from the device
    def createLogicalDevice(self, enabledFeatures, useSwapChain=True, requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT):
        # Desired queues need to be requested upon logical device creation
        # Due to differing queue family configurations of Vulkan implementations this can be a bit tricky, especially if the application
        # requests different queue types
        queueCreateInfos = []

        # Get queue family indices for the requested queue family types
        # Note that the indices may overlap depending on the implementation
        defaultQueuePriority = 0.0

        # Graphics queue
        if requestedQueueTypes & VK_QUEUE_GRAPHICS_BIT:
            self.queueFamilyIndices.graphics = self.getQueueFamiliyIndex(VK_QUEUE_GRAPHICS_BIT)
            queueInfo = VkDeviceQueueCreateInfo(
                queueFamilyIndex=self.queueFamilyIndices.graphics,
                queueCount=1,
                pQueuePriorities=defaultQueuePriority
            )
            queueCreateInfos.append(queueInfo)
        else:
            self.queueFamilyIndices.graphics = VK_NULL_HANDLE

        # Dedicated compute queue
        if requestedQueueTypes & VK_QUEUE_COMPUTE_BIT:
            self.queueFamilyIndices.compute = self.getQueueFamiliyIndex(VK_QUEUE_COMPUTE_BIT)
            if self.queueFamilyIndices.compute != self.queueFamilyIndices.graphics:
                # If compute family index differs, we need an additional queue create info for the compute queue
                queueInfo = VkDeviceQueueCreateInfo(
                    queueFamilyIndex=self.queueFamilyIndices.compute,
                    queueCount=1,
                    pQueuePriorities=defaultQueuePriority
                )
                queueCreateInfos.append(queueInfo)
        else:
            # Else we use the same queue
            self.queueFamilyIndices.compute = self.queueFamilyIndices.graphics

        # Dedicated transfer queue
        if requestedQueueTypes & VK_QUEUE_TRANSFER_BIT:
            self.queueFamilyIndices.transfer = self.getQueueFamiliyIndex(VK_QUEUE_TRANSFER_BIT)
            if (self.queueFamilyIndices.transfer != self.queueFamilyIndices.graphics) and (self.queueFamilyIndices.transfer != self.queueFamilyIndices.compute):
                # If compute family index differs, we need an additional queue create info for the compute queue
                queueInfo = VkDeviceQueueCreateInfo(
                    queueFamilyIndex=self.queueFamilyIndices.transfer,
                    queueCount=1,
                    pQueuePriorities=defaultQueuePriority
                )
                queueCreateInfos.append(queueInfo)
        else:
            # Else we use the same queue
            self.queueFamilyIndices.transfer = self.queueFamilyIndices.graphics

        # Create the logical device representation
        deviceExtensions = []
        if useSwapChain:
            # If the device will be used for presenting to a display via a swapchain we need to request the swapchain extension
            deviceExtensions.append(VK_KHR_SWAPCHAIN_EXTENSION_NAME)

        deviceCreateInfo = VkDeviceCreateInfo(
            queueCreateInfoCount=len(queueCreateInfos),
            pQueueCreateInfos=queueCreateInfos,
            pEnabledFeatures=enabledFeatures
        )
        # Enable the debug marker extension if it is present (likely meaning a debugging tool is present)
        if vkTools.checkDeviceExtensionPresent(self.physicalDevice, VK_EXT_DEBUG_MARKER_EXTENSION_NAME):
            deviceExtensions.append(VK_EXT_DEBUG_MARKER_EXTENSION_NAME)
            self.enableDebugMarkers = True

        if len(deviceExtensions) > 0:
            devExtensions = [ffi.new('char[]', i) for i in deviceExtensions]
            deArray = ffi.new('char*[]', devExtensions)
            deviceCreateInfo.enabledExtensionCount = len(deviceExtensions)
            deviceCreateInfo.ppEnabledExtensionNames = deArray

        self.logicalDevice = vkCreateDevice(self.physicalDevice, deviceCreateInfo, None)
        # Create a default command pool for graphics command buffers
        self.commandPool = self.createCommandPool(self.queueFamilyIndices.graphics)
        return True

    # Create a buffer on the device
    #
    # @param usageFlags Usage flag bitmask for the buffer (i.e. index, vertex, uniform buffer)
    # @param memoryPropertyFlags Memory properties for this buffer (i.e. device local, host visible, coherent)
    # @param size Size of the buffer in byes
    # @param buffer Pointer to the buffer handle acquired by the function
    # @param memory Pointer to the memory handle acquired by the function
    # @param data Pointer to the data that should be copied to the buffer after creation (optional, if not set, no data is copied over)
    def createBuffer(self, usageFlags, memoryPropertyFlags, size, data=None):
        bufferCreateInfo = vkTools.initializers.bufferCreateInfo(usageFlags, size)
        buf = vkCreateBuffer(self.logicalDevice, bufferCreateInfo, None)

        # Create the memory backing up the buffer handle
        memAlloc = vkTools.initializers.memoryAllocateInfo()
        memReqs = vkGetBufferMemoryRequirements(self.logicalDevice, buf)
        memAlloc.allocationSize = memReqs.size
        # Find a memory type index that fits the properties of the buffer
        memAlloc.memoryTypeIndex = self.getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags)
        memory = vkAllocateMemory(self.logicalDevice, memAlloc, None)
        # If a pointer to the buffer data has been passed, map the buffer and copy over the data
        if data:
            mapped = vkMapMemory(self.logicalDevice, memory, 0, size, 0)
            ffi.memmove(mapped, data, size)
            vkUnmapMemory(self.logicalDevice, memory)

        # Attach the memory to the buffer object
        vkBindBufferMemory(self.logicalDevice, buf, memory, 0)

        return buf, memory

    # Copy buffer data from src to dst using VkCmdCopyBuffer
    #
    # @param src Pointer to the source buffer to copy from
    # @param dst Pointer to the destination buffer to copy tp
    # @param queue Pointer
    # @param copyRegion (Optional) Pointer to a copy region, if NULL, the whole buffer is copied
    def copyBuffer(self, src, dst, queue, copyRegion=None):
        pass

    # Create a command pool for allocation command buffers from
    #
    # @param queueFamilyIndex Family index of the queue to create the command pool for
    # @param createFlags (Optional) Command pool creation flags (Defaults to VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
    #
    # @note Command buffers allocated from the created pool can only be submitted to a queue with the same family index
    #
    # @return A handle to the created command buffer
    def createCommandPool(self, queueFamilyIndex, ):
        cmdPoolInfo = VkCommandPoolCreateInfo(
            queueFamilyIndex=queueFamilyIndex,
            flags=self.createFlags
        )

        return vkCreateCommandPool(self.logicalDevice, cmdPoolInfo, None)

    # Allocate a command buffer from the command pool
    #
    # @param level Level of the new command buffer (primary or secondary)
    # @param (Optional) begin If true, recording on the new command buffer will be started (vkBeginCommandBuffer) (Defaults to false)
    #
    # @return A handle to the allocated command buffer
    def createCommandBuffer(self, level, begin=False):
        cmdBufAllocateInfo = VkCommandBufferAllocateInfo(
            commandPool=self.commandPool,
            level=level,
            commandBufferCount=1
        )
        cmdbuffer = vkAllocateCommandBuffers(self.logicalDevice, cmdBufAllocateInfo)[0]

        # If requested, also start recording for the new command buffer
        if begin:
            cmdBufInfo = VkCommandBufferBeginInfo()
            vkBeginCommandBuffer(cmdbuffer, cmdBufInfo)

        return cmdbuffer

    # Finish command buffer recording and submit it to a queue
    #
    # @param commandBuffer Command buffer to flush
    # @param queue Queue to submit the command buffer to
    # @param free (Optional) Free the command buffer once it has been submitted (Defaults to true)
    #
    # @note The queue that the command buffer is submitted to must be from the same family index as the pool it was allocated from
    # @note Uses a fence to ensure command buffer has finished executing
    def flushCommandBuffer(self, commandBuffer, queue, free=True):
        if commandBuffer == VK_NULL_HANDLE:
            return

        vkEndCommandBuffer(commandBuffer)

        submitInfo = VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=commandBuffer
        )

        # Create fence to ensure that the command buffer has finished executing
        fenceInfo = VkFenceCreateInfo(flags=0)
        fence = vkCreateFence(self.logicalDevice, fenceInfo, None)

        # Submit to the queue
        vkQueueSubmit(queue, 1, submitInfo, fence)
        # Wait for the fence to signal that command buffer has finished executing
        vkWaitForFences(self.logicalDevice, 1, fence, VK_TRUE, 100000000000)

        vkDestroyFence(self.logicalDevice, fence, None)

        if free:
            vkFreeCommandBuffers(self.logicalDevice, self.commandPool, 1, [commandBuffer])