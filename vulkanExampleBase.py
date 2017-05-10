########################################################################################
# Vulkan Example base class port from Sascha Willems - www.saschawillems.de
#
# Copyright (C) 2017 by Shi Chi
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
########################################################################################


import time

from PyQt5 import QtGui, QtCore
from pyVulkan import *

import vulkandebug as vkDebug
import vulkantools as vkTools

from vulkandevice import VulkanDevice
from vulkanswapchain import VulkanSwapChain
from vulkantextoverlay import VulkanTextOverlay


class Semaphore(object):

    def __init__(self):
        # Swap chain image presentation
        self.presentComplete = None
        # Command buffer submission and execution
        self.renderComplete = None
        # Text overlay submission and execution
        self.textOverlayComplete = None

class DepthStencil(object):

    def __init__(self):
        self.image = None
        self.mem = None
        self.view = None

class VulkanExampleBase(QtGui.QWindow):

    def __init__(self, enableValidation=False, enableVSync=False, screen=None):
        super(VulkanExampleBase, self).__init__(screen)

        # OpenGL surface
        self.setSurfaceType(self.OpenGLSurface)

        self.prepared = False
        self.winWidth = 1280
        self.winHeight = 720

        self.defaultClearColor = VkClearColorValue([0.025, 0.025, 0.025, 1.0])

        self.zoom = 0

        # Defines a frame rate independent timer value clamped from -1.0...1.0
        # For use in animations, rotations, etc.
        self.timer = 0.0
        # Multiplier for speeding up (or slowing down) the global timer
        self.timerSpeed = 0.25

        self.paused = False

        self.enableTextOverlay = False
        self.textOverlay = None

        # Use to adjust mouse rotation speed
        self.rotationSpeed = 1.0
        # Use to adjust mouse zoom speed
        self.zoomSpeed = 1.0

        self.camera = None

        self.rotation = [0.0, 0.0, 0.0]
        self.cameraPos = [0.0, 0.0, 0.0]
        self.mousePos = [0.0, 0.0]

        self.winTitle = 'pyVulkan Example'
        self.name = b'pyVulkanExample'

        self.depthStencil = DepthStencil()

        self.gamePadState = None

        # Last frame time, measured using a high performance timer (if available)
        self._frameTimer = 1.0
        # Frame counter to display fps
        self._frameCounter = 0
        self._lastFPS = 0

        # Vulkan instance, stores all per-application states
        self.instance = None
        # Physical device (GPU) that Vulkan will ise
        self.physicalDevice = None
        # Stores physical device properties (for e.g. checking device limits)
        self.deviceProperties = None
        # Stores phyiscal device features (for e.g. checking if a feature is available)
        self.deviceFeatures = None
        # Stores all available memory (type) properties for the physical device
        self.deviceMemoryProperties = None
        # @brief Logical device, application's view of the physical device (GPU)
        self.device = None
        # @brief Encapsulated physical and logical vulkan device
        self.vulkanDevice = None
        # Handle to the device graphics queue that command buffers are submitted to
        self.queue = None
        # Color buffer format
        self.colorformat = VK_FORMAT_R8G8B8A8_UNORM
        # Depth buffer format
        # Depth format is selected during Vulkan initialization
        self.depthFormat = None
        # Command buffer pool
        self.cmdPool = None
        # Command buffer used for setup
        self.setupCmdBuffer = VK_NULL_HANDLE
        # @brief Pipeline stages used to wait at for graphics queue submissions
        self.submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        # Contains command buffers and semaphores to be presented to the queue
        self.submitInfo = None
        # Command buffers used for rendering
        self.drawCmdBuffers = []
        # Global render pass for frame buffer writes
        self.renderPass = None
        # List of available frame buffers (same as number of swap chain images)
        self.frameBuffers = []
        # Active frame buffer index
        self.currentBuffer = 0
        # Descriptor set pool
        self.descriptorPool = VK_NULL_HANDLE
        # List of shader modules created (stored for cleanup)
        self.shaderModules = []
        # Pipeline cache object
        self.pipelineCache = None
        # Wraps the swap chain to present images (framebuffers) to the windowing system
        self.swapChain = VulkanSwapChain()
        # Synchronization semaphores
        self.semaphores = Semaphore()
        # Simple texture loader
        self.textureLoader = None

        # Set to true when example is created with enabled validation layers
        self.__enableValidation = enableValidation
        # Set to true if v-sync will be forced for the swapchain
        self.__enableVSync = enableVSync
        # fps timer (one second interval)
        self.__fpsTimer = 0.0
        # brief Indicates that the view (position, rotation) has changed and
        self.__viewUpdated = False
        # Destination dimensions for resizing the window
        self.__destWidth = 0
        self.__destHeight = 0
        self.__resizing = False

        self.resize(self.winWidth, self.winHeight)
        self.setTitle(self.name.decode())

        self.__timer = QtCore.QTimer(self)
        # self.__timer.setInterval(1)
        self.__timer.timeout.connect(self.renderLoop)

    def __del__(self):
        self.__timer.stop()

        del self.swapChain
        if self.descriptorPool != VK_NULL_HANDLE:
            vkDestroyDescriptorPool(self.device, self.descriptorPool, None)
        if self.setupCmdBuffer != VK_NULL_HANDLE:
            vkFreeCommandBuffers(self.device, self.cmdPool, 1, [self.setupCmdBuffer])

        # self.destroyCommandBuffers()
        if self.renderPass:
            vkDestroyRenderPass(self.device, self.renderPass, None)
        for i in self.frameBuffers:
            vkDestroyFramebuffer(self.device, i, None)

        for i in self.shaderModules:
            vkDestroyShaderModule(self.device, i, None)

        if self.depthStencil.view:
            vkDestroyImageView(self.device, self.depthStencil.view, None)
        if self.depthStencil.image:
            vkDestroyImage(self.device, self.depthStencil.image, None)
        if self.depthStencil.mem:
            vkFreeMemory(self.device, self.depthStencil.mem, None)

        if self.pipelineCache:
            vkDestroyPipelineCache(self.device, self.pipelineCache, None)

        if self.textureLoader:
            del self.textureLoader

        if self.cmdPool:
            vkDestroyCommandPool(self.device, self.cmdPool, None)

        if self.semaphores.presentComplete:
            vkDestroySemaphore(self.device, self.semaphores.presentComplete, None)
        if self.semaphores.renderComplete:
            vkDestroySemaphore(self.device, self.semaphores.renderComplete, None)
        if self.semaphores.textOverlayComplete:
            vkDestroySemaphore(self.device, self.semaphores.textOverlayComplete, None)

        if self.enableTextOverlay:
            del self.textOverlay

        del self.vulkanDevice

        if self.__enableValidation:
            vkDebug.freeDebugCallback(self.instance)

        if self.instance:
            vkDestroyInstance(self.instance, None)

    @property
    def enabledFeatures(self):
        # Device features enabled by the example
        # If not set, no additional features are enabled (may result in validation layer errors)
        return VkPhysicalDeviceFeatures()

    @property
    def assetPath(self):
        # Returns the base asset path (for shaders, models, textures) depending on the os
        return './../data/'

    @property
    def windowTitle(self):
        deviceName = ffi.string(self.deviceProperties.deviceName).decode()
        return '{} - {} - {} fps'.format(self.winTitle, deviceName, self._frameCounter)

    # create application wide Vulkan instance
    def __createInstance(self):
        appInfo = VkApplicationInfo(
            pApplicationName=self.name,
            pEngineName=self.name,
            apiVersion=VK_API_VERSION_1_0
        )

        enabledExtensions = [VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME]

        instanceCreateInfo = VkInstanceCreateInfo(
            pApplicationInfo=appInfo
        )

        if enabledExtensions:
            if self.__enableValidation:
                enabledExtensions.append(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)
            ext = [ffi.new('char[]', i.encode()) for i in enabledExtensions]
            extArray = ffi.new('char*[]', ext)
            instanceCreateInfo.enabledExtensionCount = len(enabledExtensions)
            instanceCreateInfo.ppEnabledExtensionNames = extArray
        if self.__enableValidation:
            layers = [ffi.new('char[]', i.encode()) for i in vkDebug.validationLayerNames]
            layerArray = ffi.new('char*[]', layers)
            instanceCreateInfo.enabledLayerCount = len(vkDebug.validationLayerNames)
            instanceCreateInfo.ppEnabledLayerNames = layerArray
        self.instance = vkCreateInstance(instanceCreateInfo, None)

        return True

    def initVulkan(self):
        '''Setup the vulkan instance, enable required extensions and connect to the physical device (GPU)'''
        # Vulkan instance
        self.__createInstance()

        # If requested, we enable the default validation layers for debugging
        if self.__enableValidation:
            # The report flags determine what type of messages for the layers will be displayed
            # For validating (debugging) an appplication the error and warning bits should suffice
            debugReportFlags = VK_DEBUG_REPORT_ERROR_BIT_EXT
            # Additional flags include performance info, loader and layer debug messages, etc.
            vkDebug.setupDebugging(self.instance, debugReportFlags)

        physicalDevices = vkEnumeratePhysicalDevices(self.instance)

        # Note :
        # This example will always use the first physical device reported,
        # change the vector index if you have multiple Vulkan devices installed
        # and want to use another one
        self.physicalDevice = physicalDevices[0]

        # Store properties (including limits) and features of the phyiscal device
        # So examples can check against them and see if a feature is actually supported
        self.deviceProperties = vkGetPhysicalDeviceProperties(self.physicalDevice)
        self.deviceFeatures = vkGetPhysicalDeviceFeatures(self.physicalDevice)
        # Gather physical device memory properties
        self.deviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties(self.physicalDevice)

        # Vulkan device creation
        # This is handled by a separate class that gets a logical device representation
        # and encapsulates functions related to a device
        self.vulkanDevice = VulkanDevice(self.physicalDevice)
        self.vulkanDevice.createLogicalDevice(self.enabledFeatures)
        self.device = self.vulkanDevice.logicalDevice

        # Get a graphics queue from the device
        self.queue = vkGetDeviceQueue(self.device, self.vulkanDevice.queueFamilyIndices.graphics, 0)

        # Find a suitable depth format
        self.depthFormat = vkTools.getSupportedDepthFormat(self.physicalDevice)

        self.swapChain.connect(self.instance, self.physicalDevice, self.device)

        # Create synchronization objects
        semaphoreCreateInfo = VkSemaphoreCreateInfo()
        # Create a semaphore used to synchronize image presentation
        # Ensures that the image is displayed before we start submitting new commands to the queu
        self.semaphores.presentComplete = vkCreateSemaphore(self.device, semaphoreCreateInfo, None)
        # Create a semaphore used to synchronize command submission
        # Ensures that the image is not presented until all commands have been sumbitted and executed
        self.semaphores.renderComplete = vkCreateSemaphore(self.device, semaphoreCreateInfo, None)
        # Create a semaphore used to synchronize command submission
        # Ensures that the image is not presented until all commands for the text overlay have been sumbitted and executed
        # Will be inserted after the render complete semaphore if the text overlay is enabled
        self.semaphores.textOverlayComplete = vkCreateSemaphore(self.device, semaphoreCreateInfo, None)

        # Set up submit info structure
        # Semaphores will stay the same during application lifetime
        # Command buffer submission info is set by each example
        self.submitInfo = VkSubmitInfo(
            pWaitDstStageMask=self.submitPipelineStages,
            waitSemaphoreCount=1,
            pWaitSemaphores=[self.semaphores.presentComplete],
            signalSemaphoreCount=1,
            pSignalSemaphores=[self.semaphores.renderComplete]
        )

    # override in derived class
    def render(self):
        return NotImplemented

    # Pure virtual function to be overriden by the dervice class
    # Called in case of an event where e.g. the framebuffer has to be rebuild and thus
    # all command buffers that may reference this
    def buildCommandBuffers(self):
        return NotImplemented

    # Creates a new (graphics) command pool object storing command buffers
    def createCommandPool(self):
        cmdPoolInfo = VkCommandPoolCreateInfo(
            queueFamilyIndex=self.swapChain.queueNodeIndex,
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        self.cmdPool = vkCreateCommandPool(self.device, cmdPoolInfo, None)

    # Setup default depth and stencil views
    def setupDepthStencil(self):
        image = VkImageCreateInfo(
            imageType=VK_IMAGE_TYPE_2D,
            format=self.depthFormat,
            extent=[self.winWidth, self.winHeight, 1],
            mipLevels=1,
            arrayLayers=1,
            samples=VK_SAMPLE_COUNT_1_BIT,
            tiling=VK_IMAGE_TILING_OPTIMAL,
            usage=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            flags=0
        )

        mem_alloc = VkMemoryAllocateInfo(
            allocationSize=0,
            memoryTypeIndex=0
        )

        depthStencilView = VkImageViewCreateInfo(
            viewType=VK_IMAGE_VIEW_TYPE_2D,
            format=self.depthFormat,
            flags=0,
            subresourceRange=[VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1]
        )

        self.depthStencil.image = vkCreateImage(self.device, image, None)
        memReqs = vkGetImageMemoryRequirements(self.device, self.depthStencil.image)
        mem_alloc.allocationSize = memReqs.size
        mem_alloc.memoryTypeIndex = self.vulkanDevice.getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.depthStencil.mem = vkAllocateMemory(self.device, mem_alloc, None)
        vkBindImageMemory(self.device, self.depthStencil.image, self.depthStencil.mem, 0)

        depthStencilView.image = self.depthStencil.image
        self.depthStencil.view = vkCreateImageView(self.device, depthStencilView, None)

    def setupFrameBuffer(self):
        # Depth/Stencil attachment is the same for all frame buffers
        attachments = [0, self.depthStencil.view]

        # create frame buffers for every swap chain image
        for i, im in enumerate(self.swapChain.images):
            attachments[0] = self.swapChain.buffers[i].view
            frameBufferCreateInfo = VkFramebufferCreateInfo(
                renderPass=self.renderPass,
                attachmentCount=len(attachments),
                pAttachments=attachments,
                width=self.winWidth,
                height=self.winHeight,
                layers=1
            )
            fbuffer = vkCreateFramebuffer(self.device, frameBufferCreateInfo, None)
            self.frameBuffers.append(fbuffer)

    def setupRenderPass(self):
        # color attachment
        ca = VkAttachmentDescription(
            format=self.swapChain.colorFormat,
            samples=VK_SAMPLE_COUNT_1_BIT,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        )
        # depth attachment
        da = VkAttachmentDescription(
            format=self.depthFormat,
            samples=VK_SAMPLE_COUNT_1_BIT,
            loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )
        attachments = [ca, da]

        colorReference = VkAttachmentReference(
            attachment=0,
            layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )

        depthReference = VkAttachmentReference(
            attachment=1,
            layout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )

        subpassDescription = VkSubpassDescription(
            pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[colorReference],
            pDepthStencilAttachment=[depthReference],
            inputAttachmentCount=0,
            preserveAttachmentCount=0
        )

        # Subpass dependencies for layout transitions
        dt1 = VkSubpassDependency(
            srcSubpass=ffi.cast('uint32_t', VK_SUBPASS_EXTERNAL),
            dstSubpass=0,
            srcStageMask=VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=VK_ACCESS_MEMORY_READ_BIT,
            dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT
        )
        dt2 = VkSubpassDependency(
            srcSubpass=0,
            dstSubpass=ffi.cast('uint32_t', VK_SUBPASS_EXTERNAL),
            srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstStageMask=VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dstAccessMask=VK_ACCESS_MEMORY_READ_BIT,
            dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT
        )

        dependencies = [dt1, dt2]

        renderPassInfo = VkRenderPassCreateInfo(
            attachmentCount=len(attachments),
            pAttachments=attachments,
            subpassCount=1,
            pSubpasses=[subpassDescription],
            dependencyCount=len(dependencies),
            pDependencies=dependencies
        )

        self.renderPass = vkCreateRenderPass(self.device, renderPassInfo, None)

    # Connect and prepare the swap chain
    def initSwapchain(self):
        self.swapChain.initSurface(self)
        self.colorformat = self.swapChain.colorFormat

    # Create swap chain images
    def setupSwapChain(self):
        self.swapChain.create(self.winWidth, self.winHeight, self.__enableVSync)

    # Check if command buffers are valid (!= VK_NULL_HANDLE)
    def checkCommandBuffers(self):
        for cmdBuffer in self.drawCmdBuffers:
            if cmdBuffer == VK_NULL_HANDLE:
                return False
        return True

    # Create command buffers for drawing commands
    def createCommandBuffers(self):
        # Create one command buffer for each swap chain image and reuse for rendering
        cmdBufAllocateInfo = VkCommandBufferAllocateInfo(
            commandPool=self.cmdPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=self.swapChain.imageCount
        )
        cmdBuffers = vkAllocateCommandBuffers(self.device, cmdBufAllocateInfo)
        self.drawCmdBuffers = [ffi.addressof(cmdBuffers, i)[0] for i in range(self.swapChain.imageCount)]

    # Destroy all command buffers and set their handles to VK_NULL_HANDLE
    # May be necessary during runtime if options are toggled
    def destroyCommandBuffers(self):
        cmdBuffers = ffi.new('VkCommandBuffer[]', self.drawCmdBuffers)
        vkFreeCommandBuffers(self.device, self.cmdPool, len(cmdBuffers), cmdBuffers)
        self.drawCmdBuffers = []

    # Create command buffer for setup commands
    def createSetupCommandBuffer(self):
        if self.setupCmdBuffer != VK_NULL_HANDLE:
            vkFreeCommandBuffers(self.device, self.cmdPool, 1, [self.setupCmdBuffer])
            self.setupCmdBuffer = VK_NULL_HANDLE

        cmdBufAllocateInfo = VkCommandBufferAllocateInfo(
            commandPool=self.cmdPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        self.setupCmdBuffer = vkAllocateCommandBuffers(self.device, cmdBufAllocateInfo)[0]

        cmdBufInfo = VkCommandBufferBeginInfo()

        vkBeginCommandBuffer(self.setupCmdBuffer, cmdBufInfo)

    # Finalize setup command bufferm submit it to the queue and remove it
    def flushSetupCommandBuffer(self):
        if self.setupCmdBuffer == VK_NULL_HANDLE:
            return

        vkEndCommandBuffer(self.setupCmdBuffer)

        submitInfo = VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[self.setupCmdBuffer]
        )

        vkQueueSubmit(self.queue, 1, submitInfo, VK_NULL_HANDLE)
        vkQueueWaitIdle(self.queue)

        vkFreeCommandBuffers(self.device, self.cmdPool, 1, [self.setupCmdBuffer])
        self.setupCmdBuffer = VK_NULL_HANDLE

    # Command buffer creation
    # Creates and returns a new command buffer
    def createCommandBuffer(self, level, begin):
        cmdBufAllocateInfo = VkCommandBufferAllocateInfo(
            commandPool=self.cmdPool,
            level=level,
            commandBufferCount=1
        )

        cmdBuffer = vkAllocateCommandBuffers(self.device, cmdBufAllocateInfo)[0]

        # If requested, also start the new command buffer
        if begin:
            cmdbufInfo = VkCommandBufferBeginInfo()
            vkBeginCommandBuffer(cmdBuffer, cmdbufInfo)

        return cmdBuffer

    # End the command buffer, submit it to the queue and free (if requested)
    # Note : Waits for the queue to become idle
    def flushCommandBuffer(self, commandBuffer, queue, free):
        if commandBuffer == VK_NULL_HANDLE:
            return

        vkEndCommandBuffer(commandBuffer)

        submitInfo = VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[commandBuffer]
        )

        vkQueueSubmit(self.queue, 1, submitInfo, VK_NULL_HANDLE)
        vkQueueWaitIdle(self.queue)

        vkFreeCommandBuffers(self.device, self.cmdPool, 1, [commandBuffer])

    # Create a cache pool for rendering pipelines
    def createPipelineCache(self):
        pipelineCacheCreateInfo = VkPipelineCacheCreateInfo()
        self.pipelineCache = vkCreatePipelineCache(self.device, pipelineCacheCreateInfo, None)

    # Prepare commonly used Vulkan functions
    def prepare(self):
        if self.vulkanDevice.enableDebugMarkers:
            vkDebug.DebugMarker.setup(self.device)

        self.createCommandPool()
        self.setupSwapChain()
        self.createCommandBuffers()
        self.setupDepthStencil()
        self.setupRenderPass()
        self.createPipelineCache()
        self.setupFrameBuffer()
        # Create a simple texture loader class
        # self._textureLoader = vkTools.VulkanTextureLoader(self._vulkanDevice, self._queue, self._cmdPool)
        if self.enableTextOverlay:
            # Load the text rendering shaders
            ssv, vertShader = self.loadShader("{}shaders/base/textoverlay.vert.spv".format(self._getAssetPath()),
                                              VK_SHADER_STAGE_VERTEX_BIT)
            ssf, fragShader = self.loadShader("{}shaders/base/textoverlay.frag.spv".format(self._getAssetPath()),
                                              VK_SHADER_STAGE_FRAGMENT_BIT)
            shaderStages = [ssv, ssf]
            if not vertShader in self._shaderModules:
                self.shaderModules.append(vertShader)
            if not fragShader in self._shaderModules:
                self.shaderModules.append(fragShader)
            self.textOverlay = VulkanTextOverlay(self.vulkanDevice, self.queue, self.frameBuffers,
                                                 self.colorformat,
                                                 self.depthFormat, self.width, self.height, shaderStages)
            self.updateTextOverlay()

    # Load a SPIR-V shader
    def loadShader(self, fileName, stage):
        with open(fileName, 'rb') as sf:
            code = sf.read()
            codeSize = len(code)
            c_code = ffi.new('unsigned char []', code)
            pcode = ffi.cast('uint32_t*', c_code)

            createInfo = VkShaderModuleCreateInfo(codeSize=codeSize, pCode=pcode)

            module = vkCreateShaderModule(self.device, createInfo, None)

            shaderStage = VkPipelineShaderStageCreateInfo(
                stage=stage,
                module=module,
                pName='main',
            )
            return shaderStage, module

    # Prepare the frame for workload submission
    # - Acquires the next image from the swap chain
    # - Sets the default wait and signal semaphores
    def prepareFrame(self):
        # Acquire the next image from the swap chaing
        self.currentBuffer = self.swapChain.acquireNextImage(self.semaphores.presentComplete)

    # Submit the frames' workload
    # - Submits the text overlay (if enabled)
    def submitFrame(self):
        submitTextOverlay = self.enableTextOverlay and self.textOverlay

        if submitTextOverlay:
            # Wait for color attachment output to finish before rendering the text overlay
            stageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            self.submitInfo = VkSubmitInfo(
                pWaitDstStageMask=stageFlags,
                # Set semaphores
                # Wait for render complete semaphore
                waitSemaphoreCount=1,
                pWaitSemaphores=[self.semaphores.renderComplete],
                # Signal ready with text overlay complete semaphpre
                signalSemaphoreCount=1,
                pSignalSemaphores=[self.semaphores.textOverlayComplete],
                # Submit current text overlay command buffer
                commandBufferCount=1,
                pCommandBuffers=[self.textOverlay.cmdBuffers[self.currentBuffer]]
            )
            vkQueueSubmit(self.queue, 1, self.submitInfo, VK_NULL_HANDLE)

            # Reset stage mask
            self.submitInfo.pWaitDstStageMask = self.submitPipelineStages
            # Reset wait and signal semaphores for rendering next frame
            # Wait for swap chain presentation to finish
            self.submitInfo.waitSemaphoreCount = 1
            self.submitInfo.pWaitSemaphores = self.semaphores.presentComplete
            # Signal ready with offscreen semaphore
            self.submitInfo.signalSemaphoreCount = 1
            self.submitInfo.pSignalSemaphores = self.semaphores.renderComplete
        semaphore = self.semaphores.textOverlayComplete if submitTextOverlay else self.semaphores.renderComplete
        self.swapChain.queuePresent(self.queue, self.currentBuffer, semaphore)
        vkQueueWaitIdle(self.queue)

    # the main render loop
    def renderLoop(self):
    # def exposeEvent(self, event):
        self.__destWidth = self.winWidth
        self.__destHeight = self.winHeight
        tStart = time.clock()

        if self.isExposed():
            self.render()

        self._frameCounter += 1
        tEnd = time.clock()
        self._frameTimer = tEnd - tStart

        # Convert to clamped timer value
        if not self.paused:
            self.timer += self.timerSpeed * self._frameTimer
            if self.timer > 1.0:
                self.timer -= 1.0
        self.__fpsTimer += self._frameTimer
        if self.__fpsTimer > 1.0:
            if not self.enableTextOverlay:
                self.setTitle(self.windowTitle)
            self._lastFPS = round(1.0 / self._frameTimer)
            self.__fpsTimer = 0.0
            self._frameCounter = 0

    def resizeEvent(self, event):
        super(VulkanExampleBase, self).resizeEvent(event)

        if not self.prepared:
            return

        self.prepared = False

        # recreate swapchain
        size = event.size()
        self.winWidth = size.width()
        self.winHeight = size.height()
        self.createSetupCommandBuffer()
        self.setupSwapChain()

        # recreate the frame buffers
        vkDestroyImageView(self.device, self.depthStencil.view, None)
        vkDestroyImage(self.device, self.depthStencil.image, None)
        vkFreeMemory(self.device, self.depthStencil.mem, None)
        self.setupDepthStencil()

        [vkDestroyFramebuffer(self.device, i, None) for i in self.frameBuffers]

        self.setupFrameBuffer()

        self.flushSetupCommandBuffer()

        # Command buffers need to be recreated as they may store
        # references to the recreated frame buffer
        self.destroyCommandBuffers()
        self.createCommandBuffers()
        self.buildCommandBuffers()

        vkQueueWaitIdle(self.queue)
        vkDeviceWaitIdle(self.device)

        if self.enableTextOverlay:
            pass

        self.prepared = True

        if self.isExposed():
            self.render()

    def show(self):
        super(VulkanExampleBase, self).show()

        self.initVulkan()
        self.initSwapchain()
        self.prepare()

        self.__timer.start()

if __name__ == '__main__':
    import sys

    app = QtGui.QGuiApplication(sys.argv)
    #
    win = VulkanExampleBase(True)
    win.show()

    def cleanUp():
        global win
        del win
        # win = None

    app.aboutToQuit.connect(cleanUp)

    sys.exit(app.exec())

