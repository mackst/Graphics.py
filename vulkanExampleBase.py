
import time

from pyVulkan import *

from PySide import (QtGui, QtCore)
import numpy as np

import glm
import camera
import vulkandebug as vkDebug
import vulkantools as vkTools
from vulkanswapchain import *
from vulkantextoverlay import VulkanTextOverlay
from vulkandevice import VulkanDevice

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


class GamePadState(object):

    def __init__(self):
        self.axisLeft = np.array([0, 0], np.float32)
        self.axisRight = np.array([0, 0], np.float32)


class VulkanExampleBase(QtGui.QWidget):

    INSTANCE = None

    def __init__(self, enableValidation, enabledFeatures, enableVSync=False, parent=None):
        super(VulkanExampleBase, self).__init__(parent)

        self.prepared = False
        self.width = 1280
        self.height = 720

        self.defaultClearColor = VkClearColorValue([0.025, 0.025, 0.025, 1.0])

        self.zoom = 0

        # Defines a frame rate independent timer value clamped from -1.0...1.0
        # For use in animations, rotations, etc.
        self.timer = 0.0
        # Multiplier for speeding up (or slowing down) the global timer
        self.timerSpeed = 0.25

        self.paused = False

        self.enableTextOverlay = False
        self.textOverlay = VulkanTextOverlay()

        # Use to adjust mouse rotation speed
        self.rotationSpeed = 1.0
        # Use to adjust mouse zoom speed
        self.zoomSpeed = 1.0

        self.camera = camera.Camera()

        self.rotation = np.array([0, 0, 0], np.float32)
        self.cameraPos = np.array([0, 0, 0], np.float32)
        self.mousePos = np.array([0, 0], np.float32)

        self.title = 'pyVulkan Example'
        self.name = 'pyVulkanExample'

        self.depthStencil = DepthStencil()

        self.gamePadState = GamePadState()

        # Last frame time, measured using a high performance timer (if available)
        self._frameTimer = 1.0
        # Frame counter to display fps
        self._frameCounter = 0
        self._lastFPS = 0
        # Vulkan instance, stores all per-application states
        self._instance = None
        # Physical device (GPU) that Vulkan will ise
        self._physicalDevice = None
        # Stores physical device properties (for e.g. checking device limits)
        self._deviceProperties = None
        # Stores phyiscal device features (for e.g. checking if a feature is available)
        self._deviceFeatures = None
        # Stores all available memory (type) properties for the physical device
        self._deviceMemoryProperties = None
        # @brief Logical device, application's view of the physical device (GPU)
        self._device = None
        # @brief Encapsulated physical and logical vulkan device
        self._vulkanDevice = None
        # Handle to the device graphics queue that command buffers are submitted to
        self._queue = None
        # Color buffer format
        self._colorformat = VK_FORMAT_R8G8B8A8_UNORM
        # Depth buffer format
        # Depth format is selected during Vulkan initialization
        self._depthFormat = None
        # Command buffer pool
        self._cmdPool = None
        # Command buffer used for setup
        self._setupCmdBuffer = VK_NULL_HANDLE
        # @brief Pipeline stages used to wait at for graphics queue submissions
        self._submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        # Contains command buffers and semaphores to be presented to the queue
        self._submitInfo = None
        # Command buffers used for rendering
        self._drawCmdBuffers = []
        # Global render pass for frame buffer writes
        self._renderPass = None
        # List of available frame buffers (same as number of swap chain images)
        self._frameBuffers = []
        # Active frame buffer index
        self._currentBuffer = 0
        # Descriptor set pool
        self._descriptorPool = VK_NULL_HANDLE
        # List of shader modules created (stored for cleanup)
        self._shaderModules = []
        # Pipeline cache object
        self._pipelineCache = None
        # Wraps the swap chain to present images (framebuffers) to the windowing system
        self._swapChain = VulkanSwapChain()
        # Synchronization semaphores
        self._semaphores = Semaphore()
        # Simple texture loader
        self._textureLoader = None

        # Set to true when example is created with enabled validation layers
        self.__enableValidation = enableValidation
        # Set to true if v-sync will be forced for the swapchain
        self.__enableVSync = enableVSync
        # Device features enabled by the example
        # If not set, no additional features are enabled (may result in validation layer errors)
        self.__enabledFeatures = enabledFeatures
        # fps timer (one second interval)
        self.__fpsTimer = 0.0
        # brief Indicates that the view (position, rotation) has changed and
        self.__viewUpdated = False
        # Destination dimensions for resizing the window
        self.__destWidth = 0
        self.__destHeight = 0
        self.__resizing = False

        self.__timer = QtCore.QTimer(self)
        self.__timer.timeout.connect(self.renderLoop)
        self.__timer.start()

        # # Enable console if validation is active
        # # Debug message callback will output to it
        # if self.__enableValidation:
        #     self.setupConsole('pyVulkanExample')

        self.initVulkan(self.__enableValidation)
        VulkanExampleBase.INSTANCE = self

    # Returns the base asset path (for shaders, models, textures) depending on the os
    def _getAssetPath(self):
        return './../data/'

    # Create application wide Vulkan instance
    def __createInstance(self, enableValidation):
        self.__enableValidation = enableValidation

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
            ext = [ffi.new('char[]', i) for i in enabledExtensions]
            extArray = ffi.new('char*[]', ext)
            instanceCreateInfo.enabledExtensionCount = len(enabledExtensions)
            instanceCreateInfo.ppEnabledExtensionNames = extArray
        if self.__enableValidation:
            layers = [ffi.new('char[]', i) for i in enabledExtensions]
            layerArray = ffi.new('char*[]', layers)
            instanceCreateInfo.enabledLayerCount = len(vkDebug.validationLayerNames)
            instanceCreateInfo.ppEnabledLayerNames = layerArray
        self._instance = vkCreateInstance(instanceCreateInfo, None)
        return True

    # Get window title with example name, device, et.
    def __getWindowTitle(self):
        return '{} - {} - {} fps'.format(self.title, ffi.string(self._deviceProperties.deviceName), self._frameCounter)

    # Called if the window is resized and some resources have to be recreatesd
    def resizeEvent(self, event):
        if not self.prepared:
            return

        self.prepared = False

        # recreate swap chain
        self.width = self.__destWidth
        self.height = self.__destHeight
        self.createSetupCommandBuffer()
        self.setupSwapChain()

        # recreate the frame buffers
        vkDestroyImageView(self._device, self.depthStencil.view, None)
        vkDestroyImage(self._device, self.depthStencil.image, None)
        vkFreeMemory(self._device, self.depthStencil.mem, None)
        self.setupDepthStencil()

        for i in self._frameBuffers:
            vkDestroyFramebuffer(self._device, i, None)

        self.setupFrameBuffer()

        self.flushSetupCommandBuffer()

        # Command buffers need to be recreated as they may store
        # references to the recreated frame buffer
        self.destroyCommandBuffers()
        self.createCommandBuffers()
        self.buildCommandBuffers()

        vkQueueWaitIdle(self._queue)
        vkDeviceWaitIdle(self._device)

        if self.enableTextOverlay:
            self.textOverlay.reallocateCommandBuffers()
            self.updateTextOverlay()

        self.camera.updateAspectRatio(self.width/float(self.height))

        self.viewChanged()

        self.prepared = True

        super(VulkanExampleBase, self).resizeEvent(event)

    # key press
    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_P:
            self.paused = True
        if key == QtCore.Qt.Key_F1:
            if self.enableTextOverlay:
                self.textOverlay.visible = not self.textOverlay.visible
        if key == QtCore.Qt.Key_Escape:
            self.close()

        if self.camera.firstperson:
            if key == QtCore.Qt.Key_W:
                self.camera.keys.up = True
            if key == QtCore.Qt.Key_S:
                self.camera.keys.down = True
            if key == QtCore.Qt.Key_A:
                self.camera.keys.left = True
            if key == QtCore.Qt.Key_D:
                self.camera.keys.right = True

    def keyReleaseEvent(self, event):
        key = event.key()
        if self.camera.firstperson:
            if key == QtCore.Qt.Key_W:
                self.camera.keys.up = False
            if key == QtCore.Qt.Key_S:
                self.camera.keys.down = False
            if key == QtCore.Qt.Key_A:
                self.camera.keys.left = False
            if key == QtCore.Qt.Key_D:
                self.camera.keys.right = False

    def enterEvent(self, event):
        if event.type() == QtCore.QEvent.Enter:
            pos = self.cursor().pos()
            self.mousePos[0] = pos.x()
            self.mousePos[1] = pos.y()
        else:
            pass

    def mouseMoveEvent(self, event):
        # pos = event.pos()
        self.mousePos[0] = event.x()
        self.mousePos[1] = event.y()

    def wheelEvent(self, event):
        super(VulkanExampleBase, self).wheelEvent(event)

    def __del__(self):
        # Clean up Vulkan resources
        del self._swapChain
        if self._descriptorPool != VK_NULL_HANDLE:
            vkDestroyDescriptorPool(self._device, self._descriptorPool, None)
        if self._setupCmdBuffer != VK_NULL_HANDLE:
            vkFreeCommandBuffers(self._device, self._cmdPool, 1, [self._setupCmdBuffer])

        self.destroyCommandBuffers()
        vkDestroyRenderPass(self._device, self._renderPass, None)
        for i in self._frameBuffers:
            vkDestroyFramebuffer(self._device, i, None)

        for i in self._shaderModules:
            vkDestroyShaderModule(self._device, i, None)

        vkDestroyImageView(self._device, self.depthStencil.view, None)
        vkDestroyImage(self._device, self.depthStencil.image, None)
        vkFreeMemory(self._device, self.depthStencil.mem, None)

        vkDestroyPipelineCache(self._device, self._pipelineCache, None)

        if self._textureLoader:
            del self._textureLoader

        vkDestroyCommandPool(self._device, self._cmdPool, None)

        vkDestroySemaphore(self._device, self._semaphores.presentComplete, None)
        vkDestroySemaphore(self._device, self._semaphores.renderComplete, None)
        vkDestroySemaphore(self._device, self._semaphores.textOverlayComplete, None)

        if self.enableTextOverlay:
            del self.textOverlay

        del self._vulkanDevice

        if self.__enableValidation:
            vkDebug.freeDebugCallback(self._instance)

        vkDestroyInstance(self._instance, None)

    def initVulkan(self, enableValidation):
        '''Setup the vulkan instance, enable required extensions and connect to the physical device (GPU)'''
        # Vulkan instance
        self.__createInstance(enableValidation)

        # If requested, we enable the default validation layers for debugging
        if enableValidation:
            # The report flags determine what type of messages for the layers will be displayed
            # For validating (debugging) an appplication the error and warning bits should suffice
            debugReportFlags = VK_DEBUG_REPORT_ERROR_BIT_EXT
            # Additional flags include performance info, loader and layer debug messages, etc.
            vkDebug.setupDebugging(self._instance, debugReportFlags, VK_NULL_HANDLE)

        physicalDevices = vkEnumeratePhysicalDevices(self._instance)

        # Note :
        # This example will always use the first physical device reported,
        # change the vector index if you have multiple Vulkan devices installed
        # and want to use another one
        self._physicalDevice = physicalDevices[0]

        # Vulkan device creation
        # This is handled by a separate class that gets a logical device representation
        # and encapsulates functions related to a device
        self._vulkanDevice = VulkanDevice(self._physicalDevice)
        self._vulkanDevice.createLogicalDevice(self.__enabledFeatures)
        self._device = self._vulkanDevice.logicalDevice

        # Store properties (including limits) and features of the phyiscal device
        # So examples can check against them and see if a feature is actually supported
        self._deviceProperties = vkGetPhysicalDeviceProperties(self._physicalDevice)
        self._deviceFeatures = vkGetPhysicalDeviceFeatures(self._physicalDevice)
        # Gather physical device memory properties
        self._deviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties(self._physicalDevice)

        # Get a graphics queue from the device
        self._queue = vkGetDeviceQueue(self._device, self._vulkanDevice.queueFamilyIndices.graphics, 0)

        # Find a suitable depth format
        self._depthFormat = vkTools.getSupportedDepthFormat(self._physicalDevice)

        self._swapChain.connect(self._instance, self._physicalDevice, self._device)

        # Create synchronization objects
        semaphoreCreateInfo = VkSemaphoreCreateInfo(flags=0)
        # Create a semaphore used to synchronize image presentation
        # Ensures that the image is displayed before we start submitting new commands to the queu
        self._semaphores.presentComplete = vkCreateSemaphore(self._device, semaphoreCreateInfo, None)
        # Create a semaphore used to synchronize command submission
        # Ensures that the image is not presented until all commands have been sumbitted and executed
        self._semaphores.renderComplete = vkCreateSemaphore(self._device, semaphoreCreateInfo, None)
        # Create a semaphore used to synchronize command submission
        # Ensures that the image is not presented until all commands for the text overlay have been sumbitted and executed
        # Will be inserted after the render complete semaphore if the text overlay is enabled
        self._semaphores.textOverlayComplete = vkCreateSemaphore(self._device, semaphoreCreateInfo, None)

        # Set up submit info structure
        # Semaphores will stay the same during application lifetime
        # Command buffer submission info is set by each example
        self._submitInfo = VkSubmitInfo(
            pWaitDstStageMask=self._submitPipelineStages,
            waitSemaphoreCount=1,
            pWaitSemaphores=self._semaphores.presentComplete,
            signalSemaphoreCount=1,
            pSignalSemaphores=self._semaphores.renderComplete
        )

    def setupConsole(self, title):
        pass

    def setupWindow(self):
        self.setWindowTitle(self.name)
        self.setFixedSize(self.width, self.height)
        self.createWinId()
        self.show()

    # override in derived class
    def render(self):
        return NotImplemented

    # Called when view change occurs
    # Can be overriden in derived class to e.g. update uniform buffers
    # Containing view dependant matrices
    def viewChanged(self):
        return NotImplemented

    # Pure virtual function to be overriden by the dervice class
    # Called in case of an event where e.g. the framebuffer has to be rebuild and thus
    # all command buffers that may reference this
    def buildCommandBuffers(self):
        return NotImplemented

    # Creates a new (graphics) command pool object storing command buffers
    def createCommandPool(self):
        comdPoolInfo = VkCommandPoolCreateInfo(
            queueFamilyIndex=self._swapChain.queueNodeIndex,
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        self._cmdPool = vkCreateCommandPool(self._device, comdPoolInfo, ffi.NULL)

    # Setup default depth and stencil views
    def setupDepthStencil(self):
        image = VkImageCreateInfo(
            imageType=VK_IMAGE_TYPE_2D,
            format=self._depthFormat,
            extent=[self.width, self.height, 1],
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
            format=self._depthFormat,
            flags=0,
            subresourceRange=[VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1]
        )

        self.depthStencil.image = vkCreateImage(self._device, image, ffi.NULL)
        memReqs = vkGetImageMemoryRequirements(self._device, self.depthStencil.image)
        mem_alloc.allocationSize = memReqs.size
        mem_alloc.memoryTypeIndex = VulkanDevice.getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        self.depthStencil.mem = vkAllocateMemory(self._device, mem_alloc, ffi.NULL)
        vkBindImageMemory(self._device, self.depthStencil.image, self.depthStencil.mem, 0)

        depthStencilView.image = self.depthStencil.image
        self.depthStencil.view = vkCreateImageView(self._device, depthStencilView, ffi.NULL)

    def setupFrameBuffer(self):
        # Depth/Stencil attachment is the same for all frame buffers
        attachments = [0, self.depthStencil.view]

        # create frame buffers for every swap chain image
        for im in self._swapChain.images:
            attachments[0] = im
            frameBufferCreateInfo = VkFramebufferCreateInfo(
                renderPass=self._renderPass,
                attachmentCount=2,
                pAttachments=attachments,
                width=self.width,
                height=self.height,
                layers=1
            )
            self._frameBuffers.append(vkCreateFramebuffer(self._device, frameBufferCreateInfo, ffi.NULL))

    def setupRenderPass(self):
        # color attachment
        ca = VkAttachmentDescription(
            format=self._colorformat,
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
            format=self._depthFormat,
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
            pColorAttachments=colorReference,
            pDepthStencilAttachment=depthReference,
            inputAttachmentCount=0,
            preserveAttachmentCount=0
        )

        # Subpass dependencies for layout transitions
        dt1 = VkSubpassDependency(
            srcSubpass=VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=VK_ACCESS_MEMORY_READ_BIT,
            dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dependencyFlags=VK_DEPENDENCY_BY_REGION_BIT
        )
        dt2 = VkSubpassDependency(
            srcSubpass=0,
            dstSubpass=VK_SUBPASS_EXTERNAL,
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
            pSubpasses=subpassDescription,
            dependencyCount=len(dependencies),
            pDependencies=dependencies
        )

        self._renderPass = vkCreateRenderPass(self._device, renderPassInfo, ffi.NULL)

    # Connect and prepare the swap chain
    def initSwapchain(self):
        self._swapChain.initSurface(self)

    # Create swap chain images
    def setupSwapChain(self):
        self._swapChain.create(self.width, self.height, self.__enableVSync)

    # Check if command buffers are valid (!= VK_NULL_HANDLE)
    def checkCommandBuffers(self):
        for cmdBuffer in self._drawCmdBuffers:
            if cmdBuffer == VK_NULL_HANDLE:
                return False
        return True

    # Create command buffers for drawing commands
    def createCommandBuffers(self):
        # Create one command buffer for each swap chain image and reuse for rendering
        cmdBufAllocateInfo = VkCommandBufferAllocateInfo(
            commandPool=self._cmdPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=self._swapChain.imageCount
        )
        cmdBuffers = vkAllocateCommandBuffers(self._device, cmdBufAllocateInfo)
        self._drawCmdBuffers = [ffi.addressof(cmdBuffers, i)[0] for i in range(self._swapChain.imageCount)]

    # Destroy all command buffers and set their handles to VK_NULL_HANDLE
    # May be necessary during runtime if options are toggled
    def destroyCommandBuffers(self):
        cmdBuffers = ffi.new('VkCommandBuffer[]', self._drawCmdBuffers)
        vkFreeCommandBuffers(self._device, self._cmdPool, len(cmdBuffers), cmdBuffers)
        self._drawCmdBuffers = []

    # Create command buffer for setup commands
    def createSetupCommandBuffer(self):
        if self._setupCmdBuffer != VK_NULL_HANDLE:
            vkFreeCommandBuffers(self._device, self._cmdPool, 1, [self._setupCmdBuffer])
            self._setupCmdBuffer = VK_NULL_HANDLE

        cmdBufAllocateInfo = VkCommandBufferAllocateInfo(
            commandPool=self._cmdPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        self._setupCmdBuffer = vkAllocateCommandBuffers(self._device, cmdBufAllocateInfo)[0]

        cmdBufInfo = VkCommandBufferBeginInfo()

        vkBeginCommandBuffer(self._setupCmdBuffer, cmdBufInfo)


    # Finalize setup command bufferm submit it to the queue and remove it
    def flushSetupCommandBuffer(self):
        if self._setupCmdBuffer == VK_NULL_HANDLE:
            return

        vkEndCommandBuffer(self._setupCmdBuffer)

        submitInfo = VkSubmitInfo(
            commandBufferCount=1,
            pCommandBuffers=[self._setupCmdBuffer]
        )

        vkQueueSubmit(self._queue, 1, submitInfo, VK_NULL_HANDLE)
        vkQueueWaitIdle(self._queue)

        vkFreeCommandBuffers(self._device, self._cmdPool, 1, [self._setupCmdBuffer])
        self._setupCmdBuffer = VK_NULL_HANDLE

    # Command buffer creation
    # Creates and returns a new command buffer
    def createCommandBuffer(self, level, begin):
        cmdBufAllocateInfo = VkCommandBufferAllocateInfo(
            commandPool=self._cmdPool,
            level=level,
            commandBufferCount=1
        )

        cmdBuffer = vkAllocateCommandBuffers(self._device, cmdBufAllocateInfo)[0]

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

        vkQueueSubmit(self._queue, 1, submitInfo, VK_NULL_HANDLE)
        vkQueueWaitIdle(self._queue)

        vkFreeCommandBuffers(self._device, self._cmdPool, 1, [commandBuffer])

    # Create a cache pool for rendering pipelines
    def createPipelineCache(self):
        pipelineCacheCreateInfo = VkPipelineCacheCreateInfo()
        self._pipelineCache = vkCreatePipelineCache(self._device, pipelineCacheCreateInfo, None)

    # Prepare commonly used Vulkan functions
    def prepare(self):
        if self._vulkanDevice.enableDebugMarkers:
            vkDebug.DebugMarker.setup(self._device)

        self.createCommandPool()
        self.createSetupCommandBuffer()
        self.setupSwapChain()
        self.createCommandBuffers()
        self.setupDepthStencil()
        self.setupRenderPass()
        self.createPipelineCache()
        self.setupFrameBuffer()
        self.flushSetupCommandBuffer()
        # Recreate setup command buffer for derived class
        self.createSetupCommandBuffer()
        # Create a simple texture loader class
        self._textureLoader = vkTools.VulkanTextureLoader(self._vulkanDevice, self._queue, self._cmdPool)
        if self.enableTextOverlay:
            # Load the text rendering shaders
            shaderStages = []
            shaderStages.append(self.loadShader("{}shaders/base/textoverlay.vert.spv".format(self._getAssetPath()), VK_SHADER_STAGE_VERTEX_BIT))
            shaderStages.append(self.loadShader("{}shaders/base/textoverlay.frag.spv".format(self._getAssetPath()), VK_SHADER_STAGE_FRAGMENT_BIT))
            self.textOverlay = VulkanTextOverlay(self._vulkanDevice, self._queue, self._frameBuffers, self._colorformat,
                                                 self._depthFormat, self.width, self.height, shaderStages)
            self.updateTextOverlay()

    # Load a SPIR-V shader
    def loadShader(self, fileName, stage):
        with open(fileName, 'rb') as sf:
            code = sf.read()
            codeSize = len(code)
            c_code = ffi.new('unsigned char []', code)
            pcode = ffi.cast('uint32_t*', c_code)

            createInfo = VkShaderModuleCreateInfo(codeSize=codeSize, pCode=pcode)

            module = vkCreateShaderModule(self._device, createInfo, None)

            shaderStage = VkPipelineShaderStageCreateInfo(
                stage=stage,
                module=module,
                pName='main',
            )
            return shaderStage, module

    # Create a buffer, fill it with data (if != NULL) and bind buffer memory
    def createBuffer(self, usageFlags, memoryPropertyFlags, size, data):
        memAlloc = vkTools.initializers.memoryAllocateInfo()
        bufferCreateInfo = vkTools.initializers.bufferCreateInfo()

        buf = vkCreateBuffer(self._device, bufferCreateInfo, None)

        memReqs = vkGetBufferMemoryRequirements(self._device, buf)
        memAlloc.allocationSize = memReqs.size
        memAlloc.memoryTypeIndex = self._vulkanDevice.getMemoryType(memReqs.memoryTypeBits, memoryPropertyFlags)
        memory = vkAllocateMemory(self._device, memAlloc, None)

        mapped = vkMapMemory(self._device, memory, 0, size, 0)
        ffi.memmove(mapped, data, size)
        vkUnmapMemory(self._device, memory)

        vkBindBufferMemory(self._device, buf, memory, 0)
        return buf, memory

    # Load a mesh (using ASSIMP) and create vulkan vertex and index buffers with given vertex layout
    def loadMesh(self, fiename, meshBuffer, vertexLayout):
        pass

    # Start the main render loop
    def renderLoop(self):
        self.__destWidth = self.width
        self.__destHeight = self.height
        tStart = time.clock()

        if self.__viewUpdated:
            self.__viewUpdated = False
            self.viewChanged()

        self.render()

        self._frameCounter += 1
        tEnd = time.clock()
        # tDiff = tEnd - tStart
        self._frameTimer = tEnd - tStart
        self.camera.update(self._frameTimer)
        if self.camera.moving():
            self.__viewUpdated = True
        # Convert to clamped timer value
        if not self.paused:
            self.timer += self.timerSpeed * self._frameTimer
            if self.timer > 1.0:
                self.timer -= 1.0
        self.__fpsTimer += self._frameTimer
        if self.__fpsTimer > 1.0:
            if not self.enableTextOverlay:
                self.setWindowTitle(self.__getWindowTitle())
            self._lastFPS = round(1.0 / self._frameTimer)
            self.updateTextOverlay()
            self.__fpsTimer = 0.0
            self._frameCounter = 0

        # Flush device to make sure all resources can be freed
        vkDeviceWaitIdle(self._device)

    def updateTextOverlay(self):
        if not self.enableTextOverlay:
            return

        self.textOverlay.beginTextUpdate()

    # Called when the text overlay is updating
    # Can be overriden in derived class to add custom text to the overlay
    def getOverlayText(self, textOverlay):
        pass

    # Prepare the frame for workload submission
    # - Acquires the next image from the swap chain
    # - Sets the default wait and signal semaphores
    def prepareFrame(self):
        # Acquire the next image from the swap chaing
        self._currentBuffer = self._swapChain.acquireNextImage(self._semaphores.presentComplete)

    # Submit the frames' workload
    # - Submits the text overlay (if enabled)
    def submitFrame(self):
        submitTextOverlay = self.enableTextOverlay and self.textOverlay.visible

        if submitTextOverlay:
            # Wait for color attachment output to finish before rendering the text overlay
            stageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            self._submitInfo = VkSubmitInfo(
                pWaitDstStageMask=stageFlags,
                # Set semaphores
                # Wait for render complete semaphore
                waitSemaphoreCount=1,
                pWaitSemaphores=self._semaphores.renderComplete,
                # Signal ready with text overlay complete semaphpre
                signalSemaphoreCount=1,
                pSignalSemaphores=self._semaphores.textOverlayComplete,
                # Submit current text overlay command buffer
                commandBufferCount=1,
                pCommandBuffers=[self.textOverlay.cmdBuffers[self._currentBuffer]]
            )
            vkQueueSubmit(self._queue, 1, self._submitInfo, VK_NULL_HANDLE)

            # Reset stage mask
            self._submitInfo.pWaitDstStageMask = self._submitPipelineStages
            # Reset wait and signal semaphores for rendering next frame
            # Wait for swap chain presentation to finish
            self._submitInfo.waitSemaphoreCount = 1
            self._submitInfo.pWaitSemaphores = self._semaphores.presentComplete
            # Signal ready with offscreen semaphore
            self._submitInfo.signalSemaphoreCount = 1
            self._submitInfo.pSignalSemaphores = self._semaphores.renderComplete
        semaphore = self._semaphores.textOverlayComplete if submitTextOverlay else self._semaphores.renderComplete
        self._swapChain.queuePresent(self._queue, self._currentBuffer, semaphore)
        vkQueueWaitIdle(self._queue)