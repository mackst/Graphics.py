from pyVulkan import *
from PIL import ImageFont


MAX_CHAR_COUNT = 1024


class _TextAlign(object):
    def __init__(self):
        self.alignLeft = 0
        self.alignCenter = 1
        self.alignRight = 2
TextAlign = _TextAlign()


class VulkanTextOverlay(object):

    def __init__(self, vulkanDevice=None, queue=None, framebuffers=None,
                 colorformat=None, depthformat=None, framebufferwidth=None,
                 framebufferheight=None, shaderstages=None):
        self.__vulkanDevice = vulkanDevice
        self.__queue = queue
        self.__colorFormat = colorformat
        self.__depthFormat = depthformat
        self.__frameBufferWidth = framebufferwidth
        self.__frameBufferHeight = framebufferheight

        self.__sampler = None
        self.__image = None
        self.__view = None
        self.__vertexBuffer = None
        self.__imageMemory = None
        self.__descriptorPool = None
        self.__descriptorSetLayout = None
        self.__descriptorSet = None
        self.__pipelineLayout = None
        self.__pipelineCache = None
        self.__pipeline = None
        self.__renderPass = None
        self.__commandPool = None
        self.__frameBuffers = framebuffers
        self.__shaderStages = shaderstages
        self.__fence = None

        # used during text updates
        self.__mappedLocal = 0
        self.__stbFontData = []
        self.__numLetters = 0
        self.__imFont = ImageFont.truetype('arial.ttf')

        self.visible = True
        self.invalidated = False

        self.cmdBuffers = []

        if self.__vulkanDevice:
            self.prepareResources()
            self.prepareRenderPass()
            self.preparePipeline()

    def __del__(self):
        del self.__vertexBuffer
        vkDestroySampler(self.__vulkanDevice.logicalDevice, self.__sampler, None)
        vkDestroyImage(self.__vulkanDevice.logicalDevice, self.__image, None)
        vkDestroyImageView(self.__vulkanDevice.logicalDevice, self.__view, None)
        vkFreeMemory(self.__vulkanDevice.logicalDevice, self.__imageMemory, None)
        vkDestroyDescriptorSetLayout(self.__vulkanDevice.logicalDevice, self.__descriptorSetLayout, None)
        vkDestroyDescriptorPool(self.__vulkanDevice.logicalDevice, self.__descriptorPool, None)
        vkDestroyPipelineLayout(self.__vulkanDevice.logicalDevice, self.__pipelineLayout, None)
        vkDestroyPipelineCache(self.__vulkanDevice.logicalDevice, self.__pipelineCache, None)
        vkDestroyPipeline(self.__vulkanDevice.logicalDevice, self.__pipeline, None)
        vkDestroyRenderPass(self.__vulkanDevice.logicalDevice, self.__renderPass, None)
        vkFreeCommandBuffers(self.__vulkanDevice.logicalDevice, self.__commandPool, len(self.cmdBuffers), self.cmdBuffers)
        vkDestroyCommandPool(self.__vulkanDevice.logicalDevice, self.__commandPool, None)
        vkDestroyFence(self.__vulkanDevice.logicalDevice, self.__fence, None)

    def prepareResources(self):
        fontIm = self.__imFont.getmask(self.__stbFontData, 'L')

        # command buffer

        # Pool
        cmdPoolInfo = VkCommandPoolCreateInfo(
            queueFamilyIndex=self.__vulkanDevice.queueFamilyIndices,
            flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        )
        self.__commandPool = vkCreateCommandPool(self.__vulkanDevice.logicalDevice, cmdPoolInfo, None)

        cmdBufAllocateInfo = VkCommandBufferAllocateInfo(
            commandPool=self.__commandPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=len(self.__frameBuffers)
        )
        cmdBuffers = vkAllocateCommandBuffers(self.__vulkanDevice.logicalDevice, cmdBufAllocateInfo)
        bufs = [ffi.addressof(cmdBuffers, i)[0] for i in range(len(self.__frameBuffers))]
        self.cmdBuffers = ffi.new('VkCommandBuffer[]', bufs)

        # vertex buffer
        self.__vertexBuffer = self.__vulkanDevice.createBuffer()