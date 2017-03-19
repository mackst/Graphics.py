##########################################################################
# Assorted commonly used Vulkan helper functions
#                              - port from Sascha Willems - www.saschawillems.de
#
# Copyright (C) 2017 by Shi Chi
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
##########################################################################

from pyVulkan import *

from PIL import Image


DEFAULT_FENCE_TIMEOUT = 100000000000



def checkDeviceExtensionPresent(physicalDevice, extensionName):
    extensions = vkEnumerateDeviceExtensionProperties(physicalDevice)

    for ext in extensions:
        if ffi.string(ext.extensionName) == extensionName:
            return True

    return False

def getSupportedDepthFormat(physicalDevice):
    # Since all depth formats may be optional, we need to find a suitable depth format to use
    # Start with the highest precision packed format
    depthFormats = [
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM
    ]

    for i in depthFormats:
        formatProps = vkGetPhysicalDeviceFormatProperties(physicalDevice, i)
        # Format must support depth stencil attachment for optimal tiling
        if formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT:
            return i

    return -1


class initializers(object):

    @staticmethod
    def memoryAllocateInfo():
        return VkMemoryAllocateInfo(
            allocationSize=0,
            memoryTypeIndex=0
        )

    @staticmethod
    def bufferCreateInfo(usage, size):
        return VkBufferCreateInfo(
            flags=0,
            usage=usage,
            size=size
        )


class VulkanTextureLoader(object):

    def __init__(self, vulkanDevice, queue, cmdPool):
        self.__vulkanDevice = vulkanDevice
        self.__queue = queue
        self.__cmdPool = cmdPool
        self.__cmdBuffer = VK_NULL_HANDLE

        # Create command buffer for submitting image barriers
        # and converting tilings
        cmdBufInfo = VkCommandBufferAllocateInfo(
            commandPool=self.__cmdPool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1
        )

        self.__cmdBuffer = vkAllocateCommandBuffers(self.__vulkanDevice.logicalDevice, cmdBufInfo)[0]

    def __del__(self):
        vkFreeCommandBuffers(self.__vulkanDevice.logicalDevice, self.__cmdPool, 1, [self.__cmdBuffer])