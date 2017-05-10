##########################################################################
# Class wrapping access to the swap chain
#                         - port from Sascha Willems - www.saschawillems.de
#
# Copyright (C) 2017 by Shi Chi
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
##########################################################################

from pyVulkan import *

import win32misc


class SwapChainBuffer(object):

    def __init__(self):
        self.image = None
        self.view = None

class VulkanSwapChain(object):

    def __init__(self):
        self.__instance = None
        self.__device = None
        self.__physicalDevice = None
        self.__surface = None
        self.__fpGetPhysicalDeviceSurfaceSupportKHR = None
        self.__fpGetPhysicalDeviceSurfaceCapabilitiesKHR = None
        self.__fpGetPhysicalDeviceSurfaceFormatsKHR = None
        self.__fpGetPhysicalDeviceSurfacePresentModesKHR = None
        self.__fpCreateSwapchainKHR = None
        self.__fpDestroySwapchainKHR = None
        self.__fpGetSwapchainImagesKHR = None
        self.__fpAcquireNextImageKHR = None
        self.__fpQueuePresentKHR = None
        self.__fpDestroySurfaceKHR = None

        self.colorFormat = None
        self.colorSpace = None
        # @brief Handle to the current swap chain, required for recreation
        self.swapChain = VK_NULL_HANDLE
        self.imageCount = 0
        self.images = []
        self.buffers = []
        # Index of the deteced graphics and presenting device queue
        # @brief Queue family index of the deteced graphics and presenting device queue
        self.queueNodeIndex = 0xffffffff

    def __del__(self):
        if self.swapChain != VK_NULL_HANDLE:
            for buf in self.buffers:
                vkDestroyImageView(self.__device, buf.view, None)

            if self.__surface != VK_NULL_HANDLE:
                self.__fpDestroySwapchainKHR(self.__device, self.swapChain, None)
                self.__fpDestroySurfaceKHR(self.__instance, self.__surface, None)

            self.__surface = VK_NULL_HANDLE
            self.swapChain = VK_NULL_HANDLE

    def initSurface(self, window):
        vkCreateWin32SurfaceKHR = vkGetInstanceProcAddr(self.__instance, 'vkCreateWin32SurfaceKHR')

        hwnd = window.winId()
        hinstance = win32misc.getInstance(hwnd)
        createInfo = VkWin32SurfaceCreateInfoKHR(
            hwnd=hwnd,
            hinstance=hinstance
        )
        self.__surface = vkCreateWin32SurfaceKHR(self.__instance, createInfo)
        if self.__surface is None:
            raise Exception("failed to create window surface!")

        # Get available queue family properties
        queueProps = vkGetPhysicalDeviceQueueFamilyProperties(self.__physicalDevice)

        # Iterate over each queue to learn whether it supports presenting:
        # Find a queue with present support
        # Will be used to present the swap chain images to the windowing system
        supportsPresent = [self.__fpGetPhysicalDeviceSurfaceSupportKHR(self.__physicalDevice, i, self.__surface) for i, prop in enumerate(queueProps)]

        # Search for a graphics and a present queue in the array of queue
        # families, try to find one that supports both
        graphicsQueueNodeIndex = 0
        presentQueueNodeIndex = 0
        for i, prop in enumerate(queueProps):
            if (prop.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0:
                graphicsQueueNodeIndex = i

            if self.__fpGetPhysicalDeviceSurfaceSupportKHR(self.__physicalDevice, i, self.__surface):
                presentQueueNodeIndex = i
                break

        self.queueNodeIndex = graphicsQueueNodeIndex

        # Get list of supported surface formats
        surfaceFormats = self.__fpGetPhysicalDeviceSurfaceFormatsKHR(self.__physicalDevice, self.__surface, )
        # If the surface format list only includes one entry with VK_FORMAT_UNDEFINED,
        # there is no preferered format, so we assume VK_FORMAT_B8G8R8A8_UNORM
        if len(surfaceFormats) == 1 and surfaceFormats[0].format == VK_FORMAT_UNDEFINED:
            self.colorFormat = VK_FORMAT_B8G8R8A8_UNORM
        else:
            # Always select the first available color format
            # If you need a specific format (e.g. SRGB) you'd need to
            # iterate over the list of available surface format and
            # check for it's presence
            self.colorFormat = surfaceFormats[0].format
        self.colorSpace = surfaceFormats[0].colorSpace

    def connect(self, instance, physicalDevice, device):
        self.__instance = instance
        self.__physicalDevice = physicalDevice
        self.__device = device
        self.__fpGetPhysicalDeviceSurfaceSupportKHR = vkGetInstanceProcAddr(self.__instance, 'vkGetPhysicalDeviceSurfaceSupportKHR')
        self.__fpGetPhysicalDeviceSurfaceCapabilitiesKHR = vkGetInstanceProcAddr(self.__instance, 'vkGetPhysicalDeviceSurfaceCapabilitiesKHR')
        self.__fpGetPhysicalDeviceSurfaceFormatsKHR = vkGetInstanceProcAddr(self.__instance, 'vkGetPhysicalDeviceSurfaceFormatsKHR')
        self.__fpGetPhysicalDeviceSurfacePresentModesKHR = vkGetInstanceProcAddr(self.__instance, 'vkGetPhysicalDeviceSurfacePresentModesKHR')
        self.__fpDestroySurfaceKHR = vkGetInstanceProcAddr(self.__instance, 'vkDestroySurfaceKHR')
        self.__fpCreateSwapchainKHR = vkGetDeviceProcAddr(self.__device, 'vkCreateSwapchainKHR')
        self.__fpDestroySwapchainKHR = vkGetDeviceProcAddr(self.__device, 'vkDestroySwapchainKHR')
        self.__fpGetSwapchainImagesKHR = vkGetDeviceProcAddr(self.__device, 'vkGetSwapchainImagesKHR')
        self.__fpAcquireNextImageKHR = vkGetDeviceProcAddr(self.__device, 'vkAcquireNextImageKHR')
        self.__fpQueuePresentKHR = vkGetDeviceProcAddr(self.__device, 'vkQueuePresentKHR')

    # Create the swapchain and get it's images with given width and height
    #
    # @param width Pointer to the width of the swapchain (may be adjusted to fit the requirements of the swapchain)
    # @param height Pointer to the height of the swapchain (may be adjusted to fit the requirements of the swapchain)
    # @param vsync (Optional) Can be used to force vsync'd rendering (by using VK_PRESENT_MODE_FIFO_KHR as presentation mode)
    def create(self, width, height, vsync=False):
        oldSwapchain = self.swapChain

        # Get physical device surface properties and formats
        surfCaps = self.__fpGetPhysicalDeviceSurfaceCapabilitiesKHR(self.__physicalDevice, self.__surface)

        # Get available present modes
        presentModes = self.__fpGetPhysicalDeviceSurfacePresentModesKHR(self.__physicalDevice, self.__surface)

        swapchainExtent = VkExtent2D()
        # If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
        if surfCaps.currentExtent.width == -1:
            # If the surface size is undefined, the size is set to
            # the size of the images requested.
            swapchainExtent.width = width
            swapchainExtent.height = height
        else:
            # If the surface size is defined, the swap chain size must match
            swapchainExtent = surfCaps.currentExtent
            width = surfCaps.currentExtent.width
            height = surfCaps.currentExtent.height

        # Select a present mode for the swapchain

        # The VK_PRESENT_MODE_FIFO_KHR mode must always be present as per spec
        # This mode waits for the vertical blank ("v-sync")
        swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR

        # If v-sync is not requested, try to find a mailbox mode
        # It's the lowest latency non-tearing present mode available
        if not vsync:
            for i, presentMode in enumerate(presentModes):
                if presentMode == VK_PRESENT_MODE_MAILBOX_KHR:
                    swapchainPresentMode = VK_PRESENT_MODE_MAILBOX_KHR
                    break
                if swapchainPresentMode != VK_PRESENT_MODE_MAILBOX_KHR and presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR:
                    swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR

        # Determine the number of images
        desiredNumberOfSwapchainImages = surfCaps.minImageCount + 1
        if surfCaps.maxImageCount > 0 and desiredNumberOfSwapchainImages > surfCaps.maxImageCount:
            desiredNumberOfSwapchainImages = surfCaps.maxImageCount

        # Find the transformation of the surface
        preTransform = None
        if surfCaps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR:
            # We prefer a non-rotated transform
            preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
        else:
            preTransform = surfCaps.currentTransform

        swapchainCI = VkSwapchainCreateInfoKHR(
            surface=self.__surface,
            minImageCount=desiredNumberOfSwapchainImages,
            imageFormat=self.colorFormat,
            imageColorSpace=self.colorSpace,
            imageExtent=[swapchainExtent.width, swapchainExtent.height],
            imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            preTransform=preTransform,
            imageArrayLayers=1,
            imageSharingMode=VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount=0,
            presentMode=swapchainPresentMode,
            oldSwapchain=oldSwapchain,
            clipped=VK_TRUE, # Setting clipped to VK_TRUE allows the implementation to discard rendering outside of the surface area
            compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
        )

        self.swapChain = self.__fpCreateSwapchainKHR(self.__device, swapchainCI, None)

        # If an existing sawp chain is re-created, destroy the old swap chain
        # This also cleans up all the presentable images
        if oldSwapchain != VK_NULL_HANDLE:
            for buf in self.buffers:
                vkDestroyImageView(self.__device, buf.view, None)
            self.__fpDestroySwapchainKHR(self.__device, oldSwapchain, None)

        # Get the swap chain images
        self.images = self.__fpGetSwapchainImagesKHR(self.__device, self.swapChain)
        self.imageCount = len(self.images)

        # Get the swap chain buffers containing the image and imageview
        self.buffers = []
        for i, image in enumerate(self.images):
            colorAttachmentView = VkImageViewCreateInfo(
                flags=0,
                image=image,
                viewType=VK_IMAGE_VIEW_TYPE_2D,
                format=self.colorFormat,
                components=[VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                            VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A],
                subresourceRange=[VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1]
            )
            buf = SwapChainBuffer()
            buf.image = image

            buf.view = vkCreateImageView(self.__device, colorAttachmentView, None)
            self.buffers.append(buf)

    # Acquires the next image in the swap chain
    #
    # @param presentCompleteSemaphore (Optional) Semaphore that is signaled when the image is ready for use
    # @param imageIndex Pointer to the image index that will be increased if the next image could be acquired
    #
    # @note The function will always wait until the next image has been acquired by setting timeout to UINT64_MAX
    #
    # @return VkResult of the image acquisition
    def acquireNextImage(self, presentCompleteSemaphore):
        # By setting timeout to UINT64_MAX we will always wait until the next image has been acquired or an actual error is thrown
        # With that we don't have to handle VK_NOT_READY
        return self.__fpAcquireNextImageKHR(self.__device, self.swapChain, 0xffffffffffffffff, presentCompleteSemaphore, VK_NULL_HANDLE)

    # Queue an image for presentation
    #
    # @param queue Presentation queue for presenting the image
    # @param imageIndex Index of the swapchain image to queue for presentation
    # @param waitSemaphore (Optional) Semaphore that is waited on before the image is presented (only used if != VK_NULL_HANDLE)
    #
    # @return VkResult of the queue presentation
    def queuePresent(self, queue, imageIndex, waitSemaphore=VK_NULL_HANDLE):
        presentInfo = VkPresentInfoKHR(
            swapchainCount=1,
            pSwapchains=[self.swapChain],
            pImageIndices=[imageIndex]
        )
        # Check if a wait semaphore has been specified to wait for before presenting the image
        if waitSemaphore != VK_NULL_HANDLE:
            semaphores = ffi.new('VkSemaphore[]', [waitSemaphore])
            presentInfo.pWaitSemaphores = semaphores
            presentInfo.waitSemaphoreCount = 1

        self.__fpQueuePresentKHR(queue, presentInfo)
