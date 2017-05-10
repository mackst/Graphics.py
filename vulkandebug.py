##########################################################################
# Vulkan examples debug wrapper port from Sascha Willems - www.saschawillems.de
#
# Copyright (C) 2017 by Shi Chi
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
##########################################################################

from pyVulkan import *

# Default validation layers
validationLayerNames = ["VK_LAYER_LUNARG_standard_validation"]

msgCallback = None

pfnDebugMarkerSetObjectTag = VK_NULL_HANDLE
pfnDebugMarkerSetObjectName = VK_NULL_HANDLE
pfnCmdDebugMarkerBegin = VK_NULL_HANDLE
pfnCmdDebugMarkerEnd = VK_NULL_HANDLE
pfnCmdDebugMarkerInsert = VK_NULL_HANDLE

# Default debug callback
@vkDebugReportCallbackEXT
def messageCallback(*args):
    print (ffi.string(args[6]))
    return True

# Load debug function pointers and set debug callback
# if callBack is NULL, default message callback will be used
def setupDebugging(instance, flags):
    global CreateDebugReportCallback
    global DestroyDebugReportCallback
    global dbgBreakCallback
    global msgCallback
    CreateDebugReportCallback = vkGetInstanceProcAddr(instance, 'vkCreateDebugReportCallbackEXT')
    DestroyDebugReportCallback = vkGetInstanceProcAddr(instance, 'vkDestroyDebugReportCallbackEXT')
    dbgBreakCallback = vkGetInstanceProcAddr(instance, 'vkDebugReportMessageEXT')

    dbgCreateInfo = VkDebugReportCallbackCreateInfoEXT(
        flags=flags,
        pfnCallback=messageCallback
    )

    msgCallback = CreateDebugReportCallback(instance, dbgCreateInfo, None)
    return msgCallback

# Clear debug callback
def freeDebugCallback(instance):
    global msgCallback
    if msgCallback:
        DestroyDebugReportCallback(instance, msgCallback, None)

class DebugMarker(object):

    # Set to true if function pointer for the debug marker are available
    active = False

    # Get function pointers for the debug report extensions from the device
    @staticmethod
    def setup(device):
        pfnDebugMarkerSetObjectTag = vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectTagEXT")
        pfnDebugMarkerSetObjectName = vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectNameEXT")
        pfnCmdDebugMarkerBegin = vkGetDeviceProcAddr(device, "vkCmdDebugMarkerBeginEXT")
        pfnCmdDebugMarkerEnd = vkGetDeviceProcAddr(device, "vkCmdDebugMarkerEndEXT")
        pfnCmdDebugMarkerInsert = vkGetDeviceProcAddr(device, "vkCmdDebugMarkerInsertEXT")

        # Set flag if at least one function pointer is present
        DebugMarker.active = pfnDebugMarkerSetObjectName != VK_NULL_HANDLE

    # Sets the debug name of an object
    # All Objects in Vulkan are represented by their 64-bit handles which are passed into this function
    # along with the object type
    @staticmethod
    def setObjectName(device, obj, objectType, name):
        # Check for valid function pointer (may not be present if not running in a debugging application)
        if pfnDebugMarkerSetObjectName:
            nameInfo = VkDebugMarkerObjectNameInfoEXT(
                objectType=objectType,
                object=obj,
                pObjectName=name
            )
            pfnDebugMarkerSetObjectName(device, nameInfo)

    # Set the tag for an object
    @staticmethod
    def setObjectTag(device, obj, objectType, name, tagSize, tag):
        # Check for valid function pointer (may not be present if not running in a debugging application)
        if pfnDebugMarkerSetObjectTag:
            tagInfo = VkDebugMarkerObjectTagInfoEXT(
                objectType=objectType,
                object=obj,
                tagName=name,
                tagSize=tagSize,
                pTag=tag
            )
            pfnDebugMarkerSetObjectTag(device, tagInfo)

    # Start a new debug marker region
    def beginRegion(cmdbuffer, pMarkerName, color):
        # Check for valid function pointer (may not be present if not running in a debugging application)
        if pfnCmdDebugMarkerBegin:
            markerInfo = VkDebugMarkerMarkerInfoEXT(
                color=color,
                pMarkerName=pMarkerName
            )
            pfnCmdDebugMarkerBegin(cmdbuffer, markerInfo)

    # Insert a new debug marker into the command buffer
    def insert(cmdbuffer, markerName, color):
        # Check for valid function pointer (may not be present if not running in a debugging application)
        if pfnCmdDebugMarkerInsert:
            markerInfo = VkDebugMarkerMarkerInfoEXT(
                color=color,
                pMarkerName=markerName
            )
            pfnCmdDebugMarkerInsert(cmdbuffer, markerInfo)

    # End the current debug marker region
    def endRegion(cmdBuffer):
        # Check for valid function (may not be present if not runnin in a debugging application)
        if pfnCmdDebugMarkerEnd:
            pfnCmdDebugMarkerEnd(cmdBuffer)

    # Object specific naming functions
    def setCommandBufferName(device, cmdBuffer, name):
        DebugMarker.setObjectName(device, cmdBuffer, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT, name)

    def setQueueName(device, queue, name):
        DebugMarker.setObjectName(device, queue, VK_DEBUG_REPORT_OBJECT_TYPE_QUEUE_EXT, name)

    def setImageName(device, image, name):
        DebugMarker.setObjectName(device, image, VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, name)
        
    def setSamplerName(device, sampler, name):
        DebugMarker.setObjectName(device, sampler, VK_DEBUG_REPORT_OBJECT_TYPE_SAMPLER_EXT, name)

    def setBufferName(device, buffer, name):
        DebugMarker.setObjectName(device, buffer, VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, name)

    def setDeviceMemoryName(device, memory, name):
        DebugMarker.setObjectName(device, memory, VK_DEBUG_REPORT_OBJECT_TYPE_DEVICE_MEMORY_EXT, name)

    def setShaderModuleName(device, shaderModule, name):
        DebugMarker.setObjectName(device, shaderModule, VK_DEBUG_REPORT_OBJECT_TYPE_SHADER_MODULE_EXT, name)

    def setPipelineName(device, pipeline, name):
        DebugMarker.setObjectName(device, pipeline, VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT, name)

    def setPipelineLayoutName(device, pipelineLayout, name):
        DebugMarker.setObjectName(device, pipelineLayout, VK_DEBUG_REPORT_OBJECT_TYPE_PIPELINE_LAYOUT_EXT, name)

    def setRenderPassName(device, renderPass, name):
        DebugMarker.setObjectName(device, renderPass, VK_DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT, name)

    def setFramebufferName(device, framebuffer, name):
        DebugMarker.setObjectName(device, framebuffer, VK_DEBUG_REPORT_OBJECT_TYPE_FRAMEBUFFER_EXT, name)

    def setDescriptorSetLayoutName(device, descriptorSetLayout, name):
        DebugMarker.setObjectName(device, descriptorSetLayout, VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT_EXT, name)

    def setDescriptorSetName(device, descriptorSet, name):
        DebugMarker.setObjectName(device, descriptorSet, VK_DEBUG_REPORT_OBJECT_TYPE_DESCRIPTOR_SET_EXT, name)

    def setSemaphoreName(device, semaphore, name):
        DebugMarker.setObjectName(device, semaphore, VK_DEBUG_REPORT_OBJECT_TYPE_SEMAPHORE_EXT, name)

    def setFenceName(device, fence, name):
        DebugMarker.setObjectName(device, fence, VK_DEBUG_REPORT_OBJECT_TYPE_FENCE_EXT, name)

    def setEventName(device, event, name):
        DebugMarker.setObjectName(device, event, VK_DEBUG_REPORT_OBJECT_TYPE_EVENT_EXT, name)
