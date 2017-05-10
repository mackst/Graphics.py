##########################################################################
# Vulkan buffer class port from Sascha Willems - www.saschawillems.de
#
# Copyright (C) 2017 by Shi Chi
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
##########################################################################

from pyVulkan import *


class Buffer(object):

    def __init__(self, device=None):
        self.buffer = None
        self.device = device
        self.memory = None
        self.descriptor = VkDescriptorBufferInfo()
        self.size = 0
        self.alignment = 0
        self.mapped = VK_NULL_HANDLE

        self.usageFlags = 0
        self.memoryPropertyFlags = 0


    def map(self, size=18446744073709551615, offset=0):
        self.mapped = vkMapMemory(self.device, self.memory, offset, size, 0)
        return self.mapped

    def unmap(self):
        if self.mapped:
            vkUnmapMemory(self.device, self.memory)
            self.mapped = VK_NULL_HANDLE

    def bind(self, offset=0):
        vkBindBufferMemory(self.device, self.buffer, self.memory, offset)

    def setupDescriptor(self, size=18446744073709551615, offset=0):
        self.descriptor.offset = offset
        self.descriptor.buffer = self.buffer
        self.descriptor.range = size

    def copyTo(self, data, size):
        if self.mapped:
            ffi.memmove(self.mapped, data, size)

    def flush(self, size, offset=0):
        mappedRange = VkMappedMemoryRange(
            memory=self.memory,
            offset=offset,
            size=size
        )

        vkFlushMappedMemoryRanges(self.device, 1, [mappedRange])

    def invalidate(self, size, offset=0):
        mappedRange = VkMappedMemoryRange(
            memory=self.memory,
            offset=offset,
            size=size
        )

        vkInvalidateMappedMemoryRanges(self.device, 1, [mappedRange])

    def __del__(self):
        if self.mapped:
            self.mapped = VK_NULL_HANDLE

        if self.buffer:
            vkDestroyBuffer(self.device, self.buffer, None)

        if self.memory:
            vkFreeMemory(self.device, self.memory, None)