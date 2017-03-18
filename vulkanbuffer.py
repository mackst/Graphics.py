from pyVulkan import *


class _Buffer(object):

    def __init__(self):
        self.buffer = None
        self.device = None
        self.memory = None
        self.descriptor = VkDescriptorBufferInfo()
        self.size = 0
        self.alignment = 0
        self.mapped = VK_NULL_HANDLE

        self.usageFlages = 0
        self.memoryPropertyFlags = 0


    def map(self, size, offset=0):
        return vkMapMemory(self.device, self.memory, offset, size, 0)

    def unmap(self):
        if self.mapped:
            vkUnmapMemory(self.device, self.memory)
            self.mapped = VK_NULL_HANDLE

    def bind(self, offset=0):
        vkBindBufferMemory(self.device, self.buffer, self.memory, offset)

    def setupDescriptor(self, size, offset=0):
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
        if self.buffer:
            vkDestroyBuffer(self.device, self.buffer, None)

        if self.memory:
            vkFreeMemory(self.device, self.memory, None)