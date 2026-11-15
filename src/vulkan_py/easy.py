"""Thin, high-level wrapper around vulkan_py for concise GPU compute."""

from __future__ import annotations

from ctypes import (
    byref,
    c_float,
    c_ubyte,
    c_uint32,
    c_void_p,
    cast,
    create_string_buffer,
    memmove,
    pointer,
    sizeof,
)
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

from . import (
    Device,
    Instance,
    VK_WHOLE_SIZE,
    VkApplicationInfo,
    VkBuffer,
    VkBufferCreateInfo,
    VkBufferUsageFlagBits,
    VkCommandBuffer,
    VkCommandBufferAllocateInfo,
    VkCommandBufferBeginInfo,
    VkCommandBufferLevel,
    VkCommandBufferUsageFlagBits,
    VkCommandPool,
    VkCommandPoolCreateInfo,
    VkComputePipelineCreateInfo,
    VkDescriptorBufferInfo,
    VkDescriptorPool,
    VkDescriptorPoolCreateInfo,
    VkDescriptorPoolSize,
    VkDescriptorSet,
    VkDescriptorSetAllocateInfo,
    VkDescriptorSetLayout,
    VkDescriptorSetLayoutBinding,
    VkDescriptorSetLayoutCreateInfo,
    VkDescriptorType,
    VkDevice,
    VkDeviceCreateInfo,
    VkDeviceMemory,
    VkDeviceQueueCreateInfo,
    VkFence,
    VkInstance,
    VkInstanceCreateInfo,
    VkMemoryAllocateInfo,
    VkMemoryPropertyFlagBits,
    VkMemoryRequirements,
    VkPhysicalDevice,
    VkPhysicalDeviceMemoryProperties,
    VkPhysicalDeviceProperties,
    VkPipeline,
    VkPipelineBindPoint,
    VkPipelineCache,
    VkPipelineLayout,
    VkPipelineLayoutCreateInfo,
    VkPipelineShaderStageCreateInfo,
    VkPushConstantRange,
    VkQueue,
    VkQueueFamilyProperties,
    VkQueueFlagBits,
    VkResult,
    VkShaderModule,
    VkShaderModuleCreateInfo,
    VkShaderStageFlagBits,
    VkSharingMode,
    VkSubmitInfo,
    VkWriteDescriptorSet,
    vkCreateInstance,
)


class VkError(RuntimeError):
    """Raised when a Vulkan API call returns a non-success VkResult."""


def _check(result: int, call_name: str) -> None:
    if result != VkResult.VK_SUCCESS:
        raise VkError(f"{call_name} failed with VkResult={result}")


def _compile_glsl(source: str) -> bytes:
    """Compile a GLSL compute shader to SPIR-V bytes using glslc."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "shader.comp"
        spv = Path(tmpdir) / "shader.spv"
        src.write_text(source, encoding="utf-8")
        subprocess.run(
            ["glslc", str(src), "-o", str(spv)],
            check=True,
            capture_output=True,
            text=True,
        )
        return spv.read_bytes()


@dataclass
class Buffer:
    """GPU storage buffer paired with its host-visible device memory."""

    handle: VkBuffer
    memory: VkDeviceMemory
    size: int


@dataclass
class ShaderModule:
    """Compiled SPIR-V shader module."""

    handle: VkShaderModule


@dataclass
class ComputePipeline:
    """Compute pipeline with its pipeline and descriptor set layouts."""

    pipeline: VkPipeline
    layout: VkPipelineLayout
    descriptor_layout: VkDescriptorSetLayout
    push_constant_size: int


@dataclass
class BindGroup:
    """Allocated and populated Vulkan descriptor set."""

    descriptor_set: VkDescriptorSet


class SimpleDevice:
    """High-level wrapper around a Vulkan compute device."""

    def __init__(self) -> None:
        app_info = VkApplicationInfo(
            pApplicationName=b"vk_easy",
            applicationVersion=1,
            pEngineName=b"vk_easy",
            engineVersion=1,
            apiVersion=(1 << 22) | (3 << 12),
        )
        raw_instance = VkInstance()
        _check(
            vkCreateInstance(
                byref(VkInstanceCreateInfo(pApplicationInfo=pointer(app_info))),
                None,
                byref(raw_instance),
            ),
            "vkCreateInstance",
        )
        self._instance = Instance(raw_instance)
        self._device: Device | None = None
        self._cmd_pool = VkCommandPool()
        self._desc_pools: list[VkDescriptorPool] = []
        self._buffers: list[Buffer] = []
        self._shaders: list[ShaderModule] = []
        self._pipelines: list[ComputePipeline] = []

        phys_dev = self._pick_physical_device()
        props = VkPhysicalDeviceProperties()
        self._instance.vkGetPhysicalDeviceProperties(phys_dev, props)
        self.device_name = props.deviceName.decode("utf-8", errors="replace")

        self._mem_props = VkPhysicalDeviceMemoryProperties()
        self._instance.vkGetPhysicalDeviceMemoryProperties(phys_dev, self._mem_props)

        self._queue_family = self._find_compute_queue_family(phys_dev)
        priority = c_float(1.0)
        queue_info = VkDeviceQueueCreateInfo(
            queueFamilyIndex=self._queue_family,
            queueCount=1,
            pQueuePriorities=pointer(priority),
        )
        raw_device = VkDevice()
        _check(
            self._instance.vkCreateDevice(
                phys_dev,
                VkDeviceCreateInfo(
                    queueCreateInfoCount=1,
                    pQueueCreateInfos=pointer(queue_info),
                ),
                None,
                raw_device,
            ),
            "vkCreateDevice",
        )
        self._device = Device(raw_device)

        self._queue = VkQueue()
        self._device.vkGetDeviceQueue(self._queue_family, 0, self._queue)

        _check(
            self._device.vkCreateCommandPool(
                VkCommandPoolCreateInfo(queueFamilyIndex=self._queue_family),
                None,
                self._cmd_pool,
            ),
            "vkCreateCommandPool",
        )

    def __enter__(self) -> SimpleDevice:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _pick_physical_device(self) -> VkPhysicalDevice:
        count = c_uint32(0)
        _check(
            self._instance.vkEnumeratePhysicalDevices(count, None),
            "vkEnumeratePhysicalDevices",
        )
        if count.value == 0:
            raise VkError("No Vulkan physical devices found")
        devs = (VkPhysicalDevice * count.value)()
        _check(
            self._instance.vkEnumeratePhysicalDevices(count, devs),
            "vkEnumeratePhysicalDevices",
        )
        return devs[0]

    def _find_compute_queue_family(self, phys_dev: VkPhysicalDevice) -> int:
        count = c_uint32(0)
        self._instance.vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, count, None)
        families = (VkQueueFamilyProperties * count.value)()
        self._instance.vkGetPhysicalDeviceQueueFamilyProperties(
            phys_dev, count, families
        )
        for idx, fam in enumerate(families):
            if fam.queueFlags & VkQueueFlagBits.VK_QUEUE_COMPUTE_BIT:
                return idx
        raise VkError("No compute queue family found")

    def _find_memory_type(self, type_filter: int, flags: int) -> int:
        for idx in range(self._mem_props.memoryTypeCount):
            if (type_filter & (1 << idx)) and (
                (self._mem_props.memoryTypes[idx].propertyFlags & flags) == flags
            ):
                return idx
        raise VkError("No suitable host-visible memory type found")

    def create_buffer(self, size: int) -> Buffer:
        """Create a host-visible, host-coherent storage buffer."""
        assert self._device is not None
        buf = VkBuffer()
        _check(
            self._device.vkCreateBuffer(
                VkBufferCreateInfo(
                    size=size,
                    usage=VkBufferUsageFlagBits.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    sharingMode=VkSharingMode.VK_SHARING_MODE_EXCLUSIVE,
                ),
                None,
                buf,
            ),
            "vkCreateBuffer",
        )
        reqs = VkMemoryRequirements()
        self._device.vkGetBufferMemoryRequirements(buf, reqs)

        flags = (
            VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        mem = VkDeviceMemory()
        _check(
            self._device.vkAllocateMemory(
                VkMemoryAllocateInfo(
                    allocationSize=reqs.size,
                    memoryTypeIndex=self._find_memory_type(reqs.memoryTypeBits, flags),
                ),
                None,
                mem,
            ),
            "vkAllocateMemory",
        )
        _check(self._device.vkBindBufferMemory(buf, mem, 0), "vkBindBufferMemory")
        wrapped = Buffer(handle=buf, memory=mem, size=size)
        self._buffers.append(wrapped)
        return wrapped

    def create_buffer_with_data(self, data) -> Buffer:
        """Create a storage buffer initialized from any buffer-protocol object."""
        assert self._device is not None
        raw_bytes = memoryview(data).tobytes()
        buf = self.create_buffer(len(raw_bytes))
        mapped = c_void_p()
        _check(
            self._device.vkMapMemory(buf.memory, 0, buf.size, 0, mapped),
            "vkMapMemory",
        )
        src = (c_ubyte * buf.size).from_buffer_copy(raw_bytes)
        memmove(mapped.value, src, buf.size)
        self._device.vkUnmapMemory(buf.memory)
        return buf

    def read_buffer(self, buf: Buffer) -> bytes:
        """Read back the full contents of a storage buffer as bytes."""
        assert self._device is not None
        mapped = c_void_p()
        _check(
            self._device.vkMapMemory(buf.memory, 0, buf.size, 0, mapped),
            "vkMapMemory",
        )
        out = (c_ubyte * buf.size)()
        memmove(out, mapped.value, buf.size)
        self._device.vkUnmapMemory(buf.memory)
        return bytes(out)

    def create_shader_module(self, code: str) -> ShaderModule:
        """Compile GLSL compute shader source and create a VkShaderModule."""
        assert self._device is not None
        spv_bytes = _compile_glsl(code)
        spv_words = (c_uint32 * (len(spv_bytes) // 4)).from_buffer_copy(spv_bytes)
        module = VkShaderModule()
        _check(
            self._device.vkCreateShaderModule(
                VkShaderModuleCreateInfo(codeSize=len(spv_bytes), pCode=spv_words),
                None,
                module,
            ),
            "vkCreateShaderModule",
        )
        wrapped = ShaderModule(handle=module)
        self._shaders.append(wrapped)
        return wrapped

    def create_compute_pipeline(
        self,
        shader: ShaderModule,
        num_bindings: int,
        push_constant_size: int = 0,
        entry_point: str = "main",
    ) -> ComputePipeline:
        """Create a compute pipeline with `num_bindings` storage buffer bindings."""
        assert self._device is not None
        bindings = (VkDescriptorSetLayoutBinding * num_bindings)(
            *[
                VkDescriptorSetLayoutBinding(
                    binding=i,
                    descriptorType=VkDescriptorType.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1,
                    stageFlags=VkShaderStageFlagBits.VK_SHADER_STAGE_COMPUTE_BIT,
                )
                for i in range(num_bindings)
            ]
        )
        desc_layout = VkDescriptorSetLayout()
        _check(
            self._device.vkCreateDescriptorSetLayout(
                VkDescriptorSetLayoutCreateInfo(
                    bindingCount=num_bindings, pBindings=bindings
                ),
                None,
                desc_layout,
            ),
            "vkCreateDescriptorSetLayout",
        )

        push_range = None
        if push_constant_size > 0:
            push_range = VkPushConstantRange(
                stageFlags=VkShaderStageFlagBits.VK_SHADER_STAGE_COMPUTE_BIT,
                offset=0,
                size=push_constant_size,
            )

        pipeline_layout = VkPipelineLayout()
        _check(
            self._device.vkCreatePipelineLayout(
                VkPipelineLayoutCreateInfo(
                    setLayoutCount=1,
                    pSetLayouts=pointer(desc_layout),
                    pushConstantRangeCount=1 if push_range else 0,
                    pPushConstantRanges=pointer(push_range) if push_range else None,
                ),
                None,
                pipeline_layout,
            ),
            "vkCreatePipelineLayout",
        )

        entry_buf = create_string_buffer(entry_point.encode("utf-8"))
        stage = VkPipelineShaderStageCreateInfo(
            stage=VkShaderStageFlagBits.VK_SHADER_STAGE_COMPUTE_BIT,
            module=shader.handle,
            pName=cast(entry_buf, c_void_p).value,
        )
        pipeline = VkPipeline()
        _check(
            self._device.vkCreateComputePipelines(
                VkPipelineCache(),
                1,
                VkComputePipelineCreateInfo(stage=stage, layout=pipeline_layout),
                None,
                pipeline,
            ),
            "vkCreateComputePipelines",
        )
        wrapped = ComputePipeline(
            pipeline=pipeline,
            layout=pipeline_layout,
            descriptor_layout=desc_layout,
            push_constant_size=push_constant_size,
        )
        self._pipelines.append(wrapped)
        return wrapped

    def create_bind_group(
        self, pipeline: ComputePipeline, buffers: list[Buffer]
    ) -> BindGroup:
        """Allocate and update a descriptor set binding `buffers` to 0..N-1."""
        assert self._device is not None
        pool_size = VkDescriptorPoolSize(
            type=VkDescriptorType.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=len(buffers),
        )
        pool = VkDescriptorPool()
        _check(
            self._device.vkCreateDescriptorPool(
                VkDescriptorPoolCreateInfo(
                    maxSets=1, poolSizeCount=1, pPoolSizes=pointer(pool_size)
                ),
                None,
                pool,
            ),
            "vkCreateDescriptorPool",
        )
        self._desc_pools.append(pool)

        desc_set = VkDescriptorSet()
        _check(
            self._device.vkAllocateDescriptorSets(
                VkDescriptorSetAllocateInfo(
                    descriptorPool=pool,
                    descriptorSetCount=1,
                    pSetLayouts=pointer(pipeline.descriptor_layout),
                ),
                desc_set,
            ),
            "vkAllocateDescriptorSets",
        )

        buf_infos = (VkDescriptorBufferInfo * len(buffers))(
            *[
                VkDescriptorBufferInfo(buffer=b.handle, offset=0, range=VK_WHOLE_SIZE)
                for b in buffers
            ]
        )
        writes = (VkWriteDescriptorSet * len(buffers))(
            *[
                VkWriteDescriptorSet(
                    dstSet=desc_set,
                    dstBinding=i,
                    descriptorCount=1,
                    descriptorType=VkDescriptorType.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=pointer(buf_infos[i]),
                )
                for i in range(len(buffers))
            ]
        )
        self._device.vkUpdateDescriptorSets(len(buffers), writes, 0, None)
        return BindGroup(descriptor_set=desc_set)

    def dispatch(
        self,
        pipeline: ComputePipeline,
        bind_group: BindGroup,
        workgroups: tuple[int, int, int],
        push_constants: bytes | None = None,
    ) -> None:
        """Record, submit, and wait for a single compute dispatch."""
        assert self._device is not None
        cmd_buf = VkCommandBuffer()
        _check(
            self._device.vkAllocateCommandBuffers(
                VkCommandBufferAllocateInfo(
                    commandPool=self._cmd_pool,
                    level=VkCommandBufferLevel.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=1,
                ),
                cmd_buf,
            ),
            "vkAllocateCommandBuffers",
        )
        _check(
            self._device.vkBeginCommandBuffer(
                cmd_buf,
                VkCommandBufferBeginInfo(
                    flags=VkCommandBufferUsageFlagBits.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                ),
            ),
            "vkBeginCommandBuffer",
        )
        self._device.vkCmdBindPipeline(
            cmd_buf, VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline
        )
        desc_sets = (VkDescriptorSet * 1)(bind_group.descriptor_set)
        self._device.vkCmdBindDescriptorSets(
            cmd_buf,
            VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout,
            0,
            1,
            desc_sets,
            0,
            None,
        )
        if push_constants is not None:
            pc_buf = (c_ubyte * len(push_constants)).from_buffer_copy(push_constants)
            self._device.vkCmdPushConstants(
                cmd_buf,
                pipeline.layout,
                VkShaderStageFlagBits.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_constants),
                pc_buf,
            )

        self._device.vkCmdDispatch(cmd_buf, *workgroups)
        _check(self._device.vkEndCommandBuffer(cmd_buf), "vkEndCommandBuffer")

        _check(
            self._device.vkQueueSubmit(
                self._queue,
                1,
                VkSubmitInfo(commandBufferCount=1, pCommandBuffers=pointer(cmd_buf)),
                VkFence(),
            ),
            "vkQueueSubmit",
        )
        _check(self._device.vkQueueWaitIdle(self._queue), "vkQueueWaitIdle")

    def close(self) -> None:
        """Destroy all tracked Vulkan resources."""
        if self._device is not None:
            if self._cmd_pool:
                self._device.vkDestroyCommandPool(self._cmd_pool, None)
            for pool in self._desc_pools:
                self._device.vkDestroyDescriptorPool(pool, None)
            for pipe in self._pipelines:
                self._device.vkDestroyPipeline(pipe.pipeline, None)
                self._device.vkDestroyPipelineLayout(pipe.layout, None)
                self._device.vkDestroyDescriptorSetLayout(pipe.descriptor_layout, None)
            for shader in self._shaders:
                self._device.vkDestroyShaderModule(shader.handle, None)
            for buf in self._buffers:
                self._device.vkDestroyBuffer(buf.handle, None)
                self._device.vkFreeMemory(buf.memory, None)
            self._device.vkDestroyDevice(None)
            self._device = None
        self._instance.vkDestroyInstance(None)


def get_default_device() -> SimpleDevice:
    """Create and return a default Vulkan compute device."""
    return SimpleDevice()
