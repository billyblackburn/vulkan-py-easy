"""Thin, high-level wrapper around vulkan_py for concise GPU compute and graphics."""

from __future__ import annotations

from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_float,
    c_int32,
    c_ubyte,
    c_uint32,
    c_uint64,
    c_void_p,
    cast,
    create_string_buffer,
    memmove,
    pointer,
)
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

from . import (
    Device,
    Instance,
    PFN_vkDebugUtilsMessengerCallbackEXT,
    VK_WHOLE_SIZE,
    VkApplicationInfo,
    VkBuffer,
    VkBufferCreateInfo,
    VkBufferUsageFlagBits,
    VkColorSpaceKHR,
    VkCommandBuffer,
    VkCommandBufferAllocateInfo,
    VkCommandBufferBeginInfo,
    VkCommandBufferLevel,
    VkCommandBufferUsageFlagBits,
    VkCommandPool,
    VkCommandPoolCreateFlagBits,
    VkCommandPoolCreateInfo,
    VkComponentMapping,
    VkComponentSwizzle,
    VkCompositeAlphaFlagBitsKHR,
    VkComputePipelineCreateInfo,
    VkDebugUtilsMessageSeverityFlagBitsEXT,
    VkDebugUtilsMessageTypeFlagBitsEXT,
    VkDebugUtilsMessengerCreateInfoEXT,
    VkDebugUtilsMessengerEXT,
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
    VkExtent2D,
    VkFence,
    VkFenceCreateFlagBits,
    VkFenceCreateInfo,
    VkFormat,
    VkImage,
    VkImageAspectFlagBits,
    VkImageSubresourceRange,
    VkImageUsageFlagBits,
    VkImageView,
    VkImageViewCreateInfo,
    VkImageViewType,
    VkInstance,
    VkInstanceCreateInfo,
    VkMemoryAllocateInfo,
    VkMemoryPropertyFlagBits,
    VkMemoryRequirements,
    VkPhysicalDevice,
    VkPhysicalDeviceFeatures,
    VkPhysicalDeviceMemoryProperties,
    VkPhysicalDeviceProperties,
    VkPhysicalDeviceType,
    VkPipeline,
    VkPipelineBindPoint,
    VkPipelineCache,
    VkPipelineLayout,
    VkPipelineLayoutCreateInfo,
    VkPipelineShaderStageCreateInfo,
    VkPipelineStageFlagBits,
    VkPresentInfoKHR,
    VkPresentModeKHR,
    VkPushConstantRange,
    VkQueue,
    VkQueueFamilyProperties,
    VkQueueFlagBits,
    VkResult,
    VkSemaphore,
    VkSemaphoreCreateInfo,
    VkShaderModule,
    VkShaderModuleCreateInfo,
    VkShaderStageFlagBits,
    VkSharingMode,
    VkSubmitInfo,
    VkSurfaceCapabilitiesKHR,
    VkSurfaceFormatKHR,
    VkSurfaceKHR,
    VkSwapchainCreateInfoKHR,
    VkSwapchainKHR,
    VkWriteDescriptorSet,
    vkCreateInstance,
)

UINT64_MAX = 0xFFFFFFFFFFFFFFFF


class VkError(RuntimeError):
    """Raised when a Vulkan API call returns a non-success VkResult."""


def check_vk(result: int, call_name: str) -> None:
    """Raise VkError if `result` is not VK_SUCCESS."""
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


def find_memory_type(
    mem_props: VkPhysicalDeviceMemoryProperties, type_filter: int, flags: int
) -> int:
    """Return a memory type index matching `type_filter` and `flags`."""
    for idx in range(mem_props.memoryTypeCount):
        if (type_filter & (1 << idx)) and (
            (mem_props.memoryTypes[idx].propertyFlags & flags) == flags
        ):
            return idx
    raise VkError("No suitable memory type found")


def _create_shader_module_from_spv(device: Device, spv_bytes: bytes) -> ShaderModule:
    spv_words = (c_uint32 * (len(spv_bytes) // 4)).from_buffer_copy(spv_bytes)
    module = VkShaderModule()
    check_vk(
        device.vkCreateShaderModule(
            VkShaderModuleCreateInfo(
                codeSize=len(spv_bytes),
                pCode=cast(spv_words, POINTER(c_uint32)),
            ),
            None,
            byref(module),
        ),
        "vkCreateShaderModule",
    )
    return ShaderModule(handle=module)


def _create_buffer_with_memory(
    device: Device,
    mem_props: VkPhysicalDeviceMemoryProperties,
    size: int,
    usage: int,
    properties: int,
) -> Buffer:
    buf = VkBuffer()
    check_vk(
        device.vkCreateBuffer(
            VkBufferCreateInfo(
                size=size,
                usage=usage,
                sharingMode=VkSharingMode.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
            byref(buf),
        ),
        "vkCreateBuffer",
    )
    reqs = VkMemoryRequirements()
    device.vkGetBufferMemoryRequirements(buf, byref(reqs))
    mem = VkDeviceMemory()
    check_vk(
        device.vkAllocateMemory(
            VkMemoryAllocateInfo(
                allocationSize=reqs.size,
                memoryTypeIndex=find_memory_type(
                    mem_props, reqs.memoryTypeBits, properties
                ),
            ),
            None,
            byref(mem),
        ),
        "vkAllocateMemory",
    )
    check_vk(device.vkBindBufferMemory(buf, mem, 0), "vkBindBufferMemory")
    return Buffer(handle=buf, memory=mem, size=size)


def _upload_memory(device: Device, memory: VkDeviceMemory, data, size: int) -> None:
    mapped = c_void_p()
    check_vk(device.vkMapMemory(memory, 0, size, 0, byref(mapped)), "vkMapMemory")
    src = (c_ubyte * size).from_buffer_copy(memoryview(data).tobytes())
    memmove(mapped.value, src, size)
    device.vkUnmapMemory(memory)


@dataclass
class Buffer:
    """GPU buffer paired with its device memory."""

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


@dataclass
class FrameSync:
    """Per-frame semaphores and fences for swapchain rendering."""

    image_available: list[VkSemaphore]
    render_finished: list[VkSemaphore]
    in_flight: list[VkFence]


class SimpleDevice:
    """High-level wrapper around a headless Vulkan compute device."""

    def __init__(self) -> None:
        app_info = VkApplicationInfo(
            pApplicationName=b"vk_easy",
            applicationVersion=1,
            pEngineName=b"vk_easy",
            engineVersion=1,
            apiVersion=(1 << 22) | (3 << 12),
        )
        raw_instance = VkInstance()
        check_vk(
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
        check_vk(
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

        check_vk(
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
        check_vk(
            self._instance.vkEnumeratePhysicalDevices(count, None),
            "vkEnumeratePhysicalDevices",
        )
        if count.value == 0:
            raise VkError("No Vulkan physical devices found")
        devs = (VkPhysicalDevice * count.value)()
        check_vk(
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

    def create_buffer(self, size: int) -> Buffer:
        """Create a host-visible, host-coherent storage buffer."""
        assert self._device is not None
        flags = (
            VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        wrapped = _create_buffer_with_memory(
            self._device,
            self._mem_props,
            size,
            VkBufferUsageFlagBits.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            flags,
        )
        self._buffers.append(wrapped)
        return wrapped

    def create_buffer_with_data(self, data) -> Buffer:
        """Create a storage buffer initialized from any buffer-protocol object."""
        assert self._device is not None
        raw_bytes = memoryview(data).tobytes()
        buf = self.create_buffer(len(raw_bytes))
        _upload_memory(self._device, buf.memory, raw_bytes, buf.size)
        return buf

    def read_buffer(self, buf: Buffer) -> bytes:
        """Read back the full contents of a storage buffer as bytes."""
        assert self._device is not None
        mapped = c_void_p()
        check_vk(
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
        wrapped = _create_shader_module_from_spv(self._device, _compile_glsl(code))
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
        check_vk(
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
        check_vk(
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
        check_vk(
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
        check_vk(
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
        check_vk(
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
        check_vk(
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
        check_vk(
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
        check_vk(self._device.vkEndCommandBuffer(cmd_buf), "vkEndCommandBuffer")

        check_vk(
            self._device.vkQueueSubmit(
                self._queue,
                1,
                VkSubmitInfo(commandBufferCount=1, pCommandBuffers=pointer(cmd_buf)),
                VkFence(),
            ),
            "vkQueueSubmit",
        )
        check_vk(self._device.vkQueueWaitIdle(self._queue), "vkQueueWaitIdle")

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


class GraphicsDevice:
    """High-level wrapper for windowed Vulkan graphics + swapchain setup."""

    def __init__(
        self,
        window,
        app_name: str = "vulkan_py",
        validation: bool = True,
        preferred_format: int = VkFormat.VK_FORMAT_B8G8R8A8_UNORM,
    ) -> None:
        import glfw

        self._glfw = glfw
        self.window = window
        self.debug_messenger = None
        self._debug_callback_fn = None
        self._buffers: list[Buffer] = []

        app_info = VkApplicationInfo(
            pApplicationName=app_name.encode("utf-8"),
            applicationVersion=1,
            pEngineName=b"vulkan_py",
            engineVersion=1,
            apiVersion=(1 << 22),
        )
        required_ext = glfw.get_required_instance_extensions()
        ext_bytes = [
            e.encode("utf-8") if isinstance(e, str) else e for e in required_ext
        ]
        if validation:
            ext_bytes.append(b"VK_EXT_debug_utils")
        ext_arr = (c_char_p * len(ext_bytes))(*ext_bytes)

        layer_bytes = [b"VK_LAYER_KHRONOS_validation"] if validation else []
        layer_arr = (c_char_p * len(layer_bytes))(*layer_bytes) if layer_bytes else None

        create_info = VkInstanceCreateInfo(
            pApplicationInfo=pointer(app_info),
            enabledExtensionCount=len(ext_bytes),
            ppEnabledExtensionNames=cast(ext_arr, POINTER(c_char_p)),
            enabledLayerCount=len(layer_bytes),
            ppEnabledLayerNames=cast(layer_arr, POINTER(c_char_p))
            if layer_arr is not None
            else None,
        )

        raw_instance = VkInstance()
        result = vkCreateInstance(byref(create_info), None, byref(raw_instance))
        if result != VkResult.VK_SUCCESS and validation:
            print(
                f"[!] Instance creation with validation failed ({result}), retrying without it"
            )
            create_info.enabledLayerCount = 0
            create_info.ppEnabledLayerNames = None
            create_info.enabledExtensionCount -= 1
            raw_instance = VkInstance()
            result = vkCreateInstance(byref(create_info), None, byref(raw_instance))
            validation = False

        check_vk(result, "vkCreateInstance")
        self.raw_instance = raw_instance
        self.instance = Instance(raw_instance)

        if validation:
            self._init_debug_messenger()

        surface_handle = c_void_p(0)
        surf_res = glfw.create_window_surface(
            self.raw_instance.value, self.window, None, byref(surface_handle)
        )
        if surf_res != 0:
            raise VkError(f"glfw.create_window_surface failed: {surf_res}")
        self.surface = VkSurfaceKHR(surface_handle.value)

        (
            self.physical_device,
            self.graphics_family,
            self.present_family,
            self.device_name,
        ) = self._pick_physical_device()
        print(
            f"Using GPU: {self.device_name} "
            f"(graphics family={self.graphics_family}, present family={self.present_family})"
        )

        self.mem_props = VkPhysicalDeviceMemoryProperties()
        self.instance.vkGetPhysicalDeviceMemoryProperties(
            self.physical_device, byref(self.mem_props)
        )

        self.raw_device, self.device, self.graphics_queue, self.present_queue = (
            self._create_logical_device()
        )

        self.swapchain = VkSwapchainKHR(0)
        self.swapchain_format = preferred_format
        self.swapchain_extent = VkExtent2D(width=0, height=0)
        self.swapchain_images: list[VkImage] = []
        self.swapchain_image_views: list[VkImageView] = []
        self.create_swapchain(preferred_format=preferred_format)

    def _init_debug_messenger(self) -> None:
        def callback(severity, _msg_type, callback_data_ptr, _user_data):
            data = callback_data_ptr.contents
            msg = data.pMessage.decode(errors="ignore") if data.pMessage else ""
            if (
                severity
                >= VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
            ):
                tag = "ERROR"
            elif (
                severity
                >= VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
            ):
                tag = "WARN"
            else:
                tag = "INFO"
            print(f"[validation:{tag}] {msg}")
            return 0

        self._debug_callback_fn = PFN_vkDebugUtilsMessengerCallbackEXT(callback)
        info = VkDebugUtilsMessengerCreateInfoEXT(
            messageSeverity=(
                VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                | VkDebugUtilsMessageSeverityFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT
            ),
            messageType=(
                VkDebugUtilsMessageTypeFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                | VkDebugUtilsMessageTypeFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                | VkDebugUtilsMessageTypeFlagBitsEXT.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT
            ),
            pfnUserCallback=self._debug_callback_fn,
        )
        messenger = VkDebugUtilsMessengerEXT()
        check_vk(
            self.instance.vkCreateDebugUtilsMessengerEXT(info, None, byref(messenger)),
            "vkCreateDebugUtilsMessengerEXT",
        )
        self.debug_messenger = messenger

    def _find_queue_families(self, physical_device: VkPhysicalDevice):
        count = c_uint32(0)
        self.instance.vkGetPhysicalDeviceQueueFamilyProperties(
            physical_device, byref(count), None
        )
        families = (VkQueueFamilyProperties * count.value)()
        self.instance.vkGetPhysicalDeviceQueueFamilyProperties(
            physical_device, byref(count), families
        )
        gfx = present = None
        for idx, fam in enumerate(families):
            if fam.queueFlags & VkQueueFlagBits.VK_QUEUE_GRAPHICS_BIT:
                gfx = idx
            supported = c_uint32(0)
            self.instance.vkGetPhysicalDeviceSurfaceSupportKHR(
                physical_device, idx, self.surface, byref(supported)
            )
            if supported.value:
                present = idx
            if gfx is not None and present is not None:
                break
        return gfx, present

    def _pick_physical_device(self):
        count = c_uint32(0)
        check_vk(
            self.instance.vkEnumeratePhysicalDevices(byref(count), None),
            "vkEnumeratePhysicalDevices",
        )
        if count.value == 0:
            raise VkError("No Vulkan-capable GPU found")
        devices = (VkPhysicalDevice * count.value)()
        check_vk(
            self.instance.vkEnumeratePhysicalDevices(byref(count), devices),
            "vkEnumeratePhysicalDevices",
        )
        best = None
        for pd in devices:
            gfx, present = self._find_queue_families(pd)
            if gfx is None or present is None:
                continue
            props = VkPhysicalDeviceProperties()
            self.instance.vkGetPhysicalDeviceProperties(pd, props)
            score = (
                2
                if props.deviceType
                == VkPhysicalDeviceType.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                else 1
            )
            name = bytes(props.deviceName).split(b"\0", 1)[0].decode(errors="replace")
            if best is None or score > best[0]:
                best = (score, pd, gfx, present, name)
        if best is None:
            raise VkError("No GPU with both graphics and present support found")
        return best[1], best[2], best[3], best[4]

    def _create_logical_device(self):
        unique = sorted({self.graphics_family, self.present_family})
        priority = c_float(1.0)
        queue_infos = [
            VkDeviceQueueCreateInfo(
                queueFamilyIndex=fam, queueCount=1, pQueuePriorities=pointer(priority)
            )
            for fam in unique
        ]
        queue_arr = (VkDeviceQueueCreateInfo * len(queue_infos))(*queue_infos)
        ext_arr = (c_char_p * 1)(b"VK_KHR_swapchain")
        features = VkPhysicalDeviceFeatures()
        info = VkDeviceCreateInfo(
            queueCreateInfoCount=len(queue_infos),
            pQueueCreateInfos=cast(queue_arr, POINTER(VkDeviceQueueCreateInfo)),
            enabledExtensionCount=1,
            ppEnabledExtensionNames=cast(ext_arr, POINTER(c_char_p)),
            pEnabledFeatures=pointer(features),
        )
        raw_device = VkDevice()
        check_vk(
            self.instance.vkCreateDevice(
                self.physical_device, byref(info), None, byref(raw_device)
            ),
            "vkCreateDevice",
        )
        device = Device(raw_device)
        gfx_q, pres_q = VkQueue(), VkQueue()
        device.vkGetDeviceQueue(self.graphics_family, 0, byref(gfx_q))
        device.vkGetDeviceQueue(self.present_family, 0, byref(pres_q))
        return raw_device, device, gfx_q, pres_q

    def create_swapchain(
        self, preferred_format: int = VkFormat.VK_FORMAT_B8G8R8A8_UNORM
    ) -> None:
        """Create swapchain and swapchain image views for the current window size."""
        caps = VkSurfaceCapabilitiesKHR()
        check_vk(
            self.instance.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                self.physical_device, self.surface, byref(caps)
            ),
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
        )

        fmt_count = c_uint32(0)
        self.instance.vkGetPhysicalDeviceSurfaceFormatsKHR(
            self.physical_device, self.surface, byref(fmt_count), None
        )
        formats = (VkSurfaceFormatKHR * fmt_count.value)()
        self.instance.vkGetPhysicalDeviceSurfaceFormatsKHR(
            self.physical_device, self.surface, byref(fmt_count), formats
        )
        chosen = formats[0]
        for fmt in formats:
            if (
                fmt.format == preferred_format
                and fmt.colorSpace == VkColorSpaceKHR.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
            ):
                chosen = fmt
                break

        mode_count = c_uint32(0)
        self.instance.vkGetPhysicalDeviceSurfacePresentModesKHR(
            self.physical_device, self.surface, byref(mode_count), None
        )
        modes = (c_int32 * mode_count.value)()
        self.instance.vkGetPhysicalDeviceSurfacePresentModesKHR(
            self.physical_device, self.surface, byref(mode_count), modes
        )
        present_mode = VkPresentModeKHR.VK_PRESENT_MODE_FIFO_KHR
        if VkPresentModeKHR.VK_PRESENT_MODE_MAILBOX_KHR in list(modes):
            present_mode = VkPresentModeKHR.VK_PRESENT_MODE_MAILBOX_KHR

        if caps.currentExtent.width != 0xFFFFFFFF:
            extent = caps.currentExtent
        else:
            fb_w, fb_h = self._glfw.get_framebuffer_size(self.window)
            extent = VkExtent2D(
                width=max(caps.minImageExtent.width, min(fb_w, caps.maxImageExtent.width)),
                height=max(
                    caps.minImageExtent.height, min(fb_h, caps.maxImageExtent.height)
                ),
            )

        image_count = caps.minImageCount + 1
        if caps.maxImageCount != 0 and image_count > caps.maxImageCount:
            image_count = caps.maxImageCount

        if self.graphics_family != self.present_family:
            indices = (c_uint32 * 2)(self.graphics_family, self.present_family)
            sharing, idx_count, idx_ptr = (
                VkSharingMode.VK_SHARING_MODE_CONCURRENT,
                2,
                cast(indices, POINTER(c_uint32)),
            )
        else:
            sharing, idx_count, idx_ptr = (
                VkSharingMode.VK_SHARING_MODE_EXCLUSIVE,
                0,
                None,
            )

        swapchain = VkSwapchainKHR()
        check_vk(
            self.device.vkCreateSwapchainKHR(
                VkSwapchainCreateInfoKHR(
                    surface=self.surface,
                    minImageCount=image_count,
                    imageFormat=chosen.format,
                    imageColorSpace=chosen.colorSpace,
                    imageExtent=extent,
                    imageArrayLayers=1,
                    imageUsage=VkImageUsageFlagBits.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    imageSharingMode=sharing,
                    queueFamilyIndexCount=idx_count,
                    pQueueFamilyIndices=idx_ptr,
                    preTransform=caps.currentTransform,
                    compositeAlpha=VkCompositeAlphaFlagBitsKHR.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                    presentMode=present_mode,
                    clipped=1,
                    oldSwapchain=VkSwapchainKHR(0),
                ),
                None,
                byref(swapchain),
            ),
            "vkCreateSwapchainKHR",
        )
        self.swapchain = swapchain
        self.swapchain_format = chosen.format
        self.swapchain_extent = extent

        img_count = c_uint32(0)
        self.device.vkGetSwapchainImagesKHR(self.swapchain, byref(img_count), None)
        images = (VkImage * img_count.value)()
        self.device.vkGetSwapchainImagesKHR(self.swapchain, byref(img_count), images)
        self.swapchain_images = list(images)

        self.swapchain_image_views = []
        for img in self.swapchain_images:
            view = VkImageView()
            check_vk(
                self.device.vkCreateImageView(
                    VkImageViewCreateInfo(
                        image=img,
                        viewType=VkImageViewType.VK_IMAGE_VIEW_TYPE_2D,
                        format=self.swapchain_format,
                        components=VkComponentMapping(
                            r=VkComponentSwizzle.VK_COMPONENT_SWIZZLE_IDENTITY,
                            g=VkComponentSwizzle.VK_COMPONENT_SWIZZLE_IDENTITY,
                            b=VkComponentSwizzle.VK_COMPONENT_SWIZZLE_IDENTITY,
                            a=VkComponentSwizzle.VK_COMPONENT_SWIZZLE_IDENTITY,
                        ),
                        subresourceRange=VkImageSubresourceRange(
                            aspectMask=VkImageAspectFlagBits.VK_IMAGE_ASPECT_COLOR_BIT,
                            baseMipLevel=0,
                            levelCount=1,
                            baseArrayLayer=0,
                            layerCount=1,
                        ),
                    ),
                    None,
                    byref(view),
                ),
                "vkCreateImageView",
            )
            self.swapchain_image_views.append(view)

    def destroy_swapchain(self) -> None:
        """Destroy swapchain image views and the swapchain itself."""
        for view in self.swapchain_image_views:
            self.device.vkDestroyImageView(view, None)
        self.swapchain_image_views.clear()
        if self.swapchain:
            self.device.vkDestroySwapchainKHR(self.swapchain, None)
            self.swapchain = VkSwapchainKHR(0)

    def load_shader_module(self, path: str | Path) -> ShaderModule:
        """Load a precompiled SPIR-V (.spv) file into a ShaderModule."""
        return _create_shader_module_from_spv(
            self.device, Path(path).read_bytes()
        )

    def find_memory_type(self, type_bits: int, required_flags: int) -> int:
        """Find a device memory type index matching `type_bits` and `required_flags`."""
        return find_memory_type(self.mem_props, type_bits, required_flags)

    def create_buffer(
        self,
        size: int,
        usage: int,
        properties: int = (
            VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        ),
    ) -> Buffer:
        """Create a Vulkan buffer + backing device memory."""
        buf = _create_buffer_with_memory(
            self.device, self.mem_props, size, usage, properties
        )
        self._buffers.append(buf)
        return buf

    def create_buffer_with_data(self, data, usage: int) -> Buffer:
        """Create a host-visible buffer initialized from `data`."""
        raw_bytes = memoryview(data).tobytes()
        buf = self.create_buffer(len(raw_bytes), usage)
        _upload_memory(self.device, buf.memory, raw_bytes, buf.size)
        return buf

    def upload_memory(self, memory: VkDeviceMemory, data, size: int) -> None:
        """Copy `size` bytes from `data` into host-visible `memory`."""
        _upload_memory(self.device, memory, data, size)

    def create_command_pool(
        self,
        flags: int = VkCommandPoolCreateFlagBits.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    ) -> VkCommandPool:
        """Create a graphics-queue command pool."""
        pool = VkCommandPool()
        check_vk(
            self.device.vkCreateCommandPool(
                VkCommandPoolCreateInfo(
                    flags=flags, queueFamilyIndex=self.graphics_family
                ),
                None,
                byref(pool),
            ),
            "vkCreateCommandPool",
        )
        return pool

    def allocate_command_buffers(
        self, command_pool: VkCommandPool, count: int
    ) -> list[VkCommandBuffer]:
        """Allocate `count` primary command buffers from `command_pool`."""
        bufs = (VkCommandBuffer * count)()
        check_vk(
            self.device.vkAllocateCommandBuffers(
                VkCommandBufferAllocateInfo(
                    commandPool=command_pool,
                    level=VkCommandBufferLevel.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=count,
                ),
                bufs,
            ),
            "vkAllocateCommandBuffers",
        )
        return list(bufs)

    def create_frame_sync(self, max_frames_in_flight: int) -> FrameSync:
        """Create semaphores and signaled fences for swapchain frames in flight."""
        sem_info = VkSemaphoreCreateInfo()
        fence_info = VkFenceCreateInfo(
            flags=VkFenceCreateFlagBits.VK_FENCE_CREATE_SIGNALED_BIT
        )
        image_available, in_flight = [], []
        for _ in range(max_frames_in_flight):
            sem, fence = VkSemaphore(), VkFence()
            check_vk(
                self.device.vkCreateSemaphore(sem_info, None, byref(sem)),
                "vkCreateSemaphore",
            )
            check_vk(
                self.device.vkCreateFence(fence_info, None, byref(fence)),
                "vkCreateFence",
            )
            image_available.append(sem)
            in_flight.append(fence)

        render_finished = []
        for _ in range(len(self.swapchain_images)):
            sem = VkSemaphore()
            check_vk(
                self.device.vkCreateSemaphore(sem_info, None, byref(sem)),
                "vkCreateSemaphore",
            )
            render_finished.append(sem)

        return FrameSync(
            image_available=image_available,
            render_finished=render_finished,
            in_flight=in_flight,
        )

    def destroy_frame_sync(self, sync: FrameSync) -> None:
        """Destroy all semaphores and fences in `sync`."""
        for sem in sync.image_available:
            self.device.vkDestroySemaphore(sem, None)
        for sem in sync.render_finished:
            self.device.vkDestroySemaphore(sem, None)
        for fence in sync.in_flight:
            self.device.vkDestroyFence(fence, None)

    def acquire_next_image(self, sync: FrameSync, frame_index: int) -> tuple[int, int]:
        """Wait on the frame fence and acquire the next swapchain image index."""
        fence = sync.in_flight[frame_index]
        self.device.vkWaitForFences(1, byref(fence), 1, UINT64_MAX)
        self.device.vkResetFences(1, byref(fence))
        image_index = c_uint32(0)
        res = self.device.vkAcquireNextImageKHR(
            self.swapchain,
            UINT64_MAX,
            sync.image_available[frame_index],
            VkFence(0),
            byref(image_index),
        )
        return res, image_index.value

    def submit_and_present(
        self,
        sync: FrameSync,
        frame_index: int,
        image_index: int,
        command_buffer: VkCommandBuffer,
    ) -> int:
        """Submit `command_buffer` and present `image_index`."""
        wait_sems = (VkSemaphore * 1)(sync.image_available[frame_index])
        signal_sems = (VkSemaphore * 1)(sync.render_finished[image_index])
        wait_stages = (c_uint32 * 1)(
            VkPipelineStageFlagBits.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        )
        cmd_bufs = (VkCommandBuffer * 1)(command_buffer)

        submit_info = VkSubmitInfo(
            waitSemaphoreCount=1,
            pWaitSemaphores=cast(wait_sems, POINTER(VkSemaphore)),
            pWaitDstStageMask=cast(wait_stages, POINTER(c_uint32)),
            commandBufferCount=1,
            pCommandBuffers=cast(cmd_bufs, POINTER(VkCommandBuffer)),
            signalSemaphoreCount=1,
            pSignalSemaphores=cast(signal_sems, POINTER(VkSemaphore)),
        )
        check_vk(
            self.device.vkQueueSubmit(
                self.graphics_queue, 1, byref(submit_info), sync.in_flight[frame_index]
            ),
            "vkQueueSubmit",
        )

        swapchains = (VkSwapchainKHR * 1)(self.swapchain)
        idx = c_uint32(image_index)
        present_info = VkPresentInfoKHR(
            waitSemaphoreCount=1,
            pWaitSemaphores=cast(signal_sems, POINTER(VkSemaphore)),
            swapchainCount=1,
            pSwapchains=cast(swapchains, POINTER(VkSwapchainKHR)),
            pImageIndices=pointer(idx),
        )
        return self.device.vkQueuePresentKHR(self.present_queue, byref(present_info))

    def close(self) -> None:
        """Destroy buffers, swapchain, device, surface, debug messenger, and instance."""
        if getattr(self, "device", None) is not None:
            for buf in self._buffers:
                self.device.vkDestroyBuffer(buf.handle, None)
                self.device.vkFreeMemory(buf.memory, None)
            self._buffers.clear()
            self.destroy_swapchain()
            self.device.vkDestroyDevice(None)
            self.device = None
        if getattr(self, "instance", None) is not None:
            if self.debug_messenger:
                self.instance.vkDestroyDebugUtilsMessengerEXT(
                    self.debug_messenger, None
                )
            if self.surface:
                self.instance.vkDestroySurfaceKHR(self.surface, None)
            self.instance.vkDestroyInstance(None)
            self.instance = None
