import ctypes
import math
import os
from pathlib import Path
import time

import glfw
import vulkan_py as output
from vulkan_py.easy import GraphicsDevice, check_vk as check

from ctypes import (
    POINTER,
    byref,
    c_char_p,
    c_float,
    c_uint32,
    c_uint64,
    c_void_p,
    cast,
    pointer,
)


WIDTH = 1000
HEIGHT = 700
MAX_FRAMES_IN_FLIGHT = 2

VK_TRUE = 1
VK_FALSE = 0
UINT64_MAX = 0xFFFFFFFFFFFFFFFF


SHADERS_DIR = Path(__file__).resolve().parent / "shaders"


class Vertex(ctypes.Structure):
    _fields_ = [
        ("position", c_float * 3),
        ("normal", c_float * 3),
        ("color", c_float * 3),
    ]


class SceneUBO(ctypes.Structure):
    # std140-compatible layout:
    # mat4 view       64
    # mat4 projection 64
    # vec4 camera_pos 16
    # 4 * (vec4 position + vec4 color) = 128
    _fields_ = [
        ("view", c_float * 16),
        ("projection", c_float * 16),
        ("camera_pos", c_float * 4),
        ("light_pos", (c_float * 4) * 4),
        ("light_color", (c_float * 4) * 4),
    ]


class Mat4:
    @staticmethod
    def identity():
        return [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

    @staticmethod
    def multiply(a, b):
        # Column-major matrices: result = a * b.
        r = [0.0] * 16
        for col in range(4):
            for row in range(4):
                r[col * 4 + row] = (
                    a[0 * 4 + row] * b[col * 4 + 0]
                    + a[1 * 4 + row] * b[col * 4 + 1]
                    + a[2 * 4 + row] * b[col * 4 + 2]
                    + a[3 * 4 + row] * b[col * 4 + 3]
                )
        return r

    @staticmethod
    def translation(x, y, z):
        m = Mat4.identity()
        m[12] = x
        m[13] = y
        m[14] = z
        return m

    @staticmethod
    def scale(x, y, z):
        m = Mat4.identity()
        m[0] = x
        m[5] = y
        m[10] = z
        return m

    @staticmethod
    def rotation_y(angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return [
             c, 0.0, -s, 0.0,
             0.0, 1.0, 0.0, 0.0,
             s, 0.0,  c, 0.0,
             0.0, 0.0, 0.0, 1.0,
        ]

    @staticmethod
    def rotation_x(angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return [
            1.0, 0.0, 0.0, 0.0,
            0.0, c, s, 0.0,
            0.0, -s, c, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

    @staticmethod
    def perspective(fov_y, aspect, near, far):
        f = 1.0 / math.tan(fov_y * 0.5)
        # Vulkan depth range [0, 1], with Y inverted for the usual
        # framebuffer coordinate convention.
        return [
            f / aspect, 0.0, 0.0, 0.0,
            0.0, -f, 0.0, 0.0,
            0.0, 0.0, far / (near - far), -1.0,
            0.0, 0.0, (far * near) / (near - far), 0.0,
        ]

    @staticmethod
    def look_at(eye, target, up):
        def normalize(v):
            l = math.sqrt(sum(x * x for x in v))
            return [x / l for x in v]

        def cross(a, b):
            return [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]

        def dot(a, b):
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

        f = normalize([
            target[0] - eye[0],
            target[1] - eye[1],
            target[2] - eye[2],
        ])
        s = normalize(cross(f, up))
        u = cross(s, f)

        return [
            s[0], u[0], -f[0], 0.0,
            s[1], u[1], -f[1], 0.0,
            s[2], u[2], -f[2], 0.0,
            -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0,
        ]


CUBE_VERTICES = [
    # position          normal             color
    # front (-Z)
    ((-1,-1,-1), (0,0,-1), (0.85,0.18,0.18)),
    (( 1,-1,-1), (0,0,-1), (0.85,0.18,0.18)),
    (( 1, 1,-1), (0,0,-1), (0.85,0.18,0.18)),
    ((-1, 1,-1), (0,0,-1), (0.85,0.18,0.18)),
    # back (+Z)
    (( 1,-1, 1), (0,0,1), (0.18,0.55,0.95)),
    ((-1,-1, 1), (0,0,1), (0.18,0.55,0.95)),
    ((-1, 1, 1), (0,0,1), (0.18,0.55,0.95)),
    (( 1, 1, 1), (0,0,1), (0.18,0.55,0.95)),
    # left (-X)
    ((-1,-1, 1), (-1,0,0), (0.18,0.85,0.35)),
    ((-1,-1,-1), (-1,0,0), (0.18,0.85,0.35)),
    ((-1, 1,-1), (-1,0,0), (0.18,0.85,0.35)),
    ((-1, 1, 1), (-1,0,0), (0.18,0.85,0.35)),
    # right (+X)
    ((1,-1,-1), (1,0,0), (0.95,0.65,0.15)),
    ((1,-1, 1), (1,0,0), (0.95,0.65,0.15)),
    ((1, 1, 1), (1,0,0), (0.95,0.65,0.15)),
    ((1, 1,-1), (1,0,0), (0.95,0.65,0.15)),
    # top (+Y)
    ((-1,1,-1), (0,1,0), (0.75,0.25,0.90)),
    (( 1,1,-1), (0,1,0), (0.75,0.25,0.90)),
    (( 1,1, 1), (0,1,0), (0.75,0.25,0.90)),
    ((-1,1, 1), (0,1,0), (0.75,0.25,0.90)),
    # bottom (-Y)
    ((-1,-1, 1), (0,-1,0), (0.20,0.75,0.80)),
    (( 1,-1, 1), (0,-1,0), (0.20,0.75,0.80)),
    (( 1,-1,-1), (0,-1,0), (0.20,0.75,0.80)),
    ((-1,-1,-1), (0,-1,0), (0.20,0.75,0.80)),
]

CUBE_INDICES = [
     0, 1, 2,  2, 3, 0,
     4, 5, 6,  6, 7, 4,
     8, 9,10, 10,11, 8,
    12,13,14, 14,15,12,
    16,17,18, 18,19,16,
    20,21,22, 22,23,20,
]

PLANE_VERTICES = [
    ((-6, 0, -6), (0,1,0), (0.42,0.44,0.47)),
    (( 6, 0, -6), (0,1,0), (0.42,0.44,0.47)),
    (( 6, 0,  6), (0,1,0), (0.42,0.44,0.47)),
    ((-6, 0,  6), (0,1,0), (0.42,0.44,0.47)),
]
PLANE_INDICES = [0, 1, 2, 2, 3, 0]


def make_vertices(raw):
    return (Vertex * len(raw))(*[
        Vertex(position=p, normal=n, color=c)
        for p, n, c in raw
    ])


class App:
    def __init__(self):
        self.window = None
        self.instance = None
        self.raw_instance = None
        self.surface = None
        self.physical_device = None
        self.device = None
        self.raw_device = None
        self.graphics_family = None
        self.present_family = None
        self.graphics_queue = None
        self.present_queue = None

        self.swapchain = None
        self.swapchain_format = None
        self.swapchain_extent = None
        self.swapchain_images = []
        self.swapchain_views = []
        self.render_pass = None
        self.depth_format = None
        self.depth_image = None
        self.depth_memory = None
        self.depth_view = None
        self.framebuffers = []

        self.pipeline_layout = None
        self.pipeline = None
        self.command_pool = None
        self.command_buffers = []

        self.descriptor_set_layout = output.VkDescriptorSetLayout()
        self.descriptor_pool = None
        self.descriptor_sets = []
        self.uniform_buffers = []
        self.uniform_memories = []

        self.vertex_buffer = None
        self.vertex_memory = None
        self.index_buffer = None
        self.index_memory = None
        self.index_count = 0

        self.image_available = []
        self.render_finished = []
        self.in_flight = []
        self.current_frame = 0

        self.camera_pos = [4.5, 3.1, 7.0]
        self.camera_yaw = -math.radians(58.0)
        self.camera_pitch = -math.radians(12.0)

        self.camera_speed = 5.0
        self.mouse_sensitivity = 0.0025

        self.last_time = time.perf_counter()
        self.first_mouse = True
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

        self.fps_time = time.perf_counter()
        self.fps_frames = 0

        self.debug_callback_fn = None
        self.debug_messenger = output.VkDebugUtilsMessengerEXT(0)

        self.shader_buffers = []

    def update_camera(self):
        now = time.perf_counter()
        dt = min(now - self.last_time, 0.1)
        self.last_time = now

        # Mouse look
        x, y = glfw.get_cursor_pos(self.window)

        if self.first_mouse:
            self.last_mouse_x = x
            self.last_mouse_y = y
            self.first_mouse = False

        dx = x - self.last_mouse_x
        dy = y - self.last_mouse_y

        self.last_mouse_x = x
        self.last_mouse_y = y

        self.camera_yaw += dx * self.mouse_sensitivity
        self.camera_pitch -= dy * self.mouse_sensitivity

        limit = math.radians(89.0)
        self.camera_pitch = max(
            -limit,
            min(limit, self.camera_pitch)
        )

        # Camera basis
        yaw = self.camera_yaw
        pitch = self.camera_pitch

        forward = [
            math.cos(pitch) * math.cos(yaw),
            math.sin(pitch),
            math.cos(pitch) * math.sin(yaw),
        ]

        right = [
            -math.sin(yaw),
            0.0,
            math.cos(yaw),
        ]

        up = [0.0, 1.0, 0.0]

        speed = self.camera_speed * dt

        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            for i in range(3):
                self.camera_pos[i] += forward[i] * speed

        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            for i in range(3):
                self.camera_pos[i] -= forward[i] * speed

        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            for i in range(3):
                self.camera_pos[i] -= right[i] * speed

        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            for i in range(3):
                self.camera_pos[i] += right[i] * speed

        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
            self.camera_pos[1] += speed

        if glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            self.camera_pos[1] -= speed

        target = [
            self.camera_pos[0] + forward[0],
            self.camera_pos[1] + forward[1],
            self.camera_pos[2] + forward[2],
        ]

        return target

    def init_window(self):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, False)

        self.window = glfw.create_window(
            WIDTH, HEIGHT,
            "Vulkan Python - Diffuse + Specular",
            None,
            None,
        )

        if not self.window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed")

        glfw.set_input_mode(
            self.window,
            glfw.CURSOR,
            glfw.CURSOR_DISABLED,
        )

    def init_graphics(self):
        self.gfx = GraphicsDevice(
            self.window,
            app_name="Vulkan Python Lighting",
            preferred_format=output.VkFormat.VK_FORMAT_B8G8R8A8_SRGB,
        )
        self.instance = self.gfx.instance
        self.physical_device = self.gfx.physical_device
        self.device = self.gfx.device
        self.graphics_family = self.gfx.graphics_family
        self.present_family = self.gfx.present_family
        self.graphics_queue = self.gfx.graphics_queue
        self.present_queue = self.gfx.present_queue
        self.swapchain = self.gfx.swapchain
        self.swapchain_format = self.gfx.swapchain_format
        self.swapchain_extent = self.gfx.swapchain_extent
        self.swapchain_images = self.gfx.swapchain_images
        self.swapchain_views = self.gfx.swapchain_image_views

    def find_depth_format(self):
        candidates = [
            output.VkFormat.VK_FORMAT_D32_SFLOAT,
            output.VkFormat.VK_FORMAT_D32_SFLOAT_S8_UINT,
            output.VkFormat.VK_FORMAT_D24_UNORM_S8_UINT,
        ]

        for fmt in candidates:
            props = output.VkFormatProperties()
            self.instance.vkGetPhysicalDeviceFormatProperties(
                self.physical_device, fmt, byref(props)
            )
            if props.optimalTilingFeatures & output.VkFormatFeatureFlagBits.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT:
                return fmt

        raise RuntimeError("No supported depth format found")

    def create_image(self, fmt, usage, aspect):
        info = output.VkImageCreateInfo(
            imageType=output.VkImageType.VK_IMAGE_TYPE_2D,
            format=fmt,
            extent=output.VkExtent3D(
                width=self.swapchain_extent.width,
                height=self.swapchain_extent.height,
                depth=1,
            ),
            mipLevels=1,
            arrayLayers=1,
            samples=output.VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT,
            tiling=output.VkImageTiling.VK_IMAGE_TILING_OPTIMAL,
            usage=usage,
            sharingMode=output.VkSharingMode.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=output.VkImageLayout.VK_IMAGE_LAYOUT_UNDEFINED,
        )

        image = output.VkImage()
        check(
            self.device.vkCreateImage(byref(info), None, byref(image)),
            "vkCreateImage",
        )

        requirements = output.VkMemoryRequirements()
        self.device.vkGetImageMemoryRequirements(
            image, byref(requirements)
        )

        memory_type = self.find_memory_type(
            requirements.memoryTypeBits,
            output.VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )

        alloc = output.VkMemoryAllocateInfo(
            allocationSize=requirements.size,
            memoryTypeIndex=memory_type,
        )
        memory = output.VkDeviceMemory()
        check(
            self.device.vkAllocateMemory(byref(alloc), None, byref(memory)),
            "vkAllocateMemory(image)",
        )
        check(
            self.device.vkBindImageMemory(image, memory, 0),
            "vkBindImageMemory",
        )

        subresource = output.VkImageSubresourceRange(
            aspectMask=aspect,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        )
        view_info = output.VkImageViewCreateInfo(
            image=image,
            viewType=output.VkImageViewType.VK_IMAGE_VIEW_TYPE_2D,
            format=fmt,
            subresourceRange=subresource,
        )
        view = output.VkImageView()
        check(
            self.device.vkCreateImageView(
                byref(view_info), None, byref(view)
            ),
            "vkCreateImageView(depth)",
        )
        return image, memory, view

    def create_depth_resources(self):
        self.depth_format = self.find_depth_format()

        aspect = output.VkImageAspectFlagBits.VK_IMAGE_ASPECT_DEPTH_BIT
        if self.depth_format in (
            output.VkFormat.VK_FORMAT_D32_SFLOAT_S8_UINT,
            output.VkFormat.VK_FORMAT_D24_UNORM_S8_UINT,
        ):
            aspect |= output.VkImageAspectFlagBits.VK_IMAGE_ASPECT_STENCIL_BIT

        (
            self.depth_image,
            self.depth_memory,
            self.depth_view,
        ) = self.create_image(
            self.depth_format,
            output.VkImageUsageFlagBits.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            aspect,
        )

    def create_render_pass(self):
        color = output.VkAttachmentDescription(
            format=self.swapchain_format,
            samples=output.VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT,
            loadOp=output.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=output.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=output.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=output.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=output.VkImageLayout.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=output.VkImageLayout.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )

        depth = output.VkAttachmentDescription(
            format=self.depth_format,
            samples=output.VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT,
            loadOp=output.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=output.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            stencilLoadOp=output.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=output.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=output.VkImageLayout.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=output.VkImageLayout.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )

        attachments = (output.VkAttachmentDescription * 2)(color, depth)

        color_ref = output.VkAttachmentReference(
            attachment=0,
            layout=output.VkImageLayout.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        depth_ref = output.VkAttachmentReference(
            attachment=1,
            layout=output.VkImageLayout.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )

        subpass = output.VkSubpassDescription(
            pipelineBindPoint=output.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=pointer(color_ref),
            pDepthStencilAttachment=pointer(depth_ref),
        )

        dependency = output.VkSubpassDependency(
            srcSubpass=output.VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=(
                output.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                | output.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
            ),
            dstStageMask=(
                output.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                | output.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
            ),
            srcAccessMask=0,
            dstAccessMask=(
                output.VkAccessFlagBits.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                | output.VkAccessFlagBits.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
            ),
        )

        info = output.VkRenderPassCreateInfo(
            attachmentCount=2,
            pAttachments=cast(
                attachments,
                POINTER(output.VkAttachmentDescription),
            ),
            subpassCount=1,
            pSubpasses=pointer(subpass),
            dependencyCount=1,
            pDependencies=pointer(dependency),
        )

        self.render_pass = output.VkRenderPass()
        check(
            self.device.vkCreateRenderPass(
                byref(info), None, byref(self.render_pass)
            ),
            "vkCreateRenderPass",
        )

    def _load_shader_module(self, path):
        return self.gfx.load_shader_module(SHADERS_DIR / Path(path).name).handle

    def create_descriptor_system(self):
        binding = output.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=output.VkDescriptorType.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=(
                output.VkShaderStageFlagBits.VK_SHADER_STAGE_VERTEX_BIT
                | output.VkShaderStageFlagBits.VK_SHADER_STAGE_FRAGMENT_BIT
            ),
            pImmutableSamplers=None,
        )

        info = output.VkDescriptorSetLayoutCreateInfo(
            bindingCount=1,
            pBindings=pointer(binding),
        )

        check(
            self.device.vkCreateDescriptorSetLayout(
                byref(info), None, byref(self.descriptor_set_layout)
            ),
            "vkCreateDescriptorSetLayout",
        )

        pool_size = output.VkDescriptorPoolSize(
            type=output.VkDescriptorType.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=MAX_FRAMES_IN_FLIGHT,
        )

        pool_info = output.VkDescriptorPoolCreateInfo(
            maxSets=MAX_FRAMES_IN_FLIGHT,
            poolSizeCount=1,
            pPoolSizes=pointer(pool_size),
        )

        self.descriptor_pool = output.VkDescriptorPool()
        check(
            self.device.vkCreateDescriptorPool(
                byref(pool_info), None, byref(self.descriptor_pool)
            ),
            "vkCreateDescriptorPool",
        )

        layouts = (
            output.VkDescriptorSetLayout * MAX_FRAMES_IN_FLIGHT
        )(*([self.descriptor_set_layout] * MAX_FRAMES_IN_FLIGHT))

        alloc = output.VkDescriptorSetAllocateInfo(
            descriptorPool=self.descriptor_pool,
            descriptorSetCount=MAX_FRAMES_IN_FLIGHT,
            pSetLayouts=cast(
                layouts,
                POINTER(output.VkDescriptorSetLayout),
            ),
        )

        sets = (
            output.VkDescriptorSet * MAX_FRAMES_IN_FLIGHT
        )()
        check(
            self.device.vkAllocateDescriptorSets(
                byref(alloc), sets
            ),
            "vkAllocateDescriptorSets",
        )
        self.descriptor_sets = list(sets)

    def find_memory_type(self, type_bits, required):
        return self.gfx.find_memory_type(type_bits, required)

    def create_buffer(self, size, usage, properties):
        buf = self.gfx.create_buffer(size, usage, properties)
        return buf.handle, buf.memory

    def upload_memory(self, memory, data, size):
        self.gfx.upload_memory(memory, data, size)

    def create_geometry(self):
        # Three cubes at different positions plus a large floor.
        raw_vertices = []
        raw_indices = []

        cube = make_vertices(CUBE_VERTICES)
        floor = make_vertices(PLANE_VERTICES)

        raw_vertices.extend(cube)
        raw_vertices.extend(floor)

        raw_indices.extend(CUBE_INDICES)
        floor_base = len(CUBE_VERTICES)
        raw_indices.extend([i + floor_base for i in PLANE_INDICES])

        vertices = (Vertex * len(raw_vertices))(*raw_vertices)
        indices = (c_uint32 * len(raw_indices))(*raw_indices)

        self.index_count = len(raw_indices)

        self.vertex_buffer, self.vertex_memory = self.create_buffer(
            ctypes.sizeof(vertices),
            output.VkBufferUsageFlagBits.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            output.VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | output.VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        self.upload_memory(
            self.vertex_memory, vertices, ctypes.sizeof(vertices)
        )

        self.index_buffer, self.index_memory = self.create_buffer(
            ctypes.sizeof(indices),
            output.VkBufferUsageFlagBits.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            output.VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | output.VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        self.upload_memory(
            self.index_memory, indices, ctypes.sizeof(indices)
        )

    def create_uniform_buffers(self):
        size = ctypes.sizeof(SceneUBO)

        for _ in range(MAX_FRAMES_IN_FLIGHT):
            buffer, memory = self.create_buffer(
                size,
                output.VkBufferUsageFlagBits.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                output.VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | output.VkMemoryPropertyFlagBits.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )
            self.uniform_buffers.append(buffer)
            self.uniform_memories.append(memory)

            descriptor_info = output.VkDescriptorBufferInfo(
                buffer=buffer,
                offset=0,
                range=size,
            )
            write = output.VkWriteDescriptorSet(
                dstSet=self.descriptor_sets[len(self.uniform_buffers) - 1],
                dstBinding=0,
                dstArrayElement=0,
                descriptorCount=1,
                descriptorType=output.VkDescriptorType.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                pBufferInfo=pointer(descriptor_info),
            )
            self.device.vkUpdateDescriptorSets(
                1, pointer(write), 0, None
            )

    def create_pipeline(self):
        vert = self._load_shader_module(
            os.path.join("shaders", "lighting.vert.spv")
        )
        frag = self._load_shader_module(
            os.path.join("shaders", "lighting.frag.spv")
        )

        vert_stage = output.VkPipelineShaderStageCreateInfo(
            stage=output.VkShaderStageFlagBits.VK_SHADER_STAGE_VERTEX_BIT,
            module=vert,
            pName=b"main",
        )
        frag_stage = output.VkPipelineShaderStageCreateInfo(
            stage=output.VkShaderStageFlagBits.VK_SHADER_STAGE_FRAGMENT_BIT,
            module=frag,
            pName=b"main",
        )
        stages = (output.VkPipelineShaderStageCreateInfo * 2)(
            vert_stage, frag_stage
        )

        binding_desc = output.VkVertexInputBindingDescription(
            binding=0,
            stride=ctypes.sizeof(Vertex),
            inputRate=output.VkVertexInputRate.VK_VERTEX_INPUT_RATE_VERTEX,
        )
        attributes = (output.VkVertexInputAttributeDescription * 3)(
            output.VkVertexInputAttributeDescription(
                location=0,
                binding=0,
                format=output.VkFormat.VK_FORMAT_R32G32B32_SFLOAT,
                offset=0,
            ),
            output.VkVertexInputAttributeDescription(
                location=1,
                binding=0,
                format=output.VkFormat.VK_FORMAT_R32G32B32_SFLOAT,
                offset=12,
            ),
            output.VkVertexInputAttributeDescription(
                location=2,
                binding=0,
                format=output.VkFormat.VK_FORMAT_R32G32B32_SFLOAT,
                offset=24,
            ),
        )

        vertex_input = output.VkPipelineVertexInputStateCreateInfo(
            vertexBindingDescriptionCount=1,
            pVertexBindingDescriptions=pointer(binding_desc),
            vertexAttributeDescriptionCount=3,
            pVertexAttributeDescriptions=cast(
                attributes,
                POINTER(output.VkVertexInputAttributeDescription),
            ),
        )

        assembly = output.VkPipelineInputAssemblyStateCreateInfo(
            topology=output.VkPrimitiveTopology.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=VK_FALSE,
        )

        viewport = output.VkViewport(
            x=0.0,
            y=0.0,
            width=float(self.swapchain_extent.width),
            height=float(self.swapchain_extent.height),
            minDepth=0.0,
            maxDepth=1.0,
        )
        scissor = output.VkRect2D(
            offset=output.VkOffset2D(x=0, y=0),
            extent=self.swapchain_extent,
        )
        viewport_state = output.VkPipelineViewportStateCreateInfo(
            viewportCount=1,
            pViewports=pointer(viewport),
            scissorCount=1,
            pScissors=pointer(scissor),
        )

        rasterizer = output.VkPipelineRasterizationStateCreateInfo(
            polygonMode=output.VkPolygonMode.VK_POLYGON_MODE_FILL,
            cullMode=output.VkCullModeFlagBits.VK_CULL_MODE_BACK_BIT,
            frontFace=output.VkFrontFace.VK_FRONT_FACE_CLOCKWISE,
            lineWidth=1.0,
        )
        multisample = output.VkPipelineMultisampleStateCreateInfo(
            rasterizationSamples=output.VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT
        )

        depth_state = output.VkPipelineDepthStencilStateCreateInfo(
            depthTestEnable=VK_TRUE,
            depthWriteEnable=VK_TRUE,
            depthCompareOp=output.VkCompareOp.VK_COMPARE_OP_LESS,
        )

        blend_attachment = output.VkPipelineColorBlendAttachmentState(
            blendEnable=VK_FALSE,
            colorWriteMask=0xF,
        )
        blend = output.VkPipelineColorBlendStateCreateInfo(
            attachmentCount=1,
            pAttachments=pointer(blend_attachment),
        )

        push_constant = output.VkPushConstantRange(
            stageFlags=output.VkShaderStageFlagBits.VK_SHADER_STAGE_VERTEX_BIT,
            offset=0,
            size=64,
        )
        layout_info = output.VkPipelineLayoutCreateInfo(
            setLayoutCount=1,
            pSetLayouts=pointer(self.descriptor_set_layout),
            pushConstantRangeCount=1,
            pPushConstantRanges=pointer(push_constant),
        )
        self.pipeline_layout = output.VkPipelineLayout()
        check(
            self.device.vkCreatePipelineLayout(
                byref(layout_info), None, byref(self.pipeline_layout)
            ),
            "vkCreatePipelineLayout",
        )

        pipeline_info = output.VkGraphicsPipelineCreateInfo(
            stageCount=2,
            pStages=cast(
                stages,
                POINTER(output.VkPipelineShaderStageCreateInfo),
            ),
            pVertexInputState=pointer(vertex_input),
            pInputAssemblyState=pointer(assembly),
            pViewportState=pointer(viewport_state),
            pRasterizationState=pointer(rasterizer),
            pMultisampleState=pointer(multisample),
            pDepthStencilState=pointer(depth_state),
            pColorBlendState=pointer(blend),
            layout=self.pipeline_layout,
            renderPass=self.render_pass,
            subpass=0,
        )

        self.pipeline = output.VkPipeline()
        check(
            self.device.vkCreateGraphicsPipelines(
                output.VkPipelineCache(0),
                1,
                byref(pipeline_info),
                None,
                byref(self.pipeline),
            ),
            "vkCreateGraphicsPipelines",
        )

        self.device.vkDestroyShaderModule(vert, None)
        self.device.vkDestroyShaderModule(frag, None)

    def create_framebuffers(self):
        self.framebuffers = []
        for color_view in self.swapchain_views:
            attachments = (output.VkImageView * 2)(
                color_view, self.depth_view
            )
            info = output.VkFramebufferCreateInfo(
                renderPass=self.render_pass,
                attachmentCount=2,
                pAttachments=cast(
                    attachments, POINTER(output.VkImageView)
                ),
                width=self.swapchain_extent.width,
                height=self.swapchain_extent.height,
                layers=1,
            )
            fb = output.VkFramebuffer()
            check(
                self.device.vkCreateFramebuffer(
                    byref(info), None, byref(fb)
                ),
                "vkCreateFramebuffer",
            )
            self.framebuffers.append(fb)

    def create_command_pool(self):
        self.command_pool = self.gfx.create_command_pool()

    def create_command_buffers(self):
        self.command_buffers = self.gfx.allocate_command_buffers(
            self.command_pool, len(self.framebuffers)
        )

    def create_sync(self):
        self.sync = self.gfx.create_frame_sync(MAX_FRAMES_IN_FLIGHT)
        self.image_available = self.sync.image_available
        self.render_finished = self.sync.render_finished
        self.in_flight = self.sync.in_flight

    def update_uniforms(self, frame, target):
        now = time.perf_counter()

        eye = tuple(self.camera_pos)

        ubo = SceneUBO()
        ubo.view[:] = Mat4.look_at(
            eye, target, (0.0, 1.0, 0.0)
        )

        ubo.projection[:] = Mat4.perspective(
            math.radians(55.0),
            self.swapchain_extent.width / self.swapchain_extent.height,
            0.1,
            100.0,
        )

        ubo.camera_pos[:] = (
            eye[0],
            eye[1],
            eye[2],
            1.0,
        )

        # position.xyz + radius in w
        positions = [
            (-3.0, 3.2, 1.5, 1.0),
            (3.0, 2.5, 2.0, 1.0),
            (0.0, 4.0, -3.0, 1.0),
            (0.0, 1.8, 3.0, 1.0),
        ]
        # RGB + intensity in w
        colors = [
            (1.0, 0.20, 0.08, 7.0),
            (0.10, 0.35, 1.0, 6.0),
            (0.30, 1.0, 0.20, 5.0),
            (1.0, 0.55, 0.12, 4.0),
        ]

        for i in range(4):
            ubo.light_pos[i][:] = positions[i]
            ubo.light_color[i][:] = colors[i]

        self.upload_memory(
            self.uniform_memories[frame],
            ubo,
            ctypes.sizeof(ubo),
        )

    def record_frame(self, cmd, image_index, frame):
        check(
            self.device.vkResetCommandBuffer(cmd, 0),
            "vkResetCommandBuffer",
        )
        check(
            self.device.vkBeginCommandBuffer(
                cmd, byref(output.VkCommandBufferBeginInfo())
            ),
            "vkBeginCommandBuffer",
        )

        clear_values = (output.VkClearValue * 2)(
            output.VkClearValue(
                color=output.VkClearColorValue(
                    float32=(0.008, 0.012, 0.022, 1.0)
                )
            ),
            output.VkClearValue(
                depthStencil=output.VkClearDepthStencilValue(
                    depth=1.0, stencil=0
                )
            ),
        )

        render_info = output.VkRenderPassBeginInfo(
            renderPass=self.render_pass,
            framebuffer=self.framebuffers[image_index],
            renderArea=output.VkRect2D(
                offset=output.VkOffset2D(x=0, y=0),
                extent=self.swapchain_extent,
            ),
            clearValueCount=2,
            pClearValues=cast(
                clear_values, POINTER(output.VkClearValue)
            ),
        )

        self.device.vkCmdBeginRenderPass(
            cmd,
            byref(render_info),
            output.VkSubpassContents.VK_SUBPASS_CONTENTS_INLINE,
        )

        self.device.vkCmdBindPipeline(
            cmd,
            output.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self.pipeline,
        )

        offsets = (c_uint64 * 1)(0)
        vertex_buffers = (output.VkBuffer * 1)(self.vertex_buffer)
        self.device.vkCmdBindVertexBuffers(
            cmd, 0, 1, vertex_buffers, offsets
        )

        self.device.vkCmdBindIndexBuffer(
            cmd,
            self.index_buffer,
            0,
            output.VkIndexType.VK_INDEX_TYPE_UINT32,
        )

        descriptor_set = (output.VkDescriptorSet * 1)(
            self.descriptor_sets[frame]
        )
        self.device.vkCmdBindDescriptorSets(
            cmd,
            output.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self.pipeline_layout,
            0,
            1,
            descriptor_set,
            0,
            None,
        )

        # Draw three differently transformed copies of the cube.
        models = [
            Mat4.multiply(
                Mat4.translation(-2.25, 1.25, 0.0),
                Mat4.multiply(
                    Mat4.rotation_y(time.perf_counter() * 0.65),
                    Mat4.scale(1.25, 1.25, 1.25),
                ),
            ),
            Mat4.multiply(
                Mat4.translation(1.15, 1.15, 0.0),
                Mat4.multiply(
                    Mat4.rotation_y(-time.perf_counter() * 0.9),
                    Mat4.scale(1.15, 1.15, 1.15),
                ),
            ),
            Mat4.multiply(
                Mat4.translation(0.0, 0.75, -2.4),
                Mat4.multiply(
                    Mat4.rotation_x(time.perf_counter() * 0.45),
                    Mat4.scale(0.75, 0.75, 0.75),
                ),
            ),
        ]

        for model in models:
            model_data = (c_float * 16)(*model)
            self.device.vkCmdPushConstants(
                cmd,
                self.pipeline_layout,
                output.VkShaderStageFlagBits.VK_SHADER_STAGE_VERTEX_BIT,
                0,
                ctypes.sizeof(model_data),
                cast(model_data, c_void_p),
            )

            self.device.vkCmdDrawIndexed(
                cmd,
                36,
                1,
                0,
                0,
                0,
            )

        # Floor begins after the 24 cube vertices and uses six indices.
        # Rebind the same buffer with an index offset of 36 * 4.
        floor_index_offset = 36 * ctypes.sizeof(c_uint32)
        self.device.vkCmdBindIndexBuffer(
            cmd,
            self.index_buffer,
            floor_index_offset,
            output.VkIndexType.VK_INDEX_TYPE_UINT32,
        )

        floor_model = Mat4.identity()
        floor_data = (c_float * 16)(*floor_model)
        self.device.vkCmdPushConstants(
            cmd,
            self.pipeline_layout,
            output.VkShaderStageFlagBits.VK_SHADER_STAGE_VERTEX_BIT,
            0,
            ctypes.sizeof(floor_data),
            floor_data,
        )
        self.device.vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0)

        self.device.vkCmdEndRenderPass(cmd)
        check(
            self.device.vkEndCommandBuffer(cmd),
            "vkEndCommandBuffer",
        )

    def draw_frame(self):
        frame = self.current_frame
        fence = self.in_flight[frame]

        check(
            self.device.vkWaitForFences(
                1, byref(fence), VK_TRUE, UINT64_MAX
            ),
            "vkWaitForFences",
        )

        image_index = c_uint32(0)
        result = self.device.vkAcquireNextImageKHR(
            self.swapchain,
            UINT64_MAX,
            self.image_available[frame],
            output.VkFence(0),
            byref(image_index),
        )

        if result == output.VkResult.VK_ERROR_OUT_OF_DATE_KHR:
            raise RuntimeError("Window resize is disabled; swapchain became out of date")
        if result not in (
            output.VkResult.VK_SUCCESS,
            output.VkResult.VK_SUBOPTIMAL_KHR,
        ):
            check(result, "vkAcquireNextImageKHR")

        target = self.update_camera()
        self.update_uniforms(frame, target)
        self.record_frame(
            self.command_buffers[image_index.value],
            image_index.value,
            frame,
        )

        check(
            self.device.vkResetFences(1, byref(fence)),
            "vkResetFences",
        )

        wait_semaphores = (output.VkSemaphore * 1)(
            self.image_available[frame]
        )
        signal_semaphores = (output.VkSemaphore * 1)(
            self.render_finished[image_index.value]
        )
        wait_stages = (c_uint32 * 1)(
            output.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        )
        command_buffers = (output.VkCommandBuffer * 1)(
            self.command_buffers[image_index.value]
        )

        submit = output.VkSubmitInfo(
            waitSemaphoreCount=1,
            pWaitSemaphores=cast(
                wait_semaphores, POINTER(output.VkSemaphore)
            ),
            pWaitDstStageMask=cast(
                wait_stages, POINTER(c_uint32)
            ),
            commandBufferCount=1,
            pCommandBuffers=cast(
                command_buffers, POINTER(output.VkCommandBuffer)
            ),
            signalSemaphoreCount=1,
            pSignalSemaphores=cast(
                signal_semaphores, POINTER(output.VkSemaphore)
            ),
        )

        check(
            self.device.vkQueueSubmit(
                self.graphics_queue,
                1,
                byref(submit),
                fence,
            ),
            "vkQueueSubmit",
        )

        swapchains = (output.VkSwapchainKHR * 1)(self.swapchain)
        present = output.VkPresentInfoKHR(
            waitSemaphoreCount=1,
            pWaitSemaphores=cast(
                signal_semaphores, POINTER(output.VkSemaphore)
            ),
            swapchainCount=1,
            pSwapchains=cast(
                swapchains, POINTER(output.VkSwapchainKHR)
            ),
            pImageIndices=pointer(image_index),
        )

        result = self.device.vkQueuePresentKHR(
            self.present_queue, byref(present)
        )
        if result not in (
            output.VkResult.VK_SUCCESS,
            output.VkResult.VK_SUBOPTIMAL_KHR,
        ):
            check(result, "vkQueuePresentKHR")

        self.current_frame = (frame + 1) % MAX_FRAMES_IN_FLIGHT

        self.fps_frames += 1

        now = time.perf_counter()

        if now - self.fps_time >= 0.25:
            fps = self.fps_frames / (now - self.fps_time)

            glfw.set_window_title(
                self.window,
                f"Vulkan Python - {fps:.1f} FPS"
            )

            self.fps_frames = 0
            self.fps_time = now

    def run(self):
        self.init_window()
        self.init_graphics()
        self.create_depth_resources()
        self.create_render_pass()
        self.create_descriptor_system()
        self.create_geometry()
        self.create_uniform_buffers()
        self.create_pipeline()
        self.create_framebuffers()
        self.create_command_pool()
        self.create_command_buffers()
        self.create_sync()

        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()

                if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    glfw.set_window_should_close(self.window, True)

                self.draw_frame()
        finally:
            self.cleanup()

    def cleanup(self):
        if getattr(self, "gfx", None) is not None and self.gfx.device is not None:
            self.gfx.device.vkDeviceWaitIdle()
            if getattr(self, "sync", None) is not None:
                self.gfx.destroy_frame_sync(self.sync)
            if self.command_pool:
                self.gfx.device.vkDestroyCommandPool(self.command_pool, None)
            for framebuffer in self.framebuffers:
                self.gfx.device.vkDestroyFramebuffer(framebuffer, None)
            if self.pipeline:
                self.gfx.device.vkDestroyPipeline(self.pipeline, None)
            if self.pipeline_layout:
                self.gfx.device.vkDestroyPipelineLayout(self.pipeline_layout, None)
            if self.descriptor_pool:
                self.gfx.device.vkDestroyDescriptorPool(self.descriptor_pool, None)
            if self.descriptor_set_layout:
                self.gfx.device.vkDestroyDescriptorSetLayout(
                    self.descriptor_set_layout, None
                )
            if self.depth_view:
                self.gfx.device.vkDestroyImageView(self.depth_view, None)
            if self.depth_image:
                self.gfx.device.vkDestroyImage(self.depth_image, None)
            if self.depth_memory:
                self.gfx.device.vkFreeMemory(self.depth_memory, None)
            if self.render_pass:
                self.gfx.device.vkDestroyRenderPass(self.render_pass, None)
            self.gfx.close()

        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()


if __name__ == "__main__":
    App().run()
