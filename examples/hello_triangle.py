"""Hello Triangle using vulkan_py.easy.GraphicsDevice."""

from ctypes import POINTER, byref, cast, pointer
from pathlib import Path

import glfw
import vulkan_py as vk
from vulkan_py.easy import GraphicsDevice, check_vk

MAX_FRAMES_IN_FLIGHT = 2
SHADERS_DIR = Path(__file__).resolve().parent / "shaders"


class App:
    def __init__(self, name: str = "Vulkan-Hello-Triangle", res: tuple[int, int] = (800, 600)):
        self.name = name
        self.res = res

    def init_window(self) -> None:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.window = glfw.create_window(self.res[0], self.res[1], self.name, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed")

    def create_render_pass(self) -> None:
        color_attachment = vk.VkAttachmentDescription(
            format=self.gfx.swapchain_format,
            samples=vk.VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VkAttachmentLoadOp.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VkAttachmentStoreOp.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VkImageLayout.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VkImageLayout.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )
        color_ref = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VkImageLayout.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=pointer(color_ref),
        )
        dependency = vk.VkSubpassDependency(
            srcSubpass=vk.VK_SUBPASS_EXTERNAL,
            dstSubpass=0,
            srcStageMask=vk.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstStageMask=vk.VkPipelineStageFlagBits.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            srcAccessMask=0,
            dstAccessMask=vk.VkAccessFlagBits.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        )

        render_pass = vk.VkRenderPass()
        check_vk(
            self.gfx.device.vkCreateRenderPass(
                vk.VkRenderPassCreateInfo(
                    attachmentCount=1,
                    pAttachments=pointer(color_attachment),
                    subpassCount=1,
                    pSubpasses=pointer(subpass),
                    dependencyCount=1,
                    pDependencies=pointer(dependency),
                ),
                None,
                byref(render_pass),
            ),
            "vkCreateRenderPass",
        )
        self.render_pass = render_pass

    def create_graphics_pipeline(self) -> None:
        vert_mod = self.gfx.load_shader_module(SHADERS_DIR / "triangle.vert.spv")
        frag_mod = self.gfx.load_shader_module(SHADERS_DIR / "triangle.frag.spv")

        stages = (vk.VkPipelineShaderStageCreateInfo * 2)(
            vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VkShaderStageFlagBits.VK_SHADER_STAGE_VERTEX_BIT,
                module=vert_mod.handle,
                pName=b"main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VkShaderStageFlagBits.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=frag_mod.handle,
                pName=b"main",
            ),
        )

        extent = self.gfx.swapchain_extent
        viewport = vk.VkViewport(
            x=0.0, y=0.0, width=float(extent.width), height=float(extent.height), minDepth=0.0, maxDepth=1.0
        )
        scissor = vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0), extent=extent)
        viewport_state = vk.VkPipelineViewportStateCreateInfo(
            viewportCount=1, pViewports=pointer(viewport), scissorCount=1, pScissors=pointer(scissor)
        )
        rasterizer = vk.VkPipelineRasterizationStateCreateInfo(
            polygonMode=vk.VkPolygonMode.VK_POLYGON_MODE_FILL,
            cullMode=vk.VkCullModeFlagBits.VK_CULL_MODE_BACK_BIT,
            frontFace=vk.VkFrontFace.VK_FRONT_FACE_CLOCKWISE,
            lineWidth=1.0,
        )
        multisample = vk.VkPipelineMultisampleStateCreateInfo(
            rasterizationSamples=vk.VkSampleCountFlagBits.VK_SAMPLE_COUNT_1_BIT
        )
        color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(
            blendEnable=0, colorWriteMask=0xF
        )
        color_blend = vk.VkPipelineColorBlendStateCreateInfo(
            attachmentCount=1, pAttachments=pointer(color_blend_attachment)
        )
        vertex_input = vk.VkPipelineVertexInputStateCreateInfo()
        input_assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
            topology=vk.VkPrimitiveTopology.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        )

        self.pipeline_layout = vk.VkPipelineLayout()
        check_vk(
            self.gfx.device.vkCreatePipelineLayout(
                vk.VkPipelineLayoutCreateInfo(), None, byref(self.pipeline_layout)
            ),
            "vkCreatePipelineLayout",
        )

        pipeline_info = vk.VkGraphicsPipelineCreateInfo(
            stageCount=2,
            pStages=cast(stages, POINTER(vk.VkPipelineShaderStageCreateInfo)),
            pVertexInputState=pointer(vertex_input),
            pInputAssemblyState=pointer(input_assembly),
            pViewportState=pointer(viewport_state),
            pRasterizationState=pointer(rasterizer),
            pMultisampleState=pointer(multisample),
            pColorBlendState=pointer(color_blend),
            layout=self.pipeline_layout,
            renderPass=self.render_pass,
            subpass=0,
        )
        self.pipeline = vk.VkPipeline()
        check_vk(
            self.gfx.device.vkCreateGraphicsPipelines(
                vk.VkPipelineCache(0), 1, byref(pipeline_info), None, byref(self.pipeline)
            ),
            "vkCreateGraphicsPipelines",
        )
        self.gfx.device.vkDestroyShaderModule(vert_mod.handle, None)
        self.gfx.device.vkDestroyShaderModule(frag_mod.handle, None)

    def create_framebuffers(self) -> None:
        self.framebuffers = []
        extent = self.gfx.swapchain_extent
        for view in self.gfx.swapchain_image_views:
            fb = vk.VkFramebuffer()
            check_vk(
                self.gfx.device.vkCreateFramebuffer(
                    vk.VkFramebufferCreateInfo(
                        renderPass=self.render_pass,
                        attachmentCount=1,
                        pAttachments=pointer(view),
                        width=extent.width,
                        height=extent.height,
                        layers=1,
                    ),
                    None,
                    byref(fb),
                ),
                "vkCreateFramebuffer",
            )
            self.framebuffers.append(fb)

    def record_command_buffers(self) -> None:
        self.command_pool = self.gfx.create_command_pool()
        self.command_buffers = self.gfx.allocate_command_buffers(
            self.command_pool, len(self.framebuffers)
        )
        clear_value = vk.VkClearValue(
            color=vk.VkClearColorValue(float32=(0.01, 0.01, 0.02, 1.0))
        )
        for i, cmd in enumerate(self.command_buffers):
            check_vk(
                self.gfx.device.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo()),
                "vkBeginCommandBuffer",
            )
            render_pass_info = vk.VkRenderPassBeginInfo(
                renderPass=self.render_pass,
                framebuffer=self.framebuffers[i],
                renderArea=vk.VkRect2D(
                    offset=vk.VkOffset2D(x=0, y=0), extent=self.gfx.swapchain_extent
                ),
                clearValueCount=1,
                pClearValues=pointer(clear_value),
            )
            self.gfx.device.vkCmdBeginRenderPass(
                cmd, byref(render_pass_info), vk.VkSubpassContents.VK_SUBPASS_CONTENTS_INLINE
            )
            self.gfx.device.vkCmdBindPipeline(
                cmd, vk.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline
            )
            self.gfx.device.vkCmdDraw(cmd, 3, 1, 0, 0)
            self.gfx.device.vkCmdEndRenderPass(cmd)
            check_vk(self.gfx.device.vkEndCommandBuffer(cmd), "vkEndCommandBuffer")

    def draw_frame(self) -> None:
        _, image_index = self.gfx.acquire_next_image(self.sync, self.current_frame)
        self.gfx.submit_and_present(
            self.sync, self.current_frame, image_index, self.command_buffers[image_index]
        )
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT

    def run(self) -> None:
        self.init_window()
        self.gfx = GraphicsDevice(self.window, app_name=self.name)
        self.create_render_pass()
        self.create_graphics_pipeline()
        self.create_framebuffers()
        self.record_command_buffers()
        self.sync = self.gfx.create_frame_sync(MAX_FRAMES_IN_FLIGHT)
        self.current_frame = 0
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self.draw_frame()
            self.gfx.device.vkDeviceWaitIdle()
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        self.gfx.destroy_frame_sync(self.sync)
        self.gfx.device.vkDestroyCommandPool(self.command_pool, None)
        for fb in self.framebuffers:
            self.gfx.device.vkDestroyFramebuffer(fb, None)
        self.gfx.device.vkDestroyPipeline(self.pipeline, None)
        self.gfx.device.vkDestroyPipelineLayout(self.pipeline_layout, None)
        self.gfx.device.vkDestroyRenderPass(self.render_pass, None)
        self.gfx.close()
        glfw.destroy_window(self.window)
        glfw.terminate()


if __name__ == "__main__":
    App().run()
