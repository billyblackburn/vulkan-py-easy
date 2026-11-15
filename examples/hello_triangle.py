"""Hello Triangle using vulkan_py.easy.GraphicsDevice."""

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

    def record_command_buffers(self) -> None:
        self.command_pool = self.gfx.create_command_pool()
        self.command_buffers = self.gfx.allocate_command_buffers(
            self.command_pool, len(self.framebuffers)
        )
        for fb, cmd in zip(self.framebuffers, self.command_buffers):
            check_vk(
                self.gfx.device.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo()),
                "vkBeginCommandBuffer",
            )
            self.gfx.begin_render_pass(
                cmd, self.render_pass, fb, clear_color=(0.01, 0.01, 0.02, 1.0)
            )
            self.gfx.device.vkCmdBindPipeline(
                cmd, vk.VkPipelineBindPoint.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline.pipeline
            )
            self.gfx.device.vkCmdDraw(cmd, 3, 1, 0, 0)
            self.gfx.device.vkCmdEndRenderPass(cmd)
            check_vk(self.gfx.device.vkEndCommandBuffer(cmd), "vkEndCommandBuffer")

    def run(self) -> None:
        self.init_window()
        self.gfx = GraphicsDevice(self.window, app_name=self.name)
        self.render_pass = self.gfx.create_render_pass()
        self.pipeline = self.gfx.create_graphics_pipeline(
            SHADERS_DIR / "triangle.vert.spv",
            SHADERS_DIR / "triangle.frag.spv",
            self.render_pass,
        )
        self.framebuffers = self.gfx.create_framebuffers(self.render_pass)
        self.record_command_buffers()
        self.sync = self.gfx.create_frame_sync(MAX_FRAMES_IN_FLIGHT)
        frame = 0
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                _, image_index = self.gfx.acquire_next_image(self.sync, frame)
                self.gfx.submit_and_present(
                    self.sync, frame, image_index, self.command_buffers[image_index]
                )
                frame = (frame + 1) % MAX_FRAMES_IN_FLIGHT
            self.gfx.device.vkDeviceWaitIdle()
        finally:
            self.gfx.destroy_frame_sync(self.sync)
            self.gfx.device.vkDestroyCommandPool(self.command_pool, None)
            self.gfx.close()
            glfw.destroy_window(self.window)
            glfw.terminate()


if __name__ == "__main__":
    App().run()
