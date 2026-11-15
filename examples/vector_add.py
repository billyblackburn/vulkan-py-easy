"""Simple vector addition powered by vulkan_py.easy."""

import struct
from ctypes import c_float

from vulkan_py.easy import get_default_device

SHADER = """#version 450
layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer A { float a[]; };
layout(set = 0, binding = 1) readonly buffer B { float b[]; };
layout(set = 0, binding = 2) writeonly buffer Out { float out_data[]; };

layout(push_constant) uniform Params {
    uint count;
} params;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx < params.count) {
        out_data[idx] = a[idx] + b[idx];
    }
}
"""


def main() -> None:
    count = 1024
    a_data = (c_float * count)(*[float(i) for i in range(count)])
    b_data = (c_float * count)(*[float(i * 2) for i in range(count)])

    with get_default_device() as device:
        print(f"Using device: {device.device_name}")

        buf_a = device.create_buffer_with_data(a_data)
        buf_b = device.create_buffer_with_data(b_data)
        buf_out = device.create_buffer(buf_a.size)

        shader = device.create_shader_module(SHADER)
        pipeline = device.create_compute_pipeline(
            shader, num_bindings=3, push_constant_size=4
        )
        bind_group = device.create_bind_group(pipeline, [buf_a, buf_b, buf_out])

        device.dispatch(
            pipeline,
            bind_group,
            workgroups=((count + 63) // 64, 1, 1),
            push_constants=struct.pack("<I", count),
        )

        actual = list(struct.unpack(f"<{count}f", device.read_buffer(buf_out)))

    expected = [a_data[i] + b_data[i] for i in range(count)]
    assert actual == expected, "Mismatch between GPU output and expected sum!"
    print(f"Vector add succeeded for {count} floats!")
    print(f"  a[:5]   = {list(a_data)[:5]}")
    print(f"  b[:5]   = {list(b_data)[:5]}")
    print(f"  out[:5] = {actual[:5]}")


if __name__ == "__main__":
    main()
