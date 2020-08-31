#![cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))] // TODO later

use std::{
    mem::{forget, size_of, size_of_val},
    slice::from_raw_parts,
    str::from_utf8,
};

use nuklear::{Buffer as NkBuffer, Context, ConvertConfig, DrawVertexLayoutAttribute, DrawVertexLayoutElements, DrawVertexLayoutFormat, Handle, Size, Vec2};
use wgpu::*;

pub const TEXTURE_FORMAT: TextureFormat = TextureFormat::Bgra8Unorm;

#[allow(dead_code)]
struct Vertex {
    // "Position",
    pos: [f32; 2],
    // "TexCoord",
    tex: [f32; 2],
    // "Color",
    col: [u8; 4],
}

#[allow(dead_code)]
struct WgpuTexture {
    texture: Texture,
    sampler: Sampler,

    pub bind_group: BindGroup,
}

type ProjectionMatrix = [[f32; 4]; 4];

impl WgpuTexture {
    pub fn new(device: &mut Device, queue: &mut Queue, drawer: &Drawer, image: &[u8], width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: Extent3d { width: width, height: height, depth: 1 },
            array_layer_count: 1,
            dimension: TextureDimension::D2,
            format: TEXTURE_FORMAT,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: CompareFunction::Always,
        });

        let bytes = image.len();
        let buffer = device.create_buffer_with_data(image, BufferUsage::COPY_SRC);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: None,
        });

        encoder.copy_buffer_to_texture(
            BufferCopyView {
                buffer: &buffer,
                offset: 0,
                bytes_per_row: bytes as u32 / height,
                rows_per_image: height,
            },
            TextureCopyView {
                texture: &texture,
                mip_level: 0,
                array_layer: 0,
                origin: Origin3d::ZERO,
            },
            Extent3d { width, height, depth: 1 },
        );

        queue.submit(&[encoder.finish()]);

        WgpuTexture {
            bind_group: device.create_bind_group(&BindGroupDescriptor {
                label: Some("nuklear-wgpu-bind-group"),
                layout: &drawer.tla,
                bindings: &[
                    wgpu::Binding {
                        binding: 0,
                        resource: BindingResource::TextureView(&texture.create_default_view()),
                    },
                    wgpu::Binding {
                        binding: 1,
                        resource: BindingResource::Sampler(&sampler),
                    },
                ],
            }),
            sampler,
            texture,
        }
    }
}

pub struct Drawer {
    cmd: NkBuffer,
    pso: RenderPipeline,
    tla: BindGroupLayout,
    tex: Vec<WgpuTexture>,
    ubf: Buffer,
    ubg: BindGroup,
    vsz: usize,
    esz: usize,
    vle: DrawVertexLayoutElements,

    pub col: Option<Color>,
}

impl Drawer {
    pub fn new(device: &mut Device, col: Color, texture_count: usize, vbo_size: usize, ebo_size: usize, command_buffer: NkBuffer) -> Drawer {
        let vs = include_bytes!("../shaders/vs.fx");
        let fs = include_bytes!("../shaders/ps.fx");

        let vs = device.create_shader_module(compile_glsl(from_utf8(vs).unwrap(), glsl_to_spirv::ShaderType::Vertex).as_slice());
        let fs = device.create_shader_module(compile_glsl(from_utf8(fs).unwrap(), glsl_to_spirv::ShaderType::Fragment).as_slice());

        let ubf = device.create_buffer(&BufferDescriptor {
            label: Some("nuklear-wgpu-projection-matrix-buffer"),
            size: size_of::<ProjectionMatrix>() as u64,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        });
        let ubg = BindGroupLayoutDescriptor {
            label: Some("nuklear-wgpu-vertex-buffer-bind-group-layout"),
            bindings: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer { dynamic: false },
            }],
        };

        let tbg = BindGroupLayoutDescriptor {
            label: Some("nuklear-wgpu-fragment-buffer-bind-group-layout"),
            bindings: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        component_type: TextureComponentType::Float,
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: false,
                    },
                },
            ],
        };
        let tla = device.create_bind_group_layout(&tbg);
        let ula = device.create_bind_group_layout(&ubg);

        Drawer {
            cmd: command_buffer,
            col: Some(col),
            pso: device.create_render_pipeline(&RenderPipelineDescriptor {
                layout: &device.create_pipeline_layout(&PipelineLayoutDescriptor { bind_group_layouts: &[&ula, &tla] }),
                vertex_stage: ProgrammableStageDescriptor { module: &vs, entry_point: "main" },
                fragment_stage: Some(ProgrammableStageDescriptor { module: &fs, entry_point: "main" }),
                rasterization_state: Some(RasterizationStateDescriptor {
                    front_face: FrontFace::Cw,
                    cull_mode: CullMode::None,
                    depth_bias: 0,
                    depth_bias_slope_scale: 0.0,
                    depth_bias_clamp: 0.0,
                }),
                primitive_topology: PrimitiveTopology::TriangleList,
                color_states: &[ColorStateDescriptor {
                    format: TEXTURE_FORMAT,
                    color_blend: BlendDescriptor {
                        src_factor: BlendFactor::SrcAlpha,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                        operation: BlendOperation::Add,
                    },
                    alpha_blend: BlendDescriptor {
                        src_factor: BlendFactor::OneMinusDstAlpha,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    write_mask: ColorWrite::ALL,
                }],
                depth_stencil_state: None,
                vertex_state: VertexStateDescriptor {
                    index_format: IndexFormat::Uint16,
                    vertex_buffers: &[VertexBufferDescriptor {
                        stride: (size_of::<Vertex>()) as u64,
                        step_mode: InputStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttributeDescriptor {
                                format: VertexFormat::Float2,
                                shader_location: 0,
                                offset: 0,
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: VertexFormat::Float2,
                                shader_location: 1,
                                offset: 8,
                            },
                            wgpu::VertexAttributeDescriptor {
                                format: VertexFormat::Uint,
                                shader_location: 2,
                                offset: 16,
                            },
                        ],
                    }],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            }),
            tex: Vec::with_capacity(texture_count + 1),
            vsz: vbo_size,
            esz: ebo_size,
            ubg: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nuklear-wgpu-projection-matrix-bind-group-descriptor"),
                layout: &ula,
                bindings: &[Binding {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &ubf,
                        range: 0..(size_of::<ProjectionMatrix>() as u64),
                    },
                }],
            }),
            ubf: ubf,
            vle: DrawVertexLayoutElements::new(&[
                (DrawVertexLayoutAttribute::Position, DrawVertexLayoutFormat::Float, 0),
                (DrawVertexLayoutAttribute::TexCoord, DrawVertexLayoutFormat::Float, size_of::<f32>() as Size * 2),
                (DrawVertexLayoutAttribute::Color, DrawVertexLayoutFormat::B8G8R8A8, size_of::<f32>() as Size * 4),
                (DrawVertexLayoutAttribute::AttributeCount, DrawVertexLayoutFormat::Count, 0),
            ]),
            tla: tla,
        }
    }

    pub fn add_texture(&mut self, device: &mut Device, queue: &mut Queue, image: &[u8], width: u32, height: u32) -> Handle {
        self.tex.push(WgpuTexture::new(device, queue, self, image, width, height));
        Handle::from_id(self.tex.len() as i32)
    }

    pub fn draw(&mut self, ctx: &mut Context, cfg: &mut ConvertConfig, encoder: &mut CommandEncoder, view: &TextureView, device: &mut Device, width: u32, height: u32, scale: Vec2) {
        let ortho: ProjectionMatrix = [
            [2.0f32 / width as f32, 0.0f32, 0.0f32, 0.0f32],
            [0.0f32, -2.0f32 / height as f32, 0.0f32, 0.0f32],
            [0.0f32, 0.0f32, 0.0f32, 0.0f32],
            [-1.0f32, 1.0f32, 0.0f32, 1.0f32],
        ];
        let ubf_size = size_of_val(&ortho);
        cfg.set_vertex_layout(&self.vle);
        cfg.set_vertex_size(size_of::<Vertex>());

        let mut vbf = device.create_buffer_mapped(&BufferDescriptor {
            label: Some("nuklear-wgpu-temp-vertex-buffer"),
            size: self.vsz as BufferAddress,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_SRC,
        });
        let mut ebf = device.create_buffer_mapped(&BufferDescriptor {
            label: Some("nuklear-wgpu-temp-index-buffer"),
            size: self.esz as BufferAddress,
            usage: BufferUsage::INDEX | BufferUsage::COPY_SRC,
        });
        let ubf = device.create_buffer_with_data(as_typed_slice(&ortho), BufferUsage::COPY_SRC);
        {
            let mut vbuf = NkBuffer::with_fixed(&mut vbf.data);
            let mut ebuf = NkBuffer::with_fixed(&mut ebf.data);

            ctx.convert(&mut self.cmd, &mut vbuf, &mut ebuf, cfg);
        }
        let vbf = vbf.finish();
        let ebf = ebf.finish();

        encoder.copy_buffer_to_buffer(&ubf, 0, &self.ubf, 0, ubf_size as u64);

        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[RenderPassColorAttachmentDescriptor {
                attachment: &view,
                load_op: match self.col {
                    Some(_) => wgpu::LoadOp::Clear,
                    _ => wgpu::LoadOp::Load,
                },
                resolve_target: None,
                store_op: StoreOp::Store,
                clear_color: self.col.unwrap_or(Color { r: 1.0, g: 2.0, b: 3.0, a: 1.0 }),
            }],
            depth_stencil_attachment: None,
        });
        rpass.set_pipeline(&self.pso);

        rpass.set_vertex_buffer(0, &vbf, 0, 0);
        rpass.set_index_buffer(&ebf, 0, 0);

        rpass.set_bind_group(0, &self.ubg, &[]);

        let mut start = 0;
        let mut end;

        for cmd in ctx.draw_command_iterator(&self.cmd) {
            if cmd.elem_count() < 1 {
                continue;
            }

            let id = cmd.texture().id().unwrap();
            let res = self.find_res(id).unwrap();

            end = start + cmd.elem_count();

            rpass.set_bind_group(1, &res.bind_group, &[]);

            rpass.set_scissor_rect((cmd.clip_rect().x * scale.x) as u32, (cmd.clip_rect().y * scale.y) as u32, (cmd.clip_rect().w * scale.x) as u32, (cmd.clip_rect().h * scale.y) as u32);

            rpass.draw_indexed(start..end, 0 as i32, 0..1);

            start = end;
        }
    }

    fn find_res(&self, id: i32) -> Option<&WgpuTexture> {
        if id > 0 && id as usize <= self.tex.len() {
            self.tex.get((id - 1) as usize)
        } else {
            None
        }
    }
}

fn as_typed_slice<T>(data: &[T]) -> &[u8] {
    unsafe { from_raw_parts(data.as_ptr() as *const u8, data.len() * size_of::<T>()) }
}

fn compile_glsl(code: &str, ty: glsl_to_spirv::ShaderType) -> Vec<u32> {
    use std::io::Read;

    let mut output = glsl_to_spirv::compile(code, ty).unwrap();
    let mut spv = Vec::new();
    output.read_to_end(&mut spv).unwrap();
    let spv32: Vec<u32> = unsafe { Vec::from_raw_parts(spv.as_mut_ptr() as *mut _ as *mut u32, spv.len() / 4, spv.capacity() / 4) };
    forget(spv);
    spv32
}
