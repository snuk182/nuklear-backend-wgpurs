#![cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))] // TODO later

#[macro_use]
extern crate wgpu;

use std::{
    mem::size_of,
    slice::from_raw_parts,
    str::from_utf8,
};

use nuklear::{
    Buffer as NkBuffer,
    Context,
    ConvertConfig,
    DrawVertexLayoutAttribute,
    DrawVertexLayoutElements,
    DrawVertexLayoutFormat,
    Handle,
    Size,
    Vec2,
};
use wgpu::{
    AddressMode,
    BindGroup,
    BindGroupDescriptor,
    BindGroupEntry,
    BindGroupLayout,
    BindGroupLayoutDescriptor,
    BindGroupLayoutEntry,
    BindingResource,
    BindingType,
    BlendDescriptor,
    BlendFactor,
    BlendOperation,
    Buffer,
    BufferAddress,
    BufferDescriptor,
    BufferUsage,
    ColorStateDescriptor,
    ColorWrite,
    CommandEncoder,
    CullMode,
    Device,
    Extent3d,
    FilterMode,
    FrontFace,
    IndexFormat,
    InputStepMode,
    Operations,
    Origin3d,
    PipelineLayoutDescriptor,
    PrimitiveTopology,
    ProgrammableStageDescriptor,
    Queue,
    RasterizationStateDescriptor,
    RenderPassColorAttachmentDescriptor,
    RenderPassDescriptor,
    RenderPipeline,
    RenderPipelineDescriptor,
    Sampler,
    SamplerDescriptor,
    ShaderModuleSource,
    Texture,
    TextureComponentType,
    TextureCopyView,
    TextureDataLayout,
    TextureDimension,
    TextureFormat,
    TextureUsage,
    TextureView,
    TextureViewDescriptor,
    VertexBufferDescriptor,
    VertexStateDescriptor,
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

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
            label: Some("nuklear-wgpu-texture"),
            size: Extent3d { width: width, height: height, depth: 1 },
            dimension: TextureDimension::D2,
            format: TEXTURE_FORMAT,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsage::SAMPLED | TextureUsage::COPY_DST,
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("nuklear-wgpu-texture-sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: None,
        });

        let bytes = image.len();

        queue.write_texture(
            TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
            },
            image,
            TextureDataLayout {
                offset: 0,
                bytes_per_row: bytes as u32 / height,
                rows_per_image: height,
            },
            Extent3d {
                width,
                height,
                depth: 1,
            },
        );

        WgpuTexture {
            bind_group: device.create_bind_group(&BindGroupDescriptor {
                label: Some("nuklear-wgpu-texture-bind-group"),
                layout: &drawer.tla,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&texture.create_view(&TextureViewDescriptor::default())),
                    },
                    BindGroupEntry {
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

    vbf: Buffer,
    ebf: Buffer,
}

enum ShaderStage {
    Vertex,
    Fragment,
}

impl Drawer {
    pub fn new(device: &mut Device, texture_count: usize, vbo_size: usize, ebo_size: usize, command_buffer: NkBuffer) -> Drawer {
        let vs = include_bytes!("../shaders/vs.fx");
        let fs = include_bytes!("../shaders/ps.fx");

        let vs = device.create_shader_module(compile_glsl(from_utf8(vs).unwrap(), ShaderStage::Vertex));
        let fs = device.create_shader_module(compile_glsl(from_utf8(fs).unwrap(), ShaderStage::Fragment));

        let ubf = device.create_buffer(&BufferDescriptor {
            label: Some("nuklear-wgpu-projection-matrix-buffer"),
            size: size_of::<ProjectionMatrix>() as u64,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        let ubg = BindGroupLayoutDescriptor {
            label: Some("nuklear-wgpu-vertex-buffer-bind-group-layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::VERTEX,
                ty: wgpu::BindingType::UniformBuffer {
                    dynamic: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        };

        let tbg = BindGroupLayoutDescriptor {
            label: Some("nuklear-wgpu-fragment-buffer-bind-group-layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        component_type: TextureComponentType::Float,
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: BindingType::Sampler {
                        comparison: false,
                    },
                    count: None,
                },
            ],
        };
        let tla = device.create_bind_group_layout(&tbg);
        let ula = device.create_bind_group_layout(&ubg);

        let vbf = device.create_buffer(&BufferDescriptor {
            label: Some("nuklear-wgpu-vertex-buffer"),
            size: vbo_size as BufferAddress,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        let ebf = device.create_buffer(&BufferDescriptor {
            label: Some("nuklear-wgpu-index-buffer"),
            size: ebo_size as BufferAddress,
            usage: BufferUsage::INDEX | BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        Drawer {
            cmd: command_buffer,
            pso: device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("nuklear-wgpu-render-pipeline"),
                layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("nuklear-wgpu-pipeline-layout"),
                    bind_group_layouts: &[&ula, &tla],
                    push_constant_ranges: &[],
                })),
                vertex_stage: ProgrammableStageDescriptor { module: &vs, entry_point: "main" },
                fragment_stage: Some(ProgrammableStageDescriptor { module: &fs, entry_point: "main" }),
                rasterization_state: Some(RasterizationStateDescriptor {
                    front_face: FrontFace::Cw,
                    cull_mode: CullMode::None,
                    clamp_depth: false,
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
                        attributes: &vertex_attr_array![0 => Float2, 1 => Float2, 2 => Uint],
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
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(ubf.slice(..)),
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

            vbf: vbf,
            ebf: ebf,
        }
    }

    pub fn add_texture(&mut self, device: &mut Device, queue: &mut Queue, image: &[u8], width: u32, height: u32) -> Handle {
        self.tex.push(WgpuTexture::new(device, queue, self, image, width, height));
        Handle::from_id(self.tex.len() as i32)
    }

    pub fn draw(
        &mut self,
        ctx: &mut Context,
        cfg: &mut ConvertConfig,
        device: &Device,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        width: u32,
        height: u32,
        scale: Vec2,
    ) {
        let ortho: ProjectionMatrix = [
            [2.0f32 / width as f32, 0.0f32, 0.0f32, 0.0f32],
            [0.0f32, -2.0f32 / height as f32, 0.0f32, 0.0f32],
            [0.0f32, 0.0f32, 0.0f32, 0.0f32],
            [-1.0f32, 1.0f32, 0.0f32, 1.0f32],
        ];
        cfg.set_vertex_layout(&self.vle);
        cfg.set_vertex_size(size_of::<Vertex>());


        {
            let ubf_stage = device.create_buffer_init(&BufferInitDescriptor {
                label: Some("nuklear-wgpu-stage-projection-matrix-buffer"),
                usage: BufferUsage::COPY_SRC,
                contents: as_typed_slice(&ortho),
            });

            encoder.copy_buffer_to_buffer(&ubf_stage, 0, &self.ubf, 0, size_of::<ProjectionMatrix>() as u64);
        }

        {
            let vbf_stage = device.create_buffer(&BufferDescriptor {
                label: Some("nuklear-wgpu-stage-vertex-buffer"),
                size: self.vsz as BufferAddress,
                usage: BufferUsage::COPY_SRC,
                mapped_at_creation: true,
            });
            let ebf_stage = device.create_buffer(&BufferDescriptor {
                label: Some("nuklear-wgpu-stage-index-buffer"),
                size: self.esz as BufferAddress,
                usage: BufferUsage::COPY_SRC,
                mapped_at_creation: true,
            });

            ctx.convert(
                &mut self.cmd,
                &mut NkBuffer::with_fixed(vbf_stage.slice(..).get_mapped_range_mut().as_mut()),
                &mut NkBuffer::with_fixed(ebf_stage.slice(..).get_mapped_range_mut().as_mut()),
                cfg,
            );

            vbf_stage.unmap();
            ebf_stage.unmap();

            encoder.copy_buffer_to_buffer(&vbf_stage, 0, &self.vbf, 0, self.vsz as BufferAddress);
            encoder.copy_buffer_to_buffer(&ebf_stage, 0, &self.ebf, 0, self.esz as BufferAddress);
        };

        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[RenderPassColorAttachmentDescriptor {
                attachment: view,
                ops: Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
                resolve_target: None,
            }],
            depth_stencil_attachment: None,
        });

        rpass.set_pipeline(&self.pso);

        rpass.set_vertex_buffer(0, self.vbf.slice(..));
        rpass.set_index_buffer(self.ebf.slice(..));

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

fn compile_glsl(code: &str, stage: ShaderStage) -> ShaderModuleSource {
    use std::io::Read;
    use wgpu::util::make_spirv;

    let ty = match stage {
        ShaderStage::Vertex => glsl_to_spirv::ShaderType::Vertex,
        ShaderStage::Fragment => glsl_to_spirv::ShaderType::Fragment,
    };

    let spv = {
        let mut spv = Vec::new();
        glsl_to_spirv::compile(code, ty)
            .expect("Failed to compile GLSL")
            .read_to_end(&mut spv)
            .expect("Failed to read GLSL compiled to SPIR-V");
        spv
    };
    let spirv = make_spirv(&spv);
    match spirv {
        ShaderModuleSource::SpirV(cow) => ShaderModuleSource::SpirV(std::borrow::Cow::Owned(cow.into())),
        _ => unreachable!()
    }
}
