#![cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))] // TODO later

use nuklear::{Buffer as NkBuffer, Context, ConvertConfig, DrawVertexLayoutAttribute, DrawVertexLayoutElements, DrawVertexLayoutFormat, Handle, Vec2};
use std::{mem::size_of, slice::from_raw_parts};
use wgpu::*;

const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

#[allow(dead_code)]
struct Vertex {
    pos: [f32; 2], // "Position",
    tex: [f32; 2], // "TexCoord",
    col: [u8; 4],  // "Color",
}
#[allow(dead_code)]
struct WgpuTexture {
    texture: Texture,
    sampler: Sampler,
    
    pub bind_group: BindGroup,
}

impl WgpuTexture {
    pub fn new(device: &mut Device, drawer: &Drawer, image: &[u8], width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: Extent3d { width: width, height: height, depth: 1 },
            array_size: 1,
            dimension: TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: TextureUsageFlags::SAMPLED | TextureUsageFlags::TRANSFER_DST,
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            r_address_mode: AddressMode::ClampToEdge,
            s_address_mode: AddressMode::ClampToEdge,
            t_address_mode: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            max_anisotropy: 0,
            compare_function: CompareFunction::Always,
            border_color: BorderColor::TransparentBlack,
        });

        let bytes = image.len() as u32;
        let buffer = device.create_buffer(&BufferDescriptor {
            size: bytes,
            usage: BufferUsageFlags::TRANSFER_SRC,
        });
        buffer.set_sub_data(0, image);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { todo: 0 });

        let pixel_size = bytes / width / height;
        encoder.copy_buffer_to_texture(
            BufferCopyView {
                buffer: &buffer,
                offset: 0,
                row_pitch: pixel_size * width,
                image_height: height,
            },
            TextureCopyView {
                texture: &texture,
                level: 0,
                slice: 0,
                origin: Origin3d { x: 0.0, y: 0.0, z: 0.0 },
            },
            Extent3d { width, height, depth: 1 },
        );

        device.get_queue().submit(&[encoder.finish()]);

        WgpuTexture {
            bind_group: device.create_bind_group(&BindGroupDescriptor {
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

/*gfx_defines!{
    vertex Vertex {
        pos: [f32; 2] = "Position",
        tex: [f32; 2] = "TexCoord",
        col: [U8Norm; 4] = "Color",
    }

    constant Locals {
        proj: [[f32; 4]; 4] = "ProjMtx",
    }

    pipeline pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        tex: gfx::TextureSampler<[f32; 4]> = "Texture",
        output: gfx::BlendTarget<super::ColorFormat> = ("Target0", gfx::state::ColorMask::all(), gfx::preset::blend::ALPHA),
        locals: gfx::ConstantBuffer<Locals> = "Locals",
        scissors: gfx::Scissor = (),
    }
}

impl Default for Vertex {
    fn default() -> Self {
        unsafe { ::std::mem::zeroed() }
    }
}
*/
pub struct Drawer {
    cmd: NkBuffer,
    pso: RenderPipeline,
    tla: BindGroupLayout,
    tex: Vec<WgpuTexture>,
    //vbf: Buffer,
    //ebf: Buffer,
    ubf: Buffer,
    ubg: BindGroup,
    vsz: usize,
    esz: usize,
    vle: DrawVertexLayoutElements,

    pub col: Option<Color>,
}

impl Drawer {
    pub fn new(device: &mut Device, col: Color, texture_count: usize, vbo_size: usize, ebo_size: usize, command_buffer: NkBuffer) -> Drawer {
        let vs: &[u8] = include_bytes!("../shaders/vs.fx");
        let fs: &[u8] = include_bytes!("../shaders/ps.fx");

        let vs = device.create_shader_module(vs);
        let fs = device.create_shader_module(fs);

        let ubf = device.create_buffer(&BufferDescriptor {
            size: size_of::<f32>() as u32 * 16,
            usage: BufferUsageFlags::UNIFORM | BufferUsageFlags::TRANSFER_DST,
        });

        Drawer {
            cmd: command_buffer,
            col: Some(col),
            tla: device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                bindings: &[
                    BindGroupLayoutBinding {
                        binding: 0,
                        visibility: ShaderStageFlags::FRAGMENT,
                        ty: BindingType::SampledTexture,
                    },
                    BindGroupLayoutBinding {
                        binding: 1,
                        visibility: ShaderStageFlags::FRAGMENT,
                        ty: BindingType::Sampler,
                    },
                ],
            }),
            pso: device.create_render_pipeline(&RenderPipelineDescriptor {
                layout: &device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    bind_group_layouts: &[
                        &device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                            bindings: &[BindGroupLayoutBinding {
                                binding: 0,
                                visibility: ShaderStageFlags::VERTEX,
                                ty: BindingType::UniformBuffer,
                            }],
                        }),
                        &device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                            bindings: &[
                                BindGroupLayoutBinding {
                                    binding: 0,
                                    visibility: ShaderStageFlags::FRAGMENT,
                                    ty: BindingType::SampledTexture,
                                },
                                BindGroupLayoutBinding {
                                    binding: 1,
                                    visibility: ShaderStageFlags::FRAGMENT,
                                    ty: BindingType::Sampler,
                                },
                            ],
                        }),
                    ],
                }),
                vertex_stage: PipelineStageDescriptor { module: &vs, entry_point: "main" },
                fragment_stage: PipelineStageDescriptor { module: &fs, entry_point: "main" },
                rasterization_state: RasterizationStateDescriptor {
                    front_face: FrontFace::Cw,
                    cull_mode: CullMode::None,
                    depth_bias: 0,
                    depth_bias_slope_scale: 0.0,
                    depth_bias_clamp: 0.0,
                },
                primitive_topology: PrimitiveTopology::TriangleList,
                color_states: &[ColorStateDescriptor {
                    format: TEXTURE_FORMAT,
                    color: BlendDescriptor {
                        src_factor: BlendFactor::SrcAlpha,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendDescriptor {
                        src_factor: BlendFactor::OneMinusDstAlpha,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    write_mask: ColorWriteFlags::ALL,
                }],
                depth_stencil_state: None,
                index_format: IndexFormat::Uint16,
                vertex_buffers: &[VertexBufferDescriptor {
                    stride: (size_of::<Vertex>()) as u32,
                    step_mode: InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            format: VertexFormat::Float2,
                            attribute_index: 0,
                            offset: 0,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: VertexFormat::Float2,
                            attribute_index: 1,
                            offset: 8,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: VertexFormat::Uint,
                            attribute_index: 2,
                            offset: 16,
                        },
                    ],
                }],
                sample_count: 1,
            }),
            tex: Vec::with_capacity(texture_count + 1),
            vsz: vbo_size,
            esz: ebo_size,
            ubg: device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    bindings: &[BindGroupLayoutBinding {
                        binding: 0,
                        visibility: ShaderStageFlags::VERTEX,
                        ty: BindingType::UniformBuffer,
                    }],
                }),
                bindings: &[Binding {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &ubf,
                        range: 0..(size_of::<f32>() as u32 * 16),
                    },
                }],
            }),
            ubf: ubf,
            vle: DrawVertexLayoutElements::new(&[
                (DrawVertexLayoutAttribute::NK_VERTEX_POSITION, DrawVertexLayoutFormat::NK_FORMAT_FLOAT, 0),
                (DrawVertexLayoutAttribute::NK_VERTEX_TEXCOORD, DrawVertexLayoutFormat::NK_FORMAT_FLOAT, size_of::<f32>() as u32 * 2),
                (DrawVertexLayoutAttribute::NK_VERTEX_COLOR, DrawVertexLayoutFormat::NK_FORMAT_R8G8B8A8, size_of::<f32>() as u32 * 4),
                (DrawVertexLayoutAttribute::NK_VERTEX_ATTRIBUTE_COUNT, DrawVertexLayoutFormat::NK_FORMAT_COUNT, 0u32),
            ]),
        }
    }

    pub fn add_texture(&mut self, device: &mut Device, image: &[u8], width: u32, height: u32) -> Handle {
        //let (_, view) = factory.create_texture_immutable_u8::<ColorFormat>(Kind::D2(width as u16, height as u16, AaMode::Single), Mipmap::Provided, &[image]).unwrap();

        self.tex.push(WgpuTexture::new(device, self, image, width, height));
        Handle::from_id(self.tex.len() as i32)
    }

    pub fn draw(&mut self, ctx: &mut Context, cfg: &mut ConvertConfig, encoder: &mut CommandEncoder, view: &TextureView, device: &mut Device, width: u32, height: u32, scale: Vec2) {
        /*if self.col.clone().is_none() {
            return;
        }*/
        let ortho = [
            [2.0f32 / width as f32, 0.0f32, 0.0f32, 0.0f32],
            [0.0f32, -2.0f32 / height as f32, 0.0f32, 0.0f32],
            [0.0f32, 0.0f32, -1.0f32, 0.0f32],
            [-1.0f32, 1.0f32, 0.0f32, 1.0f32],
        ];
        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[RenderPassColorAttachmentDescriptor {
                attachment: &view,
                load_op: match self.col {
                    Some(_) => wgpu::LoadOp::Clear,
                    _ => wgpu::LoadOp::Load,
                },
                store_op: StoreOp::Store,
                clear_color: self.col.unwrap_or(Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
            }],
            depth_stencil_attachment: None,
        });
        rpass.set_pipeline(&self.pso);

        cfg.set_vertex_layout(&self.vle);
        cfg.set_vertex_size(size_of::<Vertex>());

        let mut vbf = device.create_buffer_mapped(self.vsz, BufferUsageFlags::VERTEX | BufferUsageFlags::TRANSFER_DST);
        let mut ebf = device.create_buffer_mapped(self.esz, BufferUsageFlags::INDEX | BufferUsageFlags::TRANSFER_DST);
        {
            let mut vbuf = NkBuffer::with_fixed(&mut vbf.data);
            let mut ebuf = NkBuffer::with_fixed(&mut ebf.data);

            ctx.convert(&mut self.cmd, &mut vbuf, &mut ebuf, cfg);
        }
        let vbf = vbf.finish();
        let ebf = ebf.finish();

        rpass.set_vertex_buffers(&[(&vbf, 0)]);
        rpass.set_index_buffer(&ebf, 0);

        /*let mut slice = ::gfx::Slice {
            start: 0,
            end: 0,
            base_vertex: 0,
            instances: None,
            buffer: self.ebf.clone().into_index_buffer(factory),
        };*/
        self.ubf.set_sub_data(0, as_typed_slice(&ortho));
        rpass.set_bind_group(0, &self.ubg);
        //encoder.update_constant_buffer(&self.lbf, &Locals { proj: ortho });

        let mut start = 0;
        let mut end = 0;

        for cmd in ctx.draw_command_iterator(&self.cmd) {
            if cmd.elem_count() < 1 {
                continue;
            }

            end = start + cmd.elem_count();

            let x = cmd.clip_rect().x * scale.x;
            let y = cmd.clip_rect().y * scale.y;
            let w = cmd.clip_rect().w * scale.x;
            let h = cmd.clip_rect().h * scale.y;

            let id = cmd.texture().id().unwrap();
            let res = self.find_res(id).unwrap();

            rpass.set_bind_group(1, &res.bind_group);

            end = start + cmd.elem_count();
            let scissor = (
                (if x < 0f32 { 0f32 } else { x }) as u16,
                (if y < 0f32 { 0f32 } else { y }) as u16,
                (if x < 0f32 { w + x } else { w }) as u16,
                (if y < 0f32 { h + y } else { h }) as u16,
            );

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
