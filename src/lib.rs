#![cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))] // TODO later

use nuklear::{Buffer as NkBuffer, Context, ConvertConfig, DrawVertexLayoutAttribute, DrawVertexLayoutElements, DrawVertexLayoutFormat, Handle, Vec2};
use wgpu::*;
use std::mem::size_of;

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
    smp: Sampler,
    tex: Vec<Texture>,
    vbf: Buffer,
    ebf: Buffer,
    lbf: Buffer,
    vsz: usize,
    esz: usize,
    vle: DrawVertexLayoutElements,

    pub col: Option<Color>,
}

impl Drawer {
    pub fn new<F>(device: &mut Device, col: Color, texture_count: usize, vbo_size: usize, ebo_size: usize, command_buffer: NkBuffer) -> Drawer {
        let vs: &[u8] = include_bytes!("../shaders/vs.fx");
        let fs: &[u8] = include_bytes!("../shaders/ps.fx");

        let vs = device.create_shader_module(vs);
        let fs = device.create_shader_module(fs);

        Drawer {
            cmd: command_buffer,
            col: Some(col),
            smp: device.create_sampler(&SamplerDescriptor {
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
                    format: TextureFormat::Rgba8UnormSrgb,
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
                    stride: (size_of::<f32>() * 4 + size_of::<u8>() * 4) as u32,
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
            vbf: device.create_buffer(&BufferDescriptor {
			      size: vbo_size as u32,
			      usage: BufferUsageFlags::VERTEX | BufferUsageFlags::TRANSFER_DST,
			}),
            ebf: device.create_buffer(&BufferDescriptor {
			      size: ebo_size as u32,
			      usage: BufferUsageFlags::INDEX | BufferUsageFlags::TRANSFER_DST,
			}),
            vsz: vbo_size,
            esz: ebo_size,
            lbf: device.create_buffer(&BufferDescriptor {
			      size: size_of::<f32>() as u32 * 16, 
			      usage: BufferUsageFlags::UNIFORM | BufferUsageFlags::TRANSFER_DST,
			}),
            vle: DrawVertexLayoutElements::new(&[
                (DrawVertexLayoutAttribute::NK_VERTEX_POSITION, DrawVertexLayoutFormat::NK_FORMAT_FLOAT, 0),
                (DrawVertexLayoutAttribute::NK_VERTEX_TEXCOORD, DrawVertexLayoutFormat::NK_FORMAT_FLOAT, size_of::<f32>() as u32 * 2),
                (DrawVertexLayoutAttribute::NK_VERTEX_COLOR, DrawVertexLayoutFormat::NK_FORMAT_R8G8B8A8, size_of::<f32>() as u32 * 4),
                (DrawVertexLayoutAttribute::NK_VERTEX_ATTRIBUTE_COUNT, DrawVertexLayoutFormat::NK_FORMAT_COUNT, 0u32),
            ]),
        }
    }

    pub fn add_texture(&mut self, device: &mut Device, image: &[u8], width: u32, height: u32) -> Handle {
        let (_, view) = factory.create_texture_immutable_u8::<ColorFormat>(Kind::D2(width as u16, height as u16, AaMode::Single), Mipmap::Provided, &[image]).unwrap();

        self.tex.push(view);

        Handle::from_id(self.tex.len() as i32)
    }

    pub fn draw(&mut self, ctx: &mut Context, cfg: &mut ConvertConfig, encoder: &mut Encoder, device: &mut Device, width: u32, height: u32, scale: Vec2) {
        use gfx::IntoIndexBuffer;

        if self.col.clone().is_none() {
            return;
        }

        let ortho = [
            [2.0f32 / width as f32, 0.0f32, 0.0f32, 0.0f32],
            [0.0f32, -2.0f32 / height as f32, 0.0f32, 0.0f32],
            [0.0f32, 0.0f32, -1.0f32, 0.0f32],
            [-1.0f32, 1.0f32, 0.0f32, 1.0f32],
        ];

        cfg.set_vertex_layout(&self.vle);
        cfg.set_vertex_size(::std::mem::size_of::<Vertex>());

        {
            let mut rwv = factory.write_mapping(&self.vbf).unwrap();
            let mut rvbuf = unsafe { ::std::slice::from_raw_parts_mut(&mut *rwv as *mut [Vertex] as *mut u8, ::std::mem::size_of::<Vertex>() * self.vsz) };
            let mut vbuf = NkBuffer::with_fixed(&mut rvbuf);

            let mut rwe = factory.write_mapping(&self.ebf).unwrap();
            let mut rebuf = unsafe { ::std::slice::from_raw_parts_mut(&mut *rwe as *mut [u16] as *mut u8, ::std::mem::size_of::<u16>() * self.esz) };
            let mut ebuf = NkBuffer::with_fixed(&mut rebuf);

            ctx.convert(&mut self.cmd, &mut vbuf, &mut ebuf, cfg);
        }

        let mut slice = ::gfx::Slice {
            start: 0,
            end: 0,
            base_vertex: 0,
            instances: None,
            buffer: self.ebf.clone().into_index_buffer(factory),
        };

        encoder.update_constant_buffer(&self.lbf, &Locals { proj: ortho });

        for cmd in ctx.draw_command_iterator(&self.cmd) {
            if cmd.elem_count() < 1 {
                continue;
            }

            slice.end = slice.start + cmd.elem_count();

            let id = cmd.texture().id().unwrap();

            let x = cmd.clip_rect().x * scale.x;
            let y = cmd.clip_rect().y * scale.y;
            let w = cmd.clip_rect().w * scale.x;
            let h = cmd.clip_rect().h * scale.y;

            let sc_rect = gfx::Rect {
                x: (if x < 0f32 { 0f32 } else { x }) as u16,
                y: (if y < 0f32 { 0f32 } else { y }) as u16,
                w: (if x < 0f32 { w + x } else { w }) as u16,
                h: (if y < 0f32 { h + y } else { h }) as u16,
            };

            let res = self.find_res(id).unwrap();

            let data = pipe::Data {
                vbuf: self.vbf.clone(),
                tex: (res, self.smp.clone()),
                output: self.col.clone().unwrap(),
                scissors: sc_rect,
                locals: self.lbf.clone(),
            };

            encoder.draw(&slice, &self.pso, &data);

            slice.start = slice.end;
        }
    }

    fn find_res(&self, id: i32) -> Option<ShaderResourceView<R, [f32; 4]>> {
        if id > 0 && id as usize <= self.tex.len() {
            Some(self.tex[(id - 1) as usize].clone())
        } else {
            None
        }
    }
}
