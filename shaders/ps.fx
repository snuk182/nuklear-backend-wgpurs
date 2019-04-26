#version 150
precision mediump float;

layout(set = 1, binding = 0) uniform texture2D Texture;
layout(set = 1, binding = 0) uniform sampler Sampler;

layout(location = 0) in vec2 Frag_UV;
layout(location = 1) in vec4 Frag_Color;

layout(location = 0) out vec4 Target0;
void main(){
   Target0 = Frag_Color * texture(sampler2D(Texture, Sampler), Frag_UV.st);
}
