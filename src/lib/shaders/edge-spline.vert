attribute vec2 a_position;
attribute float a_t;
uniform mat3 u_projectionMatrix;
uniform float u_thickness;
varying float v_t;
void main(){
  v_t=a_t;
  vec3 pos=u_projectionMatrix*vec3(a_position,1.0);
  gl_Position=vec4(pos.xy,0.0,1.0);
  gl_PointSize=u_thickness;
}
