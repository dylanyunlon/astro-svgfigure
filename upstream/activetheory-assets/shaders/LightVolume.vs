
#require(instance.vs)
#require(rotation.glsl)

void main() {
    vec3 pos = transformPosition(position, offset * uSeparation, uScale);
    pos = vec3(vec4(pos, 1.0) * rotationMatrix(vec3(0.0, 0.0, 1.0), radians(360.0 * 0.1 * offset.z * uOffset)));

    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);

    vUv = uv;
    vPos = pos;
    vAttribs = attribs;
    vOffset = offset.z * 10.0;
}