//!HOOK MAIN
//!BIND HOOKED
//!DESC soft vignette — darken the edges of the frame

vec4 hook() {
    vec4 c = HOOKED_tex(HOOKED_pos);
    // HOOKED_pos is in 0..1 normalized coords. Distance from center 0.5,0.5.
    vec2 uv = HOOKED_pos - vec2(0.5, 0.5);
    // Multiply by 1.4 so the edges go fully dark before the corner;
    // smoothstep gives a soft falloff from center to edge.
    float r = length(uv) * 1.4;
    float v = smoothstep(0.35, 1.0, r);
    // Strength of darkening at the corners (0.0 = no effect, 1.0 = black corners)
    float strength = 0.55;
    return vec4(c.rgb * (1.0 - v * strength), c.a);
}
