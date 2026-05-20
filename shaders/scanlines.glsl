//!HOOK MAIN
//!BIND HOOKED
//!DESC CRT scanlines — horizontal alternating-brightness bands

vec4 hook() {
    vec4 c = HOOKED_tex(HOOKED_pos);
    // Density: ~half the screen height worth of bands at 1080p
    float scan_freq = target_size.y * 1.2;
    // sin-based mask: 0 in dark bands, 1 in bright bands, smoothed
    float scan = sin(HOOKED_pos.y * scan_freq) * 0.5 + 0.5;
    // Strength: how dark the dark bands get (0.0 = no effect, 0.5 = strong CRT)
    float strength = 0.22;
    float modulated = mix(1.0 - strength, 1.0, scan);
    return vec4(c.rgb * modulated, c.a);
}
