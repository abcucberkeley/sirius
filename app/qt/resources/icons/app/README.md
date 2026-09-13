# Application icon

`sirius-app.svg` is the source: the viewer ground (`#0a0909`) with a
`neutral-700` edge, the brand's accent square (`#ec3013`) and a four-point star
(`#f3f2f2`), after the tokens in `docs/design/README.md`.

The PNGs are renditions of it, used for the window icon (Qt reads PNG without
any plugin; an SVG icon needs Qt's SVG icon engine, which a deployment may
not carry) and installed into the hicolor theme on Linux. Regenerate them after
editing the SVG:

    for s in 16 24 32 48 64 128 256; do
        inkscape sirius-app.svg --export-type=png --export-filename=sirius-app-$s.png -w $s -h $s
    done
