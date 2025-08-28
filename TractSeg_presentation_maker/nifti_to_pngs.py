#!/usr/bin/env python3
import argparse, sys
from pathlib import Path

def parse_pair(s, typ=float, name="pair"):
    try:
        a,b = [typ(x.strip()) for x in s.split(",")]
        return a,b
    except Exception:
        sys.exit(f"--{name} must be 'a,b'")

def parse_rgb(s: str):
    try:
        r,g,b = [int(x.strip()) for x in s.split(",")]
        if any(v<0 or v>255 for v in (r,g,b)): raise ValueError
        return r,g,b
    except Exception:
        sys.exit("--solid-color must be 'R,G,B' in 0..255")

def robust_window(image, p_low=2.0, p_high=98.0, bins=4096):
    import vtk
    sr = image.GetScalarRange()
    if sr[1] <= sr[0]:
        return (0.0, 1.0)
    acc = vtk.vtkImageAccumulate()
    acc.SetInputData(image)
    acc.SetComponentExtent(0,bins-1,0,0,0,0)
    acc.SetComponentOrigin(sr[0],0,0)
    acc.SetComponentSpacing((sr[1]-sr[0])/bins,0,0)
    acc.Update()
    scalars = acc.GetOutput().GetPointData().GetScalars()
    total = 0
    n = scalars.GetNumberOfTuples()
    counts = [0]*n
    for i in range(n):
        c = int(scalars.GetTuple1(i))
        counts[i] = c
        total += c
    if total == 0:
        return sr
    tgt_low  = total * (p_low/100.0)
    tgt_high = total * (p_high/100.0)
    run = 0
    low_idx = 0
    for i,c in enumerate(counts):
        run += c
        if run >= tgt_low:
            low_idx = i; break
    run = 0
    high_idx = n-1
    for i,c in enumerate(counts):
        run += c
        if run >= tgt_high:
            high_idx = i; break
    wmin = sr[0] + low_idx*(sr[1]-sr[0])/bins
    wmax = sr[0] + high_idx*(sr[1]-sr[0])/bins
    if wmax <= wmin: wmax = wmin + 1e-3
    return (wmin, wmax)

def add_cmap_points(ctf, wmin, wmax, name="gray", invert=False):
    # Simple, presentation-friendly maps
    def add(points):
        for x,r,g,b in (points[::-1] if invert else points):
            ctf.AddRGBPoint(x, r, g, b)
    if name == "gray":
        add([(wmin,0,0,0),(wmax,1,1,1)])
    elif name == "hot":
        add([(wmin,0,0,0),(wmin+0.4*(wmax-wmin),1,0,0),
             (wmin+0.7*(wmax-wmin),1,1,0),(wmax,1,1,1)])
    elif name == "cool":
        add([(wmin,0,0,0.2),(wmin+0.5*(wmax-wmin),0,1,1),(wmax,1,1,1)])
    elif name == "viridis":
        pts = [(0.0, 0.267, 0.005, 0.329),
               (0.3, 0.283, 0.141, 0.458),
               (0.6, 0.254, 0.265, 0.530),
               (0.8, 0.190, 0.407, 0.556),
               (0.9, 0.208, 0.593, 0.554),
               (1.0, 0.993, 0.906, 0.144)]
        mapped = [(wmin+(wmax-wmin)*t, r,g,b) for (t,r,g,b) in pts]
        add(mapped)
    elif name == "magma":
        pts = [(0.0, 0.001, 0.000, 0.014),
               (0.25,0.246,0.036,0.205),
               (0.5, 0.496,0.090,0.274),
               (0.75,0.788,0.288,0.235),
               (1.0, 0.987,0.991,0.749)]
        mapped = [(wmin+(wmax-wmin)*t, r,g,b) for (t,r,g,b) in pts]
        add(mapped)
    elif name == "rainbow":
        add([(wmin,0,0,1),(wmin+0.25*(wmax-wmin),0,1,1),
             (wmin+0.5*(wmax-wmin),0,1,0),
             (wmin+0.75*(wmax-wmin),1,1,0),(wmax,1,0,0)])
    else:
        add([(wmin,0,0,0),(wmax,1,1,1)])

def render_nifti_to_png(
    nifti_path: Path, out_dir: Path, prefix: str,
    mode: str, cmap: str, invert_cmap: bool, solid_rgb,
    width: int, height: int, perspective: bool,
    window, robust, opacity: float, shade: bool,
    bg_nifti, bg_mode, bg_window, bg_opacity,
    slice_thickness, slice_opacity, slice_placement, slice_margin,
    # ISO-specific:
    surface_opacity: float, iso_frac: float, iso_level, iso_gauss_mm: float, surface_smooth_iters: int
):
    import vtk

    # --- Read main volume ---
    main_reader = vtk.vtkNIFTIImageReader()
    main_reader.SetFileName(str(nifti_path))
    main_reader.Update()
    main_img = main_reader.GetOutput()
    # Grab foreground world transform (sform/qform)
    m_s = main_reader.GetSFormMatrix(); m_q = main_reader.GetQFormMatrix()
    mat_main = m_s if m_s is not None else m_q

    # Window/level
    if window is not None:
        wmin, wmax = window
    elif robust is not None:
        wmin, wmax = robust_window(main_img, p_low=robust[0], p_high=robust[1])
    else:
        wmin, wmax = main_img.GetScalarRange()

    # --- Renderer, window, camera ---
    ren = vtk.vtkRenderer()
    ren.SetBackground(0,0,0)
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetMultiSamples(8)  # anti-aliasing for cleaner edges
    win.AddRenderer(ren)
    win.SetSize(width, height)
    cam = ren.GetActiveCamera()
    if not perspective:
        cam.ParallelProjectionOn()

    # --- Build foreground: volume or isosurface ---
    if mode == "iso":
        # choose input for MC (optionally pre-smooth in mm)
        iso_input = main_reader.GetOutputPort()
        if iso_gauss_mm > 0:
            sx,sy,sz = main_img.GetSpacing()
            def std(mm, sp): return mm/max(sp,1e-6)
            gs = vtk.vtkImageGaussianSmooth()
            gs.SetInputConnection(main_reader.GetOutputPort())
            gs.SetStandardDeviations(std(iso_gauss_mm,sx), std(iso_gauss_mm,sy), std(iso_gauss_mm,sz))
            gs.SetRadiusFactors(2,2,2)
            gs.Update()
            iso_input = gs.GetOutputPort()
        # iso threshold
        iso_val = iso_level if iso_level is not None else (wmin + iso_frac*(wmax - wmin))
        mc = vtk.vtkMarchingCubes()
        mc.SetInputConnection(iso_input)
        mc.SetValue(0, iso_val)
        mc.Update()
        poly = mc.GetOutput()
        # optional surface smoothing
        if surface_smooth_iters > 0:
            smooth = vtk.vtkWindowedSincPolyDataFilter()
            smooth.SetInputData(poly)
            smooth.SetNumberOfIterations(surface_smooth_iters)
            smooth.SetPassBand(0.1)
            smooth.BoundarySmoothingOn()
            smooth.NonManifoldSmoothingOn()
            smooth.NormalizeCoordinatesOn()
            smooth.Update()
            poly = smooth.GetOutput()
        # normals + mapper
        norms = vtk.vtkPolyDataNormals()
        norms.SetInputData(poly)
        norms.SetFeatureAngle(80)
        norms.SplittingOff()
        norms.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(norms.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if solid_rgb:
            r,g,b = solid_rgb
            actor.GetProperty().SetColor(r/255.0,g/255.0,b/255.0)
        else:
            actor.GetProperty().SetColor(1,1,1)
        actor.GetProperty().SetAmbient(0.2)
        actor.GetProperty().SetDiffuse(0.8)
        actor.GetProperty().SetSpecular(0.1)
        actor.GetProperty().SetOpacity(max(0.0, min(1.0, surface_opacity)))
        if mat_main is not None:
            actor.SetUserMatrix(mat_main)            # <— align iso with NIfTI space
        ren.AddActor(actor)
        ren.ResetCameraClippingRange()
        data_bounds = actor.GetBounds()              # <— use world-space bounds
    else:
        # volume ray cast with TFs
        import vtk
        ctf = vtk.vtkColorTransferFunction()
        add_cmap_points(ctf, wmin, wmax, name=cmap, invert=invert_cmap)
        otf = vtk.vtkPiecewiseFunction()
        # Gentle ramp: nothing below wmin, readable at wmax
        otf.AddPoint(wmin, 0.0)
        otf.AddPoint(0.5*(wmin+wmax), opacity*0.5)
        otf.AddPoint(wmax, opacity)
        prop = vtk.vtkVolumeProperty()
        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)
        prop.SetInterpolationTypeToLinear()
        if shade:
            prop.ShadeOn()
            prop.SetAmbient(0.15); prop.SetDiffuse(0.9); prop.SetSpecular(0.05)
        else:
            prop.ShadeOff()
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputConnection(main_reader.GetOutputPort())
        vol = vtk.vtkVolume()
        vol.SetMapper(mapper)
        vol.SetProperty(prop)
        if mat_main is not None:
            vol.SetUserMatrix(mat_main)              # <— align volume with NIfTI space
        ren.AddVolume(vol)
        ren.ResetCameraClippingRange()
        data_bounds = vol.GetBounds()                # <— world-space bounds

    # --- Optional background (slice or soft volume) ---
    plane = None  # used for slice placement per view
    if bg_nifti:
        # be robust to missing/bad files
        p = Path(bg_nifti)
        if not p.is_file():
            print(f"[WARN] Background NIfTI not found: {bg_nifti} — continuing without BG")
            bg_nifti = None
        else:
            bg_reader = vtk.vtkNIFTIImageReader()
            bg_reader.SetFileName(bg_nifti)
            bg_reader.Update()
            bg_img = bg_reader.GetOutput()
            if bg_img is None or bg_img.GetDimensions() == (0,0,0):
                print(f"[WARN] Failed to read background NIfTI: {bg_nifti} — continuing without BG")
                bg_nifti = None
    if bg_nifti:
        import vtk
        bg_reader = vtk.vtkNIFTIImageReader()
        bg_reader.SetFileName(bg_nifti)
        bg_reader.Update()
        bg_img = bg_reader.GetOutput()

        if bg_window is not None:
            bwmin,bwmax = bg_window
        else:
            bwmin,bwmax = robust_window(bg_img, 2, 98)

        sform = bg_reader.GetSFormMatrix(); qform = bg_reader.GetQFormMatrix()
        mat = sform if sform is not None else qform

        if bg_mode == "slices":
            smap = vtk.vtkImageResliceMapper()
            smap.SetInputConnection(bg_reader.GetOutputPort())
            smap.SetSlabThickness(max(0.1, slice_thickness))
            smap.SetSlabTypeToMean()
            plane = vtk.vtkPlane()
            smap.SetSlicePlane(plane)
            bg_slice = vtk.vtkImageSlice()
            bg_slice.SetMapper(smap)
            isp = bg_slice.GetProperty()
            isp.SetOpacity(max(0.0, min(1.0, slice_opacity)))
            isp.SetColorWindow(bwmax - bwmin)
            isp.SetColorLevel(0.5*(bwmin + bwmax))
            if mat is not None: bg_slice.SetUserMatrix(mat)
            ren.AddViewProp(bg_slice)
        else:
            ctf_bg = vtk.vtkColorTransferFunction()
            ctf_bg.AddRGBPoint(bwmin, 0,0,0); ctf_bg.AddRGBPoint(bwmax, 1,1,1)
            otf_bg = vtk.vtkPiecewiseFunction()
            mid = 0.5*(bwmin+bwmax)
            otf_bg.AddPoint(bwmin, 0.0)
            otf_bg.AddPoint(mid, bg_opacity*0.6)
            otf_bg.AddPoint(bwmax, bg_opacity)
            prop_bg = vtk.vtkVolumeProperty()
            prop_bg.SetColor(ctf_bg); prop_bg.SetScalarOpacity(otf_bg)
            prop_bg.SetInterpolationTypeToLinear(); prop_bg.ShadeOff()
            mapper_bg = vtk.vtkSmartVolumeMapper()
            mapper_bg.SetInputConnection(bg_reader.GetOutputPort())
            vol_bg = vtk.vtkVolume(); vol_bg.SetMapper(mapper_bg); vol_bg.SetProperty(prop_bg)
            if mat is not None: vol_bg.SetUserMatrix(mat)
            ren.AddVolume(vol_bg)

    # --- Camera fit & snapshots ---
    b = data_bounds
    cx,cy,cz = (0.5*(b[0]+b[1]), 0.5*(b[2]+b[3]), 0.5*(b[4]+b[5]))
    dx,dy,dz = (b[1]-b[0], b[3]-b[2], b[5]-b[4])
    max_dim = max(dx,dy,dz); pad = 0.6*max_dim + 1e-3

    def place_bg_slice():
        if plane is None: return
        fx,fy,fz = cam.GetFocalPoint(); px,py,pz = cam.GetPosition()
        vx,vy,vz = (fx-px, fy-py, fz-pz)
        import math
        vm = math.sqrt(vx*vx+vy*vy+vz*vz) or 1.0
        ux,uy,uz = (vx/vm, vy/vm, vz/vm)
        corners = [(b[i0], b[i1], b[i2]) for i0 in (0,1) for i1 in (2,3) for i2 in (4,5)]
        projs = [ (cx1-px)*ux + (cy1-py)*uy + (cz1-pz)*uz for (cx1,cy1,cz1) in corners ]
        near_t, far_t = min(projs), max(projs)
        if slice_placement == "behind":
            t = far_t + max(0.0, slice_margin)
        elif slice_placement == "front":
            t = near_t - max(0.0, slice_margin)
        else:
            t = 0.5*(near_t+far_t)
        ox,oy,oz = (px + ux*t, py + uy*t, pz + uz*t)
        plane.SetNormal(ux,uy,uz)
        plane.SetOrigin(ox,oy,oz)

    def snap(name, pos, view_up):
        cam.SetFocalPoint(cx,cy,cz); cam.SetPosition(*pos); cam.SetViewUp(*view_up)
        if not perspective:
            cam.SetParallelScale(max_dim*0.6)
        place_bg_slice()
        ren.ResetCameraClippingRange()
        win.Render()
        w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.Update()
        writer = vtk.vtkPNGWriter()
        out_path = out_dir / f"{prefix}_{name}.png"
        writer.SetFileName(str(out_path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        print(f"Saved {out_path}")

    views = [
        ("posX", (b[1]+pad, cy, cz), (0,0,1)),
        ("negX", (b[0]-pad, cy, cz), (0,0,1)),
        ("posY", (cx, b[3]+pad, cz), (0,0,1)),
        ("negY", (cx, b[2]-pad, cz), (0,0,1)),
        ("posZ", (cx, cy, b[5]+pad), (0,1,0)),
        ("negZ", (cx, cy, b[4]-pad), (0,1,0)),
    ]
    return snap, views

def main():
    ap = argparse.ArgumentParser(
        description="Render 3D PNGs of a NIfTI volume (.nii/.nii.gz) with optional background. Input can be a single file or a directory."
    )
    ap.add_argument("input_path", help="Path to a .nii/.nii.gz file OR a directory containing such files")
    ap.add_argument("--out-dir", default="nifti_renders", help="Output folder (root for batch mode)")
    ap.add_argument("--prefix", default=None, help="Output file prefix. In batch mode + --flat, actual prefix becomes '<prefix>_<file_stem>'.")
    ap.add_argument("--recursive", action="store_true", help="When input is a directory, also scan subfolders")
    ap.add_argument("--pattern", default="*.nii*", help="Glob pattern for directory input (default: *.nii*)")
    ap.add_argument("--flat", action="store_true", help="Batch mode: write all images directly under --out-dir (prefixing filenames).")

    ap.add_argument("--mode", choices=["volume","iso"], default="volume", help="Foreground rendering mode")
    ap.add_argument("--cmap", choices=["gray","viridis","magma","hot","cool","rainbow"], default="gray", help="Colormap (volume mode)")
    ap.add_argument("--invert-cmap", action="store_true", help="Invert the chosen colormap")
    ap.add_argument("--solid-color", type=str, default=None, help="ISO mode: solid color 'R,G,B'")
    ap.add_argument("--surface-opacity", type=float, default=1.0, help="ISO mode: surface opacity 0..1")
    ap.add_argument("--iso-frac", type=float, default=0.6, help="ISO mode: fraction within window [0..1] for threshold (ignored if --iso-level given)")
    ap.add_argument("--iso-level", type=float, help="ISO mode: absolute intensity threshold (overrides --iso-frac)")
    ap.add_argument("--iso-gauss-mm", type=float, default=0.0, help="ISO mode: Gaussian pre-smoothing in mm (0 = off)")
    ap.add_argument("--surface-smooth-iters", type=int, default=0, help="ISO mode: mesh smoothing iterations (0 = off)")

    ap.add_argument("--opacity", type=float, default=0.25, help="Volume opacity at upper window (0..1)")
    ap.add_argument("--shade", action="store_true", help="Enable shading for volume mode")
    ap.add_argument("--window", type=str, help="Intensity window for foreground as 'min,max' (overrides robust)")
    ap.add_argument("--robust", type=str, default="2,98", help="Percentiles for auto-window (e.g., '2,98'). Use --window to override.")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--perspective", action="store_true", help="Use perspective projection (default: orthographic)")

    # Background (optional)
    ap.add_argument("--bg-nifti", type=str, help="Optional background NIfTI")
    ap.add_argument("--bg-mode", choices=["slices","volume"], default="slices", help="Background display mode")
    ap.add_argument("--bg-window", type=str, help="Background intensity window 'min,max' (default: robust 2..98)")
    ap.add_argument("--bg-opacity", type=float, default=0.12, help="Background volume opacity (if bg-mode=volume)")
    ap.add_argument("--slice-thickness", type=float, default=6.0, help="Background slice slab thickness in mm (bg-mode=slices)")
    ap.add_argument("--slice-opacity", type=float, default=0.65, help="Background slice opacity 0..1")
    ap.add_argument("--slice-placement", choices=["behind","front","center"], default="behind",
                    help="Place background slice behind the data (default), in front, or through center")
    ap.add_argument("--slice-margin", type=float, default=2.0, help="Extra mm beyond bounds for slice placement")

    # Views
    ap.add_argument("--views", type=str, default="posX,negX,posY,negY,posZ,negZ",
                    help="Comma-separated subset of {posX,negX,posY,negY,posZ,negZ}")

    import vtk
    args = ap.parse_args()

    in_path = Path(args.input_path)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    win_pair = parse_pair(args.window, float, "window") if args.window else None
    robust_pair = parse_pair(args.robust, float, "robust") if (args.window is None and args.robust) else None
    bg_win = parse_pair(args.bg_window, float, "bg-window") if args.bg_window else None
    solid_rgb = parse_rgb(args.solid_color) if args.solid_color else None

    # Collect files
    if in_path.is_dir():
        it = in_path.rglob(args.pattern) if args.recursive else in_path.glob(args.pattern)
        files = [p for p in sorted(it) if p.is_file() and (p.suffix.lower() == ".nii" or p.suffix.lower() == ".gz" or p.name.lower().endswith(".nii.gz"))]
        if not files:
            sys.exit(f"No NIfTI files matching '{args.pattern}' under: {in_path}")
    elif in_path.is_file():
        files = [in_path]
    else:
        sys.exit(f"Input path not found: {in_path}")

    multi = len(files) > 1
    for idx, nifti_path in enumerate(files, 1):
        stem = nifti_path.stem if not nifti_path.name.endswith(".nii.gz") else nifti_path.name[:-7]
        # Pick output folder
        if multi and not args.flat:
            out_dir = out_root / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = args.prefix or stem
        else:
            out_dir = out_root
            if multi:
                prefix = (args.prefix + "_" + stem) if args.prefix else stem
            else:
                prefix = args.prefix or stem

        print(f"[{idx}/{len(files)}] Rendering {nifti_path} -> {out_dir} (prefix='{prefix}')")

        snap, all_views = render_nifti_to_png(
            nifti_path, out_dir, prefix,
            mode=args.mode, cmap=args.cmap, invert_cmap=args.invert_cmap, solid_rgb=solid_rgb,
            width=args.width, height=args.height, perspective=args.perspective,
            window=win_pair, robust=robust_pair, opacity=args.opacity, shade=args.shade,
            bg_nifti=args.bg_nifti, bg_mode=args.bg_mode, bg_window=bg_win, bg_opacity=args.bg_opacity,
            slice_thickness=args.slice_thickness, slice_opacity=args.slice_opacity,
            slice_placement=args.slice_placement, slice_margin=args.slice_margin,
            surface_opacity=args.surface_opacity, iso_frac=args.iso_frac, iso_level=args.iso_level,
            iso_gauss_mm=args.iso_gauss_mm, surface_smooth_iters=args.surface_smooth_iters
        )

        want = set([v.strip() for v in args.views.split(",") if v.strip()])
        for name,pos,up in all_views:
            if name in want:
                snap(name,pos,up)

if __name__ == "__main__":
    main()
