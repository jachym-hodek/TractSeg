#!/usr/bin/env python3
import argparse, os, sys, glob, subprocess, shutil, tempfile
from pathlib import Path

def check_exec(name):
    if shutil.which(name) is None:
        sys.exit(f"ERROR: Required executable '{name}' not found on PATH.")

def run(cmd, **kwargs):
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)
    if res.returncode != 0:
        print(res.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return res

def parse_rgb(s: str):
    try:
        r,g,b = [int(x.strip()) for x in s.split(",")]
        for v in (r,g,b):
            if v < 0 or v > 255: raise ValueError
        return r,g,b
    except Exception:
        sys.exit("--solid-color must be 'R,G,B' with each in 0..255")

# ---- Add DEC colors to a polydata (per-point RGB) ----
def add_direction_colors(poly, signed: bool, gain: float):
    import vtk
    npts = poly.GetNumberOfPoints()
    lines = poly.GetLines()
    lines.InitTraversal()
    idlist = vtk.vtkIdList()

    # accumulate local tangents at each point
    acc = [[0.0,0.0,0.0] for _ in range(npts)]
    cnt = [0]*npts
    while lines.GetNextCell(idlist):
        m = idlist.GetNumberOfIds()
        if m < 2: continue
        for i in range(m-1):
            a = idlist.GetId(i)
            b = idlist.GetId(i+1)
            pa = poly.GetPoint(a)
            pb = poly.GetPoint(b)
            v = (pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2])
            for k in range(3):
                acc[a][k] += v[k]
                acc[b][k] += v[k]
            cnt[a] += 1
            cnt[b] += 1

    import math, vtk
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("RGB")
    for i in range(npts):
        vx,vy,vz = acc[i]
        mag = math.sqrt(vx*vx+vy*vy+vz*vz)
        if mag > 0: nx,ny,nz = vx/mag, vy/mag, vz/mag
        else: nx,ny,nz = 0.0,0.0,1.0
        if signed:
            r = int((nx*0.5+0.5)*255)
            g = int((ny*0.5+0.5)*255)
            b = int((nz*0.5+0.5)*255)
        else:
            r = int(abs(nx)*255); g = int(abs(ny)*255); b = int(abs(nz)*255)
        # apply global gain and clamp
        r = max(0, min(255, int(r * gain)))
        g = max(0, min(255, int(g * gain)))
        b = max(0, min(255, int(b * gain)))
        colors.InsertNextTuple3(r,g,b)
    poly.GetPointData().SetScalars(colors)

# ---------- VTK renderer (six directions, DEC coloring, optional BG) ----------
def render_with_vtk(vtk_path: Path, out_dir: Path, prefix: str, width: int, height: int,
                    tube_radius: float|None, bg_nifti: str|None, bg_opacity: float,
                    bg_window: tuple|None, coloring: str, solid_rgb: tuple[int,int,int]|None,
                    tract_style: str, bg_mode: str, slice_thickness: float, slice_opacity: float,
                    halo_scale: float, halo_opacity: float, slice_offset: float, line_width: float, dec_gain: float,
                    slice_placement: str, slice_margin: float):
    try:
        import vtk
    except Exception:
        sys.exit("ERROR: VTK is required for --renderer vtk. Install with:  pip install vtk")

    # Streamlines
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_path)); reader.Update()
    poly = reader.GetOutput()

    # Colorize points
    if coloring == "solid":
        # set a constant color via actor property later; no scalars
        pass
    else:
        add_direction_colors(poly, signed=(coloring == "signed"), gain=dec_gain)

    # Optional tubes (keeps point-data; vtkTubeFilter passes scalars)
    data_for_render = poly
    if tube_radius and tube_radius > 0:
        tf = vtk.vtkTubeFilter()
        tf.SetInputData(poly)
        tf.SetRadius(tube_radius)
        tf.SetNumberOfSides(12)
        tf.CappingOn()
        tf.Update()
        data_for_render = tf.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(data_for_render)
    if coloring == "solid":
        mapper.ScalarVisibilityOff()
    else:
        mapper.ScalarVisibilityOn()
        mapper.SetColorModeToDirectScalars()
        mapper.SetScalarModeToUsePointData()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if coloring == "solid":
        rgb = solid_rgb if solid_rgb else (230,230,230)
        actor.GetProperty().SetColor(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
    # line width (when using polylines)
    if (not tube_radius) or tube_radius <= 0:
        actor.GetProperty().SetLineWidth(max(1.0, line_width))

    # Tract lighting style (default: vivid unlit colors)
    if tract_style == "unlit":
        prop = actor.GetProperty()
        prop.LightingOff()
        # Alternative:
        # prop.SetAmbient(1.0); prop.SetDiffuse(0.0); prop.SetSpecular(0.0)

    ren = vtk.vtkRenderer()
    ren.SetBackground(0,0,0)

    # Optional halo (black, slightly thicker tube underneath)
    if halo_opacity > 0 and tube_radius and tube_radius > 0:
        halo = vtk.vtkTubeFilter()
        halo.SetInputData(poly)
        halo.SetRadius(max(1e-6, tube_radius * max(halo_scale, 1.0)))
        halo.SetNumberOfSides(12)
        halo.CappingOn()
        halo.Update()
        hmap = vtk.vtkPolyDataMapper()
        hmap.SetInputData(halo.GetOutput())
        hmap.ScalarVisibilityOff()
        hactor = vtk.vtkActor()
        hactor.SetMapper(hmap)
        hactor.GetProperty().SetColor(0,0,0)
        hactor.GetProperty().SetOpacity(halo_opacity)
        ren.AddActor(hactor)

    ren.AddActor(actor)

    # Optional background (volume or camera-facing slice)
    if bg_nifti:
        nii_reader = vtk.vtkNIFTIImageReader()
        nii_reader.SetFileName(bg_nifti); nii_reader.Update()
        im = nii_reader.GetOutput()
        data_min, data_max = im.GetScalarRange()
        if bg_window is not None: wmin,wmax = bg_window
        else: wmin,wmax = data_min, data_max
        if wmax <= wmin: wmax = wmin + 1.0

        sform = nii_reader.GetSFormMatrix(); qform = nii_reader.GetQFormMatrix()
        mat = sform if sform is not None else qform

        if bg_mode == "slices":
            # We'll drive a plane per-view (in snap()) so the slice never intersects the bundle.
            smap = vtk.vtkImageResliceMapper()
            smap.SetInputConnection(nii_reader.GetOutputPort())
            smap.SetSlabThickness(max(0.1, slice_thickness))
            smap.SetSlabTypeToMean()  # or SetSlabTypeToMax() for MIP look
            plane = vtk.vtkPlane()
            smap.SetSlicePlane(plane)
            smap.SliceFacesCameraOff()
            smap.SliceAtFocalPointOff()
            islice = vtk.vtkImageSlice()
            islice.SetMapper(smap)
            isp = islice.GetProperty()
            isp.SetOpacity(max(0.0, min(1.0, slice_opacity)))
            isp.SetColorWindow(wmax - wmin)
            isp.SetColorLevel(0.5*(wmin + wmax))
            if mat is not None: islice.SetUserMatrix(mat)
            ren.AddViewProp(islice)
        else:
            ctf = vtk.vtkColorTransferFunction()
            ctf.AddRGBPoint(wmin, 0,0,0); ctf.AddRGBPoint(wmax, 1,1,1)
            otf = vtk.vtkPiecewiseFunction()
            mid = 0.5*(wmin+wmax)
            otf.AddPoint(wmin, 0.0); otf.AddPoint(mid, bg_opacity*0.6); otf.AddPoint(wmax, bg_opacity)
            vol_property = vtk.vtkVolumeProperty()
            vol_property.SetColor(ctf); vol_property.SetScalarOpacity(otf)
            vol_property.SetInterpolationTypeToLinear(); vol_property.ShadeOff()
            vol_mapper = vtk.vtkSmartVolumeMapper()
            vol_mapper.SetInputConnection(nii_reader.GetOutputPort())
            volume = vtk.vtkVolume(); volume.SetMapper(vol_mapper); volume.SetProperty(vol_property)
            if mat is not None: volume.SetUserMatrix(mat)
            ren.AddVolume(volume)

    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1); win.AddRenderer(ren); win.SetSize(width, height)
    cam = ren.GetActiveCamera(); cam.ParallelProjectionOn()

    # Fit to data bounds (streamlines)
    b = data_for_render.GetBounds()
    cx,cy,cz = (0.5*(b[0]+b[1]), 0.5*(b[2]+b[3]), 0.5*(b[4]+b[5]))
    dx,dy,dz = (b[1]-b[0], b[3]-b[2], b[5]-b[4])
    max_dim = max(dx,dy,dz); pad = 0.6*max_dim + 1e-3

    def snap(name, pos, view_up):
        # base camera
        cam.SetFocalPoint(cx,cy,cz); cam.SetPosition(*pos); cam.SetViewUp(*view_up)
        # Place camera-facing slice according to placement strategy
        if bg_nifti and bg_mode == "slices":
            fx,fy,fz = cam.GetFocalPoint(); px,py,pz = cam.GetPosition()
            vx,vy,vz = (fx-px, fy-py, fz-pz)
            import math
            vm = math.sqrt(vx*vx+vy*vy+vz*vz) or 1.0
            ux,uy,uz = (vx/vm, vy/vm, vz/vm)  # camera view direction
            # 8 corners of the tract bounds
            corners = [(b[i0], b[i1], b[i2]) for i0 in (0,1) for i1 in (2,3) for i2 in (4,5)]
            # projections along view dir from camera position
            projs = [ (cx1-px)*ux + (cy1-py)*uy + (cz1-pz)*uz for (cx1,cy1,cz1) in corners ]
            near_t, far_t = min(projs), max(projs)
            if slice_placement == "behind":
                t = far_t + max(0.0, slice_margin)
            elif slice_placement == "front":
                t = near_t - max(0.0, slice_margin)
            else:  # center (original behavior)
                t = 0.5*(near_t + far_t) + slice_offset
            # allow extra per-user offset even in behind/front
            t += slice_offset if slice_placement != "center" else 0.0
            ox,oy,oz = (px + ux*t, py + uy*t, pz + uz*t)
            plane.SetNormal(ux,uy,uz)
            plane.SetOrigin(ox,oy,oz)
        cam.SetParallelScale(max_dim*0.6); ren.ResetCameraClippingRange(); win.Render()
        w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.Update()
        writer = vtk.vtkPNGWriter(); out_path = out_dir / f"{prefix}_{name}.png"
        writer.SetFileName(str(out_path)); writer.SetInputConnection(w2i.GetOutputPort()); writer.Write()
        print(f"Saved {out_path}")

    snap("posX", (b[1]+pad, cy, cz), (0,0,1))
    snap("negX", (b[0]-pad, cy, cz), (0,0,1))
    snap("posY", (cx, b[3]+pad, cz), (0,0,1))
    snap("negY", (cx, b[2]-pad, cz), (0,0,1))
    snap("posZ", (cx, cy, b[5]+pad), (0,1,0))
    snap("negZ", (cx, cy, b[4]-pad), (0,1,0))

# ---------- mrview fallback (3 views) ----------
def render_with_mrview(ref_image: str, tck_path: Path, out_dir: Path, prefix: str,
                       width: int, height: int, show_background: bool, solid_rgb: tuple[int,int,int]|None):
    planes = [("sagittal", "0"), ("coronal", "1"), ("axial", "2")]
    for name, plane_idx in planes:
        cmd = [
            "mrview", ref_image,
            "-size", f"{width},{height}",
            "-noannotations", "1",
            "-lock", "1",
            "-tractography.load", str(tck_path),
            "-tractography.geometry", "pseudotubes",
            "-tractography.lighting", "1",
            "-tractography.slab", "-1",
            "-plane", plane_idx,
            "-capture.folder", str(out_dir),
            "-capture.prefix", f"{prefix}_{name}",
            "-capture.grab",
            "-exit"
        ]
        if not show_background:
            cmd.insert(6, "-imagevisible"); cmd.insert(7, "0")
        if solid_rgb is not None:
            cmd.extend(["-tractography.colour", f"{solid_rgb[0]},{solid_rgb[1]},{solid_rgb[2]}"])
        run(cmd)

def parse_bg_window(s: str|None):
    if not s: return None
    try:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) != 2: raise ValueError
        return (parts[0], parts[1])
    except Exception:
        sys.exit("--bg-window must be min,max")

def main():
    ap = argparse.ArgumentParser(description="Render .tck to PNGs with optional DEC coloring and background NIfTI. VTK: six views; mrview: three views.")
    ap.add_argument("input_dir", help="Folder to search for .tck files (recurses).")
    ap.add_argument("--out-dir", default="tck_pngs", help="Output root (default: ./tck_pngs)")
    ap.add_argument("--renderer", choices=["vtk","mrview"], default="vtk")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--tube-radius", type=float, default=0.6, help="VTK only. 0 = lines.")
    # Background
    ap.add_argument("--bg-nifti", help="Anatomical .nii(.gz) as background (optional).")
    ap.add_argument("--bg-opacity", type=float, default=0.15, help="VTK only: 0..1 (default 0.15).")
    ap.add_argument("--bg-window", type=str, default=None, help="VTK only: intensity window 'min,max'.")
    ap.add_argument("--bg-mode", choices=["volume","slices"], default="volume",
                    help="VTK only: 'volume' (default) or 'slices' (camera-facing slab).")
    ap.add_argument("--slice-thickness", type=float, default=6.0,
                    help="VTK only (slices mode): slab thickness in mm.")
    ap.add_argument("--slice-opacity", type=float, default=0.6,
                    help="VTK only (slices mode): slab opacity 0..1.")
    ap.add_argument("--slice-placement", choices=["behind","front","center"], default="behind",
                    help="VTK slices: place slice behind the bundle (default), in front, or through the center.")
    ap.add_argument("--slice-margin", type=float, default=2.0,
                    help="VTK slices: extra mm beyond the tract bounds for 'behind'/'front'.")
    ap.add_argument("--slice-offset", type=float, default=0.0,
                    help="VTK only (slices mode): move slice behind tracts by this many mm (positive = deeper).")
    # Coloring
    ap.add_argument("--coloring", choices=["axis","signed","solid"], default="axis",
                    help="VTK: 'axis' (|dx|,|dy|,|dz|), 'signed' (encodes sign), or 'solid'. mrview defaults to directional coloring unless --solid-color is given.")
    ap.add_argument("--solid-color", type=str, default=None, help="When --coloring solid (VTK) or to override mrview: 'R,G,B' in 0..255.")
    # Tract style & halo (VTK only)
    ap.add_argument("--tract-style", choices=["unlit","lit"], default="unlit",
                    help="VTK only: 'unlit' (vivid colors, default) or 'lit' (shaded).")
    ap.add_argument("--halo-scale", type=float, default=1.5,
                    help="VTK only: halo radius = tube_radius * halo-scale (<=1 disables).")
    ap.add_argument("--halo-opacity", type=float, default=0.0,
                    help="VTK only: 0 disables halo (default 0).")
    ap.add_argument("--line-width", type=float, default=1.0,
                    help="VTK only: line width in pixels when --tube-radius 0.")
    ap.add_argument("--dec-gain", type=float, default=1.0,
                    help="VTK only (DEC): multiply DEC brightness (0..1, e.g., 0.8).")
    # mrview compatibility when no visible background
    ap.add_argument("--ref-image", help="mrview only: image to satisfy mrview if no --bg-nifti (hidden).")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    tck_files = sorted(glob.glob(str(in_dir / "**" / "*.tck"), recursive=True))
    if not tck_files: sys.exit(f"No .tck files found under: {in_dir}")

    # MRtrix tool(s)
    check_exec("tckconvert")
    solid_rgb = parse_rgb(args.solid_color) if args.solid_color else None
    bg_window = parse_bg_window(args.bg_window)

    if args.renderer == "mrview":
        check_exec("mrview")
        if not args.bg_nifti and not args.ref_image:
            sys.exit("--renderer mrview requires either --bg-nifti (visible) or --ref-image (hidden).")
        if args.bg_mode == "slices":
            sys.exit("--bg-mode slices is VTK-only. Use --renderer vtk.")

    for tck in tck_files:
        tck_path = Path(tck)
        stem = tck_path.stem
        case_out = out_root / stem
        case_out.mkdir(parents=True, exist_ok=True)

        if args.renderer == "vtk":
            with tempfile.TemporaryDirectory() as td:
                vtk_path = Path(td) / f"{stem}.vtk"
                run(["tckconvert", str(tck_path), str(vtk_path)])
                render_with_vtk(
                    vtk_path, case_out, stem, args.width, args.height,
                    args.tube_radius, args.bg_nifti, args.bg_opacity, bg_window,
                    args.coloring, solid_rgb,
                    args.tract_style, args.bg_mode, args.slice_thickness, args.slice_opacity,
                    args.halo_scale, args.halo_opacity, args.slice_offset, args.line_width, args.dec_gain,
                    args.slice_placement, args.slice_margin
                )
        else:
            ref_image = args.bg_nifti if args.bg_nifti else args.ref_image
            show_bg = bool(args.bg_nifti)
            render_with_mrview(ref_image, tck_path, case_out, stem,
                               args.width, args.height, show_bg, solid_rgb)

    print(f"\nDone. Images saved under: {out_root}")

if __name__ == "__main__":
    main()
