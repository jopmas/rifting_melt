#!/usr/bin/env python3
'''
Build a Mandyoc interfaces.txt from an SVG drawing (e.g. made in Inkscape).

Each filled shape is a body of rock described by its object properties
(Inkscape: Object > Object Properties):

    Label        name of the unit (layer_label). Falls back to the id.
    Title        optional, used in the legend of the plot.
    Description  first line: rheology (wet olivine, dry olivine, wet quartz,
                 dry quartz, basalt, plagioclase, air); next lines: key = value
                 with the layer properties, e.g.

                     wet quartz
                     C = 1.0
                     rho = 2800.0
                     H = 0.19e-6/2800.0

                 Keys: C, rho, H (required), and A, n, Q, V (optional, they
                 override the values of the rheology). Long names such as
                 density or radiogenic_heat_production are also accepted.

Shapes with the same label are the same unit, so the description only needs
to be written once. A shape without description inherits the one of the group
(e.g. an Inkscape layer) that contains it. Shapes without any description are
ignored (annotations, arrows, ...). The fill colour is used only in the plot.

The SVG page is mapped to the model domain: top-left corner = (x=0, z=0),
bottom-right = (Lx, -Lz). x and z are scaled independently, so the drawing may
be vertically exaggerated. Shapes are painted in document order (later shapes
cover earlier ones), as Inkscape renders them.

Since Mandyoc layers are bounded by single-valued interfaces z(x), the script
reads, for every column x, the sequence of units from bottom to top and finds a
single ordered stack containing all of them (shortest common supersequence).
Units absent in a column get zero thickness there. A seed inside the lower
crust becomes [..., lower crust, seed, lower crust, ...]. Each slot of the stack
is a MandyocLayer whose interface is its base depth, as in generate_input_files*.py.

Usage
-----
    python svg2interfaces.py scenario.svg --Lx 1600 --Lz 300 --Nx 1601 -o interfaces.txt --plot setup.png

or, from another script:

    from svg2interfaces import layers_from_svg
    layers = layers_from_svg("scenario.svg", Lx=1600e3, Lz=300e3, Nx=1601)

Requires: numpy, matplotlib, svgelements (pip install svgelements)
'''

import argparse
import ast
import copy
import operator
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch
import svgelements as se

from mandyoc_layers import MandyocLayer, get_rheology

SVG_NS = "{http://www.w3.org/2000/svg}"
INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
SHAPE_TAGS = {"rect", "path", "circle", "ellipse", "polygon", "polyline"}

# key in the description -> attribute of MandyocLayer
PROPERTY_KEYS = {
    "C": "effective_viscosity_scale_factor",
    "rho": "density",
    "H": "radiogenic_heat_production",
    "A": "pre_exponential_constant",
    "n": "power_law_exponent",
    "Q": "activation_energy",
    "V": "activation_volume",
}
PROPERTY_KEYS.update({attr: attr for attr in list(PROPERTY_KEYS.values())})

#Build layer_properties according to the order of the stack_layers
rheological_symbols = ['C', 'rho', 'H', 'A', 'n', 'Q', 'V']
rheological_properties = ['effective_viscosity_scale_factor',
                          'density',
                          'radiogenic_heat_production',
                          'pre_exponential_constant',
                          'power_law_exponent',
                          'activation_energy',
                          'activation_volume']

###############################################################################
# Functions
###############################################################################

_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}

def eval_number(expr):
    '''Evaluate a numeric expression such as "1.67e-6 / 2700.0" (only numbers and + - * / **).'''
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError(f"invalid number: {expr!r}")
    return _eval(ast.parse(expr.strip(), mode="eval"))


def parse_description(text, where):
    '''Parse the description of a shape into (rheology class, {MandyocLayer attribute: value}).'''
    rheology, props = None, {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if "=" not in line:
            if rheology is not None:
                sys.exit(f"{where}: unexpected line '{line}' in the description.")
            rheology = line
            continue
        key, value = (s.strip() for s in line.split("=", 1))
        if key.lower() == "rheology":
            rheology = value
            continue
        if key not in PROPERTY_KEYS:
            sys.exit(f"{where}: unknown property '{key}'. Use {', '.join(rheological_symbols)}.")
        try:
            props[PROPERTY_KEYS[key]] = eval_number(value)
        except (ValueError, SyntaxError):
            sys.exit(f"{where}: could not read the value of {key} = {value}")

    if rheology is None:
        sys.exit(f"{where}: the first line of the description must be the rheology (e.g. wet quartz).")
    try:
        rheology = get_rheology(rheology)
    except ValueError as err:
        sys.exit(f"{where}: {err}")
    missing = [s for s in ["C", "rho", "H"] if PROPERTY_KEYS[s] not in props]
    if missing:
        sys.exit(f"{where}: missing {', '.join(missing)} in the description.")
    return rheology, props


def read_units(svg_file):
    '''
    Read label, title and description of the shapes.
    Return {shape id: unit label} and {unit label: dict(rheology, props, title)}.
    '''
    root = ET.parse(svg_file).getroot()
    parent = {child: el for el in root.iter() for child in el}

    def child_text(el, tag):
        node = el.find(SVG_NS + tag)
        return node.text.strip() if node is not None and node.text and node.text.strip() else None

    def label_of(el):
        return el.get(INKSCAPE_LABEL) or el.get("id")

    shape_unit, units = {}, {}
    pending = []  # shapes with a label but no description of their own

    for el in root.iter():
        if el.tag.replace(SVG_NS, "") not in SHAPE_TAGS:
            continue
        # nearest element (itself or a group above it) with a description
        node = el
        while node is not None and child_text(node, "desc") is None:
            node = parent.get(node)

        if node is None:
            if el.get(INKSCAPE_LABEL):
                pending.append(el)
            continue

        label = label_of(node)
        where = f"Shape '{label}' (id={node.get('id')})"
        rheology, props = parse_description(child_text(node, "desc"), where)
        unit = dict(rheology=rheology, props=props, title=child_text(node, "title"))
        if label in units and (units[label]["rheology"], units[label]["props"]) != (rheology, props):
            sys.exit(f"{where}: another shape with the same label has a different description.")
        units.setdefault(label, unit)
        shape_unit[el.get("id")] = label

    for el in pending:
        label = el.get(INKSCAPE_LABEL)
        if label in units:
            shape_unit[el.get("id")] = label
        else:
            print(f"Warning: shape '{label}' (id={el.get('id')}) has no description and was ignored.")

    if None in shape_unit:
        sys.exit("Some shapes with description have no id. Save the file with Inkscape to create them.")
    return shape_unit, units


def shape_to_mplpath(shape, n_curve=32):
    '''Convert an svgelements shape (already transformed) into a matplotlib Path.'''
    path = se.Path(shape)
    verts, codes = [], []
    for seg in path:
        if isinstance(seg, se.Move):
            verts.append((seg.end.x, seg.end.y))
            codes.append(MplPath.MOVETO)
        elif isinstance(seg, se.Close):
            verts.append((seg.end.x, seg.end.y))
            codes.append(MplPath.CLOSEPOLY)
        elif isinstance(seg, se.Line):
            verts.append((seg.end.x, seg.end.y))
            codes.append(MplPath.LINETO)
        else:  # curves and arcs: sample them
            for p in seg.npoint(np.linspace(0, 1, n_curve)[1:]):
                verts.append((p[0], p[1]))
                codes.append(MplPath.LINETO)
    if not verts:
        return None
    return MplPath(np.array(verts), codes)


def rasterize(svg_file, shape_unit, unit_names, Lx, Lz, Nx, dz, domain_id=None):
    '''
    Sample the SVG on a (Nzs, Nx) grid and return the unit index of each point (-1 = empty).
    Also return the fill colour of each unit (from its first shape).
    '''
    svg = se.SVG.parse(svg_file, reify=True)

    # SVG region mapped to the domain: page, or the bounding box of an element
    if domain_id:
        ref = svg.get_element_by_id(domain_id)
        if ref is None:
            sys.exit(f"Element with id '{domain_id}' not found in the SVG.")
        x0, y0, x1, y1 = ref.bbox(with_stroke=False)
    else:
        x0, y0, x1, y1 = 0.0, 0.0, float(svg.width), float(svg.height)

    Nzs = int(round(Lz / dz))
    dz = Lz / Nzs
    x = np.linspace(0, Lx, Nx)
    zdepth = (np.arange(Nzs) + 0.5) * dz  # depth of samples (positive downwards)

    # sample points in SVG coordinates
    XS, YS = np.meshgrid(x0 + x / Lx * (x1 - x0), y0 + zdepth / Lz * (y1 - y0))
    pts = np.column_stack([XS.ravel(), YS.ravel()])

    grid = -np.ones(XS.shape, dtype=int)
    colors = {}
    for el in svg.elements():
        if not isinstance(el, se.Shape) or el.id not in shape_unit:
            continue
        if el.values.get("display") == "none" or el.values.get("visibility") == "hidden":
            continue
        iunit = unit_names.index(shape_unit[el.id])
        if el.fill is not None and el.fill.value is not None:
            colors.setdefault(iunit, el.fill.hex[:7])
        mpath = shape_to_mplpath(el)
        if mpath is None:
            continue
        bx0, by0 = mpath.vertices.min(axis=0)
        bx1, by1 = mpath.vertices.max(axis=0)
        sel = (pts[:, 0] >= bx0) & (pts[:, 0] <= bx1) & (pts[:, 1] >= by0) & (pts[:, 1] <= by1)
        if not sel.any():
            continue
        inside = np.zeros(len(pts), dtype=bool)
        inside[sel] = mpath.contains_points(pts[sel])
        grid.ravel()[inside] = iunit

    return x, zdepth, dz, grid, colors


def scs(a, b):
    '''Shortest common supersequence of two sequences.'''
    m, n = len(a), len(b)
    L = np.zeros((m + 1, n + 1), dtype=int)
    L[m, :] = np.arange(n, -1, -1)
    L[:, n] = np.arange(m, -1, -1)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            L[i, j] = 1 + (L[i + 1, j + 1] if a[i] == b[j] else min(L[i + 1, j], L[i, j + 1]))
    out, i, j = [], 0, 0
    while i < m and j < n:
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif L[i + 1, j] <= L[i, j + 1]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out + list(a[i:]) + list(b[j:])


def embed(seq, stack):
    '''Slot index in stack of each element of seq (leftmost subsequence match).'''
    slots, j = [], 0
    for m in seq:
        while stack[j] != m:
            j += 1
        slots.append(j)
        j += 1
    return slots


def build_stack(grid, dz, Lz, unit_names):
    '''
    Return the stack of units (indices, bottom to top) and the base depth of each
    slot, an array (n_layers, Nx) in m (positive downwards).
    '''
    Nzs, Nx = grid.shape
    if (grid < 0).any():
        iz, ix = np.argwhere(grid < 0)[0]
        n = (grid < 0).sum()
        sys.exit(f"{n} sample points are not covered by any unit "
                 f"(first one at column {ix}, depth {(iz + 0.5) * dz / 1e3:.2f} km). "
                 "Check gaps between shapes or draw a background rectangle.")

    # runs of each column, from bottom to top: (units, depth of the top of each run)
    columns = []
    for ix in range(Nx):
        col = grid[::-1, ix]
        change = np.nonzero(np.diff(col))[0]  # last index of each run (except the top one)
        units = tuple(int(u) for u in col[np.r_[change, Nzs - 1]])
        tops = [Lz - (c + 1) * dz for c in change] + [0.0]
        columns.append((units, tops))

    unique = sorted({c[0] for c in columns}, key=len, reverse=True)
    stack = list(unique[0])
    for seq in unique[1:]:
        stack = scs(stack, seq)

    slots_of = {seq: embed(seq, stack) for seq in unique}
    base = np.empty((len(stack), Nx))
    base[0] = Lz
    for ix, (units, tops) in enumerate(columns):
        slots = slots_of[units]
        r = -1
        for k in range(1, len(stack)):
            # base of slot k = top of the highest run assigned to a slot below k
            while r + 1 < len(slots) and slots[r + 1] < k:
                r += 1
            base[k, ix] = tops[r] if r >= 0 else Lz
    return stack, unique, base


def layers_from_svg(svg_file, Lx, Lz, Nx, dz=100.0, domain_id=None, verbose=True):
    '''
    Read the SVG and return the list of MandyocLayer (bottom to top) with their
    interfaces (base depth, m) set. Lx, Lz and dz in m.
    '''
    shape_unit, units = read_units(svg_file)
    if not units:
        sys.exit("No shape with description found in the SVG.")
    unit_names = list(units)

    x, zdepth, dz, grid, colors = rasterize(svg_file, shape_unit, unit_names, Lx, Lz, Nx, dz, domain_id)
    stack, unique, base = build_stack(grid, dz, Lz, unit_names)

    # one MandyocLayer per unit, copied for each slot of the stack
    unit_layers = []
    for i, name in enumerate(unit_names):
        u = units[name]
        props = dict(u["props"])
        layer = MandyocLayer(name, u["rheology"],
                             density=props.pop("density"),
                             effective_viscosity_scale_factor=props.pop("effective_viscosity_scale_factor"),
                             radiogenic_heat_production=props.pop("radiogenic_heat_production"),
                             Nx=Nx)
        for attr, value in props.items():  # A, n, Q, V given in the description
            setattr(layer, attr, value)
        layer.title = u["title"] or name
        layer.color = colors.get(i, "#808080")
        unit_layers.append(layer)

    layers = []
    for k, iunit in enumerate(stack):
        layer = copy.copy(unit_layers[iunit])
        layer.interface = base[k].copy()
        layer.base_depth = base[k].max()
        layers.append(layer)

    # consistency check: rebuild the unit field from the interfaces (Mandyoc convention)
    rebuilt = np.array(stack)[(zdepth[:, None, None] < base[None, 1:, :].transpose(0, 2, 1)).sum(axis=2)]
    mismatch = (rebuilt != grid).sum()

    if verbose:
        print(f"Distinct column patterns: {len(unique)}")
        print("Layers (bottom to top):")
        for k, layer in enumerate(layers):
            print(f"  {k:2d}  {layer.layer_label:25s} {layer.rheology_name:12s} "
                  f"C={layer.effective_viscosity_scale_factor:g} rho={layer.density:g} H={layer.radiogenic_heat_production:.4g}")
        print(f"n_interfaces = {len(layers) - 1}   # put this in param.txt")
        print(f"Sampling: dx = {x[1] - x[0]:.1f} m, dz = {dz:.1f} m; mismatched samples after rebuild: {mismatch}")

    return layers


def write_interfaces(layers, fname="interfaces.txt"):
    '''Write interfaces.txt as in generate_input_files*.py.'''
    to_save = []
    for symbol, prop in zip(rheological_symbols, rheological_properties):
        to_save.append(f"{symbol} {' '.join([str(layer.__dict__[prop]) for layer in layers])}")

    with open(fname, "w") as f:
        for line in to_save:
            if len(line):
                f.write(f"{' '.join(line.split())}\n")

        # layer interfaces
        data = -1 * np.array(tuple(layer.interface for layer in layers[1::])).T
        np.savetxt(f, data, fmt="%.1f")


def plot_layers(layers, Lx, Lz, fname):
    x = np.linspace(0, Lx, layers[0].Nx)
    fig, ax = plt.subplots(figsize=(14, 6))
    for k, layer in enumerate(layers):
        top = layers[k + 1].interface if k < len(layers) - 1 else np.zeros_like(x)
        ax.fill_between(x / 1e3, -layer.interface / 1e3, -top / 1e3, color=layer.color, lw=0)
    ax.set_xlim(0, Lx / 1e3)
    ax.set_ylim(-Lz / 1e3, 0)
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("z [km]")

    # one legend entry per unit, from top to bottom
    unique = {}
    for layer in layers[::-1]:
        unique.setdefault(layer.layer_label, layer)
    handles = [Patch(facecolor=layer.color, edgecolor="k",
                     label=f"{layer.title}\n{layer.effective_viscosity_scale_factor:g} x {layer.rheology_name.replace('_', ' ')}\n"
                           fr"$\rho$ = {layer.density:g} kg/m³")
               for layer in unique.values()]
    fig.subplots_adjust(bottom=0.28)
    ax.legend(handles=handles, ncol=len(handles), loc="lower center", bbox_to_anchor=(0.5, -0.42),
              frameon=False, title="Layers properties", title_fontsize=12, fontsize=8)
    fig.savefig(fname, dpi=200)
    plt.close(fig)


###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("svg", help="SVG file with the scenario")
    parser.add_argument("--Lx", type=float, default=1600.0, help="domain width [km] (default 1600)")
    parser.add_argument("--Lz", type=float, default=300.0, help="domain depth [km] (default 300)")
    parser.add_argument("--Nx", type=int, default=1601, help="number of points in x (default 1601)")
    parser.add_argument("--dz", type=float, default=0.1, help="vertical sampling to locate the interfaces [km] (default 0.1)")
    parser.add_argument("--domain-id", help="id of an element whose bounding box is the domain (default: the page)")
    parser.add_argument("-o", "--output", default="interfaces.txt")
    parser.add_argument("--plot", help="save a figure of the resulting layers (e.g. numerical_setup.png)")
    args = parser.parse_args()

    Lx, Lz = args.Lx * 1.0e3, args.Lz * 1.0e3
    layers = layers_from_svg(args.svg, Lx, Lz, args.Nx, dz=args.dz * 1.0e3, domain_id=args.domain_id)
    write_interfaces(layers, args.output)
    print(f"Wrote {args.output}")

    if args.plot:
        plot_layers(layers, Lx, Lz, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
