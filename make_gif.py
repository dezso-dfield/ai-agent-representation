import math, numpy as np, imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")  # off-screen rendering, safe on macOS
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------- Canvas ----------
W, H = 16, 9
plt.rcParams.update({
    "figure.figsize": (9.6, 5.4),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
fig, ax = plt.subplots()
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

# ---------- Style ----------
CLR_TEXT   = "#111111"
CLR_EDGE   = "#CCCCCC"
CLR_ACTIVE = "#2F6FEB"

# ---------- Nodes ----------
nodes = {
    "User":      (2.2,  6.8, "User"),
    "Agent":     (7.0,  6.8, "AI Agent"),
    "Model":     (12.2, 6.8, "LLM / Model"),
    "Tools":     (7.0,  4.1, "Tools / APIs"),
    "Knowledge": (12.2, 4.1, "Files / Knowledge"),
    "Output":    (12.2, 2.0, "Result"),
}

# ---------- Helpers ----------
def canvas_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)
    rgb = np.roll(buf, -1, axis=2)
    return rgb[..., :3].copy()

def measure_text(ax, text, fontsize=16):
    tmp = ax.text(0, 0, text, fontsize=fontsize, visible=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = tmp.get_window_extent(renderer=renderer)
    tmp.remove()
    axbb = ax.get_window_extent(renderer=renderer)
    xdat = (bb.width  / axbb.width ) * (ax.get_xlim()[1] - ax.get_xlim()[0])
    ydat = (bb.height / axbb.height) * (ax.get_ylim()[1] - ax.get_ylim()[0])
    return xdat, ydat

def draw_node(ax, x, y, label, pad_x=1.2, pad_y=0.6, min_width=None):
    tw, th = measure_text(ax, label, fontsize=16)
    bw = max(2.8, tw + pad_x)
    bh = max(1.0, th + pad_y)
    if min_width:
        bw = max(bw, min_width)
    box = FancyBboxPatch(
        (x - bw/2, y - bh/2), bw, bh,
        boxstyle="round,pad=0.02,rounding_size=0.22",
        linewidth=1.8, edgecolor=CLR_EDGE, facecolor="white"
    )
    ax.add_patch(box)
    txt = ax.text(x, y, label, ha="center", va="center", fontsize=16, color=CLR_TEXT)
    return {"x": x, "y": y, "w": bw, "h": bh, "patch": box, "txt": txt}

# Draw nodes with custom widths
node = {}
for k, (x, y, label) in nodes.items():
    if k == "Knowledge":
        node[k] = draw_node(ax, x, y, label, min_width=5.5)
    elif k in ["Model", "Tools"]:
        node[k] = draw_node(ax, x, y, label, min_width=4.5)
    else:
        node[k] = draw_node(ax, x, y, label)

def edge_points(src, dst, inset=0.4):
    sx, sy, sw, sh = node[src]["x"], node[src]["y"], node[src]["w"], node[src]["h"]
    tx, ty, tw, th = node[dst]["x"], node[dst]["y"], node[dst]["w"], node[dst]["h"]
    dx, dy = tx - sx, ty - sy
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    start = (sx + ux * (sw/2 + inset), sy + uy * (sh/2 + inset))
    end   = (tx - ux * (tw/2 + inset), ty - uy * (th/2 + inset))
    return start, end

# ---------- Edges ----------
edges = [
    ("User","Agent"),
    ("Agent","Model"), ("Model","Agent"),
    ("Agent","Tools"), ("Agent","Knowledge"),
    ("Agent","Output"),
]
arrows = {}
for s, t in edges:
    (x1, y1), (x2, y2) = edge_points(s, t)
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle="-|>", mutation_scale=14,
                         linewidth=1.4, color=CLR_EDGE, alpha=0.5)  # lighter arrows
    ax.add_patch(ar)
    arrows[(s, t)] = ar

# ---------- Steps ----------
steps = [
    ("User","Agent",      "Step 1: User sends a request"),
    ("Agent","Model",     "Step 2: Agent queries the model"),
    ("Model","Agent",     "Step 3: Model returns guidance"),
    ("Agent","Tools",     "Step 4: Agent calls tools / APIs"),
    ("Agent","Knowledge", "Step 5: Agent reads knowledge files"),
    ("Agent","Output",    "Step 6: Agent returns the result"),
]
caption = ax.text(W/2, H-0.8, steps[0][2], fontsize=18, ha="center", va="center", color="#000000")
subtitle = ax.text(W/2, 0.7, "Flow: User → Agent → Model/Tools/Knowledge → Agent → Output",
                   fontsize=12, ha="center", va="center", color="#555555")

# ---------- Animation ----------
FPS = 10
FRAMES_PER_STEP = 22
frames = []
for f in range(FRAMES_PER_STEP * len(steps)):
    step_idx = f // FRAMES_PER_STEP
    src, dst, label = steps[step_idx]
    caption.set_text(label)

    # reset
    for nb in node.values():
        nb["patch"].set_edgecolor(CLR_EDGE)
        nb["patch"].set_linewidth(1.8)
    for ar in arrows.values():
        ar.set_color(CLR_EDGE)
        ar.set_alpha(0.5)

    # highlight active
    node[src]["patch"].set_edgecolor(CLR_ACTIVE)
    node[src]["patch"].set_linewidth(3.0)
    node[dst]["patch"].set_edgecolor(CLR_ACTIVE)
    node[dst]["patch"].set_linewidth(3.0)
    arrows[(src, dst)].set_color(CLR_ACTIVE)
    arrows[(src, dst)].set_alpha(0.9)

    frames.append(canvas_to_rgb(fig))

plt.close(fig)
imageio.mimsave("ai_agent_process_clean.gif", frames, fps=FPS)
print("Saved ai_agent_process_clean.gif")
