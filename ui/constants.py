"""
ui/constants.py
===============
Every colour, dimension, and numeric constant used by the renderer.
Nothing here imports from other ui modules — safe as a leaf dependency.
"""

# ── Map colours ───────────────────────────────────────────────────────────────
COLOR_GRASS           = (184, 212, 160)      # muted green background
COLOR_GRASS_DARK      = (164, 196, 140)      # shadow under roads
COLOR_ROAD            = (224, 224, 218)      # light-gray road surface
COLOR_ROAD_EDGE       = (200, 200, 194)      # solid road edge
COLOR_SIDEWALK        = (215, 205, 185)      # beige sidewalk strip
COLOR_LANE_WHITE      = (255, 255, 255)      # lane markings
COLOR_INTERSECTION    = (216, 216, 210)      # slightly different shade

# ── Sign colours ──────────────────────────────────────────────────────────────
COLOR_STOP_RED        = (220, 50, 50)
COLOR_STOP_WHITE      = (255, 255, 255)
COLOR_YIELD_RED       = (220, 50, 50)
COLOR_YIELD_WHITE     = (255, 255, 255)
COLOR_PRIORITY_YELLOW = (255, 210, 0)
COLOR_PRIORITY_WHITE  = (255, 255, 255)
COLOR_SIGN_POLE       = (110, 110, 110)

# ── Traffic light (semaphore) colours ─────────────────────────────────────────
COLOR_LIGHT_RED       = (220, 40, 40)
COLOR_LIGHT_YELLOW    = (240, 200, 40)
COLOR_LIGHT_GREEN     = (40, 200, 60)
COLOR_LIGHT_OFF       = (60, 60, 60)
COLOR_LIGHT_HOUSING   = (35, 35, 40)

# ── Decoration colours ────────────────────────────────────────────────────────
COLOR_TREE_TRUNK      = (139, 90, 60)
COLOR_TREE_CANOPY_A   = (62, 145, 62)
COLOR_TREE_CANOPY_B   = (85, 165, 75)
COLOR_HOUSE_WALL      = (225, 215, 200)
COLOR_HOUSE_ROOF_A    = (165, 82, 72)
COLOR_HOUSE_ROOF_B    = (100, 120, 165)
COLOR_HOUSE_DOOR      = (105, 75, 55)
COLOR_HOUSE_WINDOW    = (170, 210, 240)
COLOR_PERSON_SKIN     = (210, 175, 140)
COLOR_PERSON_SHIRT_A  = (70, 130, 200)
COLOR_PERSON_SHIRT_B  = (200, 85, 85)

# ── HUD / panel colours ──────────────────────────────────────────────────────
COLOR_HUD_BG          = (28, 30, 42, 210)
COLOR_HUD_BORDER      = (58, 62, 80)
COLOR_HUD_TEXT        = (230, 230, 235)
COLOR_HUD_DIM         = (145, 145, 165)
COLOR_HUD_ACCENT      = (86, 168, 255)
COLOR_WARNING         = (255, 70, 70)
COLOR_BTN_BG          = (50, 55, 72)
COLOR_BTN_HOVER       = (72, 78, 100)
COLOR_BTN_TEXT        = (230, 230, 235)

# ── Launch screen colours ─────────────────────────────────────────────────────
COLOR_LAUNCH_BG       = (22, 26, 38)
COLOR_LAUNCH_GRID     = (30, 34, 48)
COLOR_LAUNCH_ROAD     = (48, 52, 68)
COLOR_LAUNCH_DASH     = (78, 82, 100)
COLOR_LAUNCH_TITLE    = (235, 235, 245)
COLOR_LAUNCH_SUB      = (140, 150, 180)
COLOR_LAUNCH_BTN      = (86, 168, 255)
COLOR_LAUNCH_BTN_H    = (120, 190, 255)

# ── Camera defaults ───────────────────────────────────────────────────────────
DEFAULT_ZOOM  = 4.0               # zoomed in to hide road ends
MIN_ZOOM      = 1.5               # allow zoom out for large networks
MAX_ZOOM      = 10.0
ZOOM_STEP     = 0.3

# ── Road dimensions (world metres) ───────────────────────────────────────────
LANE_WIDTH_M       = 7.0          # half the road — matches LANE_OFFSET in world.py
ROAD_HALF_W        = 10.0         # total half-width including shoulder
SIDEWALK_W         = 3.0
ROAD_LENGTH        = 180.0        # how far roads extend from centre
DASH_LEN           = 3.0          # centre-line dash length
DASH_GAP           = 3.0

# ── Vehicle dimensions (world metres) ────────────────────────────────────────
CAR_LENGTH         = 4.5
CAR_WIDTH          = 2.2
HEADLIGHT_R        = 0.35
HEADLIGHT_INSET    = 0.55

# ── Awareness zone ────────────────────────────────────────────────────────────
AWARENESS_DIVISOR  = 5.0          # awareness_m = speed_kmh / this

# ── HUD dimensions (pixels) ──────────────────────────────────────────────────
HUD_PANEL_W        = 275
HUD_PAD            = 12
HUD_ROW_H          = 30
CONTROL_BAR_H      = 48
BTN_W              = 90
BTN_H              = 34
LEGEND_W           = 210
TOP_BAR_H          = 32

# ── Font sizes ────────────────────────────────────────────────────────────────
FONT_SM   = 13
FONT_MD   = 16
FONT_LG   = 22
FONT_XL   = 42
FONT_TTL  = 56
