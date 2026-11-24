import argparse
import math
import os
import random
import struct
import time
import wave

import pygame

from app.audio_io import PitchEngine
from app.calibration import Calibration, default_range

# --------------------------------------------------
# GAME WINDOW
# --------------------------------------------------

WIDTH = 1440
HEIGHT = 800
MIN_WINDOW_WIDTH = 960
MIN_WINDOW_HEIGHT = 600

GAME_WIDTH = 320
GAME_HEIGHT = 180

screen = None
canvas = None
clock = None
game_rect = pygame.Rect(0, 0, GAME_WIDTH, GAME_HEIGHT)
dashboard_rect = pygame.Rect(0, 0, 0, 0)

# --------------------------------------------------
# MENU MUSIC
# --------------------------------------------------

MUSIC_ENABLED = True

def create_menu_music():

    os.makedirs("data", exist_ok=True)

    music_path = os.path.join("data", "menu_theme.wav")

    if os.path.exists(music_path):
        return music_path

    sample_rate = 22050

    # A cheerful little 8-bit melody
    notes = [
        523, 659, 784, 659,
        587, 698, 880, 698,
        659, 784, 988, 784,
        659, 587, 523, 587
    ]

    samples = []

    for frequency in notes:

        duration = 0.18
        total_samples = int(sample_rate * duration)

        for i in range(total_samples):

            t = i / sample_rate

            envelope = min(1, i / 200) * min(
                1, (total_samples - i) / 500
            )

            value = math.sin(2 * math.pi * frequency * t)

            sample = int(9000 * value * envelope)

            samples.append(sample)

    with wave.open(music_path, "wb") as audio:

        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)

        audio.writeframes(
            struct.pack("<" + "h" * len(samples), *samples)
        )

    return music_path


def play_menu_music():

    if not MUSIC_ENABLED:
        return

    if not pygame.mixer.get_init():
        return

    if pygame.mixer.music.get_busy():
        return

    music_path = create_menu_music()

    pygame.mixer.music.load(music_path)
    pygame.mixer.music.set_volume(0.20)
    pygame.mixer.music.play(-1)


def stop_menu_music():

    if pygame.mixer.get_init():
        pygame.mixer.music.stop()

# Initialized by main() so importing game logic does not open a window.
title_font = medium_font = small_font = how_to_font = None
ui_font = ui_title_font = None


def layout_for_window(size):
    """Return non-overlapping game and dashboard rectangles for a window."""
    width, height = size
    split = int(width * 0.43)
    game_width = min(split - 32, int((height - 200) * GAME_WIDTH / GAME_HEIGHT))
    game_width = max(1, game_width)
    game_height = max(1, round(game_width * GAME_HEIGHT / GAME_WIDTH))
    game = pygame.Rect((split - game_width) // 2, 52, game_width, game_height)
    return game, pygame.Rect(split, 0, width - split, height)


def screen_to_game(pos):
    """Translate a window mouse position into pixel-art game coordinates."""
    if not game_rect.collidepoint(pos):
        return None
    return (int((pos[0] - game_rect.left) * GAME_WIDTH / game_rect.width),
            int((pos[1] - game_rect.top) * GAME_HEIGHT / game_rect.height))


def initialize_display():
    global screen, canvas, clock, game_rect, dashboard_rect
    global title_font, medium_font, small_font, how_to_font, ui_font, ui_title_font
    pygame.init()
    info = pygame.display.Info()
    width = min(WIDTH, max(MIN_WINDOW_WIDTH, info.current_w - 40))
    height = min(HEIGHT, max(MIN_WINDOW_HEIGHT, info.current_h - 100))
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    game_rect, dashboard_rect = layout_for_window(screen.get_size())
    pygame.display.set_caption("Humming Bird - Live Pitch Lab")
    canvas = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
    clock = pygame.time.Clock()
    title_font = pygame.font.Font(None, 30)
    medium_font = pygame.font.Font(None, 18)
    small_font = pygame.font.Font(None, 12)
    how_to_font = pygame.font.Font(None, 16)
    ui_font = pygame.font.Font(None, 19)
    ui_title_font = pygame.font.Font(None, 25)

# --------------------------------------------------
# COLOURS
# --------------------------------------------------

SKY = (150, 220, 245)
WHITE = (255, 255, 255)
DARK = (40, 65, 75)

GREEN = (75, 160, 85)
DARK_GREEN = (40, 115, 65)

YELLOW = (255, 220, 75)
PINK = (255, 125, 180)
PURPLE = (175, 125, 220)
ORANGE = (255, 165, 75)

BEE_YELLOW = (255, 215, 65)
BEE_BLACK = (45, 45, 45)

BIRD_BLUE = (65, 155, 220)
BIRD_DARK = (35, 100, 165)

# --------------------------------------------------
# GAME STATES
# --------------------------------------------------

game_state = "menu"

# --------------------------------------------------
# BIRD SETTINGS
# --------------------------------------------------

bird_x = 85
bird_y = 100

# Positions are in game pixels, velocities in pixels per second.
bird_velocity = 0.0
GRAVITY = 324.0
FLAP_STRENGTH = -126.0
MAX_RISE_SPEED = 125.0
MAX_FALL_SPEED = 145.0
PITCH_STIFFNESS = 100.0
PITCH_DAMPING = 20.0
PITCH_LIFT = 850.0
PITCH_DROP = 700.0
PITCH_GRACE_SECONDS = 0.25
MIN_REST_SECONDS = 0.35
TAKEOFF_DISTANCE = 12.0
MAX_STEP_SECONDS = 1.0 / 60.0
MAX_FRAME_SECONDS = 0.1

bird_width = 15
bird_height = 9

bird_animation = 0.0
pitch_target = None
pitch_silence_seconds = 0.0

# --------------------------------------------------
# GAME SETTINGS
# --------------------------------------------------

score = 0
gaps_passed = 0

game_speed = 45.0  # world pixels per second

ground_y = 166

camera_x = 0.0

landing = False
landing_timer = 0.0
walls = []
WALL_WIDTH = 28
WALL_GAP = 72
WALL_SPACING = 190

# --------------------------------------------------
# FLOWERS
# --------------------------------------------------

flowers = []

flower_colours = [PINK, PURPLE, YELLOW, ORANGE]

def create_flower(x, y):

    return {
        "x": x,
        "y": y,
        "colour": random.choice(flower_colours),
        "width": 28
    }


def create_flowers():

    global flowers

    flowers = []

    flowers.append(create_flower(170, 135))
    flowers.append(create_flower(300, 120))
    flowers.append(create_flower(495, 130))


# Walls are separated by flowers so a resting bird is always safe.
def create_wall(x):
    return {"x": x, "gap_top": random.randint(38, 76), "passed": False}


def create_walls():
    global walls
    walls = [create_wall(400)]


# --------------------------------------------------
# DRAWING FUNCTIONS
# --------------------------------------------------

def draw_text(surface, text, font, colour, x, y):

    image = font.render(text, False, colour)

    surface.blit(image, (x, y))


def draw_cloud(surface, x, y):

    pygame.draw.rect(surface, WHITE, (x + 5, y, 17, 6))
    pygame.draw.rect(surface, WHITE, (x, y + 4, 27, 6))
    pygame.draw.rect(surface, WHITE, (x + 7, y + 8, 16, 4))


def draw_background(surface):

    surface.fill(SKY)

    # Sun
    pygame.draw.rect(surface, YELLOW, (270, 15, 18, 18))
    pygame.draw.rect(surface, ORANGE, (275, 10, 8, 4))

    # Clouds
    draw_cloud(surface, 35, 30)
    draw_cloud(surface, 175, 48)
    draw_cloud(surface, 260, 75)

    # Distant hills
    pygame.draw.rect(surface, (130, 205, 145), (0, 145, 320, 35))


def draw_ground(surface):

    pygame.draw.rect(
        surface,
        DARK_GREEN,
        (0, ground_y, GAME_WIDTH, 14)
    )

    pygame.draw.rect(
        surface,
        GREEN,
        (0, ground_y, GAME_WIDTH, 4)
    )


# --------------------------------------------------
# DRAW WALLS AND FLOWERS
# --------------------------------------------------

def draw_wall(surface, wall):
    x = int(wall["x"] - camera_x)
    gap_top = wall["gap_top"]
    gap_bottom = gap_top + WALL_GAP
    for top, height in ((0, gap_top), (gap_bottom, ground_y - gap_bottom)):
        pygame.draw.rect(surface, DARK_GREEN, (x, top, WALL_WIDTH, height))
        pygame.draw.rect(surface, GREEN, (x + 3, top, WALL_WIDTH - 6, height))
    pygame.draw.rect(surface, DARK_GREEN, (x - 2, gap_top - 5, WALL_WIDTH + 4, 5))
    pygame.draw.rect(surface, DARK_GREEN, (x - 2, gap_bottom, WALL_WIDTH + 4, 5))


def draw_flower(surface, flower):

    x = int(flower["x"] - camera_x)
    y = int(flower["y"])

    # Stem
    pygame.draw.rect(
        surface,
        DARK_GREEN,
        (x + 13, y + 5, 3, 18)
    )

    # Leaves
    pygame.draw.rect(
        surface,
        GREEN,
        (x + 5, y + 13, 9, 4)
    )

    pygame.draw.rect(
        surface,
        GREEN,
        (x + 16, y + 17, 9, 4)
    )

    # Flower petals
    colour = flower["colour"]

    pygame.draw.rect(surface, colour, (x + 7, y - 5, 7, 7))
    pygame.draw.rect(surface, colour, (x + 17, y - 5, 7, 7))
    pygame.draw.rect(surface, colour, (x + 3, y, 7, 7))
    pygame.draw.rect(surface, colour, (x + 21, y, 7, 7))
    pygame.draw.rect(surface, colour, (x + 8, y + 6, 7, 6))
    pygame.draw.rect(surface, colour, (x + 16, y + 6, 7, 6))

    # Flower centre
    pygame.draw.rect(
        surface,
        YELLOW,
        (x + 11, y, 9, 9)
    )


# --------------------------------------------------
# DRAW BIRD
# --------------------------------------------------

def draw_bird(surface, x=None, y=None):

    # Use the given position, or the gameplay position
    if x is None:
        x = bird_x

    if y is None:
        y = bird_y

    x = int(x)
    y = int(y)

    # Body
    pygame.draw.rect(
        surface,
        BIRD_BLUE,
        (x, y, 13, 9)
    )

    # Head
    pygame.draw.rect(
        surface,
        BIRD_BLUE,
        (x + 7, y - 4, 8, 8)
    )

    # Wing animation
    wing_offset = 2 if int(bird_animation * 10) % 2 == 0 else -2

    pygame.draw.rect(
        surface,
        BIRD_DARK,
        (x + 3, y + 2 + wing_offset, 7, 4)
    )

    # Eye
    pygame.draw.rect(
        surface,
        DARK,
        (x + 12, y - 1, 2, 2)
    )

    # Beak
    pygame.draw.rect(
        surface,
        ORANGE,
        (x + 15, y + 2, 4, 3)
    )

    # Tail
    pygame.draw.rect(
        surface,
        BIRD_DARK,
        (x - 4, y + 3, 5, 3)
    )

# --------------------------------------------------
# DRAW CUTE BEE
# --------------------------------------------------

def draw_bee(surface, bee):

    x = int(bee["x"])
    y = int(bee["y"])

    wing_offset = 2 if bee["wing"] % 2 == 0 else -2

    # Wings
    pygame.draw.rect(
        surface,
        WHITE,
        (x + 3, y - 5 + wing_offset, 5, 4)
    )

    pygame.draw.rect(
        surface,
        WHITE,
        (x + 10, y - 5 - wing_offset, 5, 4)
    )

    # Body
    pygame.draw.rect(
        surface,
        BEE_YELLOW,
        (x, y, 16, 8)
    )

    # Black stripes
    pygame.draw.rect(
        surface,
        BEE_BLACK,
        (x + 5, y, 3, 8)
    )

    pygame.draw.rect(
        surface,
        BEE_BLACK,
        (x + 11, y, 2, 8)
    )

    # Eyes - face the direction the bee is moving
    if bee.get("direction", 1) == -1:
        eye_x = x + 1
    else:
        eye_x = x + 13

    # Eyes
    pygame.draw.rect(
        surface,
        DARK,
        (eye_x, y + 2, 2, 2)
    )


# --------------------------------------------------
# RESET GAME
# --------------------------------------------------

def reset_game():
    global bird_y, bird_velocity, bird_animation, score, gaps_passed, camera_x
    global landing, landing_timer, game_speed, pitch_target, pitch_silence_seconds

    bird_y = 100.0
    bird_velocity = 0.0
    bird_animation = 0.0
    score = gaps_passed = 0
    camera_x = 0.0
    landing = False
    landing_timer = 0.0
    pitch_target = None
    pitch_silence_seconds = 0.0
    game_speed = 45.0
    create_flowers()
    create_walls()



# --------------------------------------------------
# MENU BUTTONS
# --------------------------------------------------

start_button = pygame.Rect(105, 85, 110, 18)
how_button = pygame.Rect(105, 108, 110, 18)
quit_button = pygame.Rect(105, 131, 110, 18)


def draw_menu_button(surface, rect, text, mouse_pos):

    hovered = rect.collidepoint(mouse_pos)

    if hovered:
        button_colour = (255, 205, 115)
    else:
        button_colour = (255, 230, 155)

    # Button shadow
    pygame.draw.rect(
        surface,
        (110, 100, 90),
        (rect.x + 2, rect.y + 2, rect.width, rect.height)
    )

    # Button body
    pygame.draw.rect(
        surface,
        button_colour,
        rect
    )

    # Border
    pygame.draw.rect(
        surface,
        DARK,
        rect,
        1
    )

    # Button text
    text_image = small_font.render(text, False, DARK)

    text_rect = text_image.get_rect(
        center=rect.center
    )

    surface.blit(text_image, text_rect)


# --------------------------------------------------
# MAIN MENU
# --------------------------------------------------

def draw_menu(surface):
    

    draw_background(surface)

    mouse_pos = screen_to_game(pygame.mouse.get_pos()) or (-1, -1)

    # Title
    draw_text(
        surface,
        "HUMMING BIRD",
        title_font,
        DARK,
        95,
        17
    )

    draw_text(
        surface,
        "FLOWER ADVENTURE",
        small_font,
        DARK_GREEN,
        119,
        42
    )

    # Animation timer
    tick = pygame.time.get_ticks()

    # Honey bee on the LEFT
    left_bee = {
        "x": 45,
        "y": 65,
        "wing": tick // 100
    }
    draw_bee(surface, left_bee)

    # Hummingbird FIXED in the CENTER
    draw_bird(surface, 150, 65)

    # Honey bee on the RIGHT
    right_bee = {
        "x": 255,
        "y": 65,
        "wing": tick // 100
    }
    draw_bee(surface, right_bee)

    
    # Flowers along the bottom
    for i, x in enumerate([20, 65, 260, 295]):

        flower = {
            "x": x + camera_x,  # Menu decorations stay fixed after returning from a game.
            "y": 153,
            "colour": flower_colours[i % len(flower_colours)]
        }

        draw_flower(surface, flower)

    # Buttons
    draw_menu_button(
        surface,
        start_button,
        "START GAME",
        mouse_pos
    )

    draw_menu_button(
        surface,
        how_button,
        "HOW TO PLAY",
        mouse_pos
    )

    draw_menu_button(
        surface,
        quit_button,
        "QUIT",
        mouse_pos
    )

# --------------------------------------------------
# HOW TO PLAY SCREEN
# --------------------------------------------------

back_button = pygame.Rect(110, 143, 100, 18)


def draw_how_to_play(surface):

    draw_background(surface)

    draw_text(
        surface,
        "HOW TO PLAY",
        title_font,
        DARK,
        105,
        20
    )

    instructions = [
        "1. Hum higher to rise; lower to descend.",
        "2. Fly through gaps; silence makes you fall.",
        "3. Rest on flowers; hum higher to take off.",
        "4. SPACE: manual flap; P: pitch control.",
        "5. C: calibrate; ESC: back to menu."
    ]

    for i, instruction in enumerate(instructions):

        draw_text(
            surface,
            instruction,
            how_to_font,
            DARK,
            55,
            58 + i * 14
        )

    mouse_pos = screen_to_game(pygame.mouse.get_pos()) or (-1, -1)

    draw_menu_button(
        surface,
        back_button,
        "BACK TO MENU",
        mouse_pos
    )

# --------------------------------------------------
# COLLISION DETECTION
# --------------------------------------------------

def check_wall_collision(wall):
    left = wall["x"] - camera_x - 2
    right = wall["x"] - camera_x + WALL_WIDTH + 2
    bird_left = bird_x + 1
    bird_right = bird_x + bird_width - 1
    return (bird_right > left and bird_left < right
            and (bird_y - 3 < wall["gap_top"]
                 or bird_y + bird_height > wall["gap_top"] + WALL_GAP))


# --------------------------------------------------
# GAME UPDATE
# --------------------------------------------------

def flap():
    """Take off or flap using the manual fallback."""
    global bird_velocity, landing, landing_timer
    landing = False
    landing_timer = 0.0
    bird_velocity = FLAP_STRENGTH


def update_game(state=None, pitch_range=None, pitch_control=False, dt=1.0 / 60.0):
    """Advance flight using elapsed seconds and bounded physics steps."""
    remaining = min(max(0.0, dt), MAX_FRAME_SECONDS)
    while remaining > 1e-9 and game_state == "game":
        step = min(remaining, MAX_STEP_SECONDS)
        _update_game_step(state or {}, pitch_range, pitch_control, step)
        remaining -= step


def _update_game_step(state, pitch_range, pitch_control, dt):
    global bird_y, bird_velocity, bird_animation, camera_x, score, gaps_passed
    global landing, landing_timer, game_state, pitch_target, pitch_silence_seconds

    bird_animation += dt
    semitone = state.get("semitone")
    voiced = (pitch_control and pitch_range is not None and semitone is not None
              and math.isfinite(semitone) and state.get("gate_open", True)
              and state.get("f0", 1) is not None and not state.get("calibrating"))
    if voiced:
        pitch_target = pitch_range.height(semitone, 14, ground_y - bird_height - 12)
        pitch_silence_seconds = 0.0
    else:
        pitch_silence_seconds += dt

    if landing:
        landing_timer += dt
        bird_velocity = 0.0
        if (voiced and landing_timer >= MIN_REST_SECONDS
                and pitch_target <= bird_y - TAKEOFF_DISTANCE):
            landing = False
            landing_timer = 0.0
        else:
            return  # The bird, world, and hazards remain stationary and safe.

    previous_bottom = bird_y + bird_height
    previous_camera = camera_x
    acceleration = GRAVITY
    if pitch_control and pitch_target is not None and pitch_silence_seconds <= PITCH_GRACE_SECONDS:
        # A damped spring follows pitch without teleporting; gravity remains active.
        lift = (PITCH_STIFFNESS * (pitch_target - bird_y)
                - PITCH_DAMPING * bird_velocity - GRAVITY)
        acceleration += max(-PITCH_LIFT, min(PITCH_DROP, lift))
    bird_velocity = max(-MAX_RISE_SPEED, min(MAX_FALL_SPEED, bird_velocity + acceleration * dt))
    bird_y += bird_velocity * dt
    camera_x += game_speed * dt

    # A descending bird lands at the visible petal tops. Sweep the crossing
    # horizontally as well as vertically so a frame-rate change cannot miss it.
    bird_bottom = bird_y + bird_height
    if bird_velocity > 0 and bird_bottom > previous_bottom:
        for flower in flowers:
            flower_top = flower["y"] - 5
            if previous_bottom <= flower_top <= bird_bottom:
                fraction = (flower_top - previous_bottom) / (bird_bottom - previous_bottom)
                world_x = bird_x + previous_camera + (camera_x - previous_camera) * fraction
                if world_x + bird_width > flower["x"] + 3 and world_x < flower["x"] + 28:
                    bird_y = flower_top - bird_height
                    bird_velocity = 0.0
                    landing = True
                    landing_timer = 0.0
                    if not flower.get("visited"):
                        flower["visited"] = True
                        score += 1
                    return

    for wall in walls:
        if check_wall_collision(wall):
            game_state = "game_over"
            return
        if not wall["passed"] and bird_x + camera_x > wall["x"] + WALL_WIDTH + 2:
            wall["passed"] = True
            gaps_passed += 1

    if bird_y - 4 < 0 or bird_y + bird_height >= ground_y:
        game_state = "game_over"
        return

    # Keep a wall and a flower between successive walls available ahead.
    if walls:
        while walls[-1]["x"] - camera_x < GAME_WIDTH + 220:
            next_x = walls[-1]["x"] + WALL_SPACING
            walls.append(create_wall(next_x))
            flowers.append(create_flower(next_x + 95, random.randint(105, 140)))
    walls[:] = [wall for wall in walls if wall["x"] - camera_x > -WALL_WIDTH - 5]
    flowers[:] = [flower for flower in flowers if flower["x"] - camera_x > -30]


# --------------------------------------------------
# DRAW GAME
# --------------------------------------------------

def draw_game(surface, pitch_control=True, state=None):

    draw_background(surface)

    for wall in walls:
        draw_wall(surface, wall)

    for flower in flowers:
        draw_flower(surface, flower)

    # Bird
    draw_bird(surface)

    # Ground
    draw_ground(surface)

    # Score
    draw_text(
        surface,
        f"GAPS: {gaps_passed}  FLOWERS: {score}",
        medium_font,
        DARK,
        8,
        8
    )

    # Instructions
    draw_text(
        surface,
        "PITCH" if pitch_control else "MANUAL: SPACE",
        small_font,
        DARK,
        235,
        10
    )

    source = (state or {}).get("source", "starting")
    draw_text(surface, f"SOURCE: {source}", small_font, DARK, 8, 25)

    if landing:

        draw_text(
            surface,
            "RESTING: HUM HIGHER OR SPACE",
            small_font,
            DARK_GREEN,
            95,
            40
        )


# --------------------------------------------------
# GAME OVER
# --------------------------------------------------

retry_button = pygame.Rect(105, 105, 110, 18)
game_over_menu_button = pygame.Rect(105, 130, 110, 18)


def draw_game_over(surface):

    draw_background(surface)

    draw_text(
        surface,
        "GAME OVER!",
        title_font,
        DARK,
        112,
        35
    )

    draw_text(
        surface,
        f"GAPS: {gaps_passed}   FLOWERS: {score}",
        medium_font,
        DARK_GREEN,
        90,
        70
    )

    mouse_pos = screen_to_game(pygame.mouse.get_pos()) or (-1, -1)

    draw_menu_button(
        surface,
        retry_button,
        "PLAY AGAIN",
        mouse_pos
    )

    draw_menu_button(
        surface,
        game_over_menu_button,
        "BACK TO MENU",
        mouse_pos
    )

# --------------------------------------------------
# MAIN GAME LOOP
# --------------------------------------------------

def start_audio(source="mic", device=None):
    """Start the requested source, falling back to a demo if the mic fails."""
    engine = PitchEngine(source=source, device=device)
    try:
        engine.start()
        return engine, None
    except Exception as exc:
        engine.stop()
        if source != "mic":
            raise
        fallback = PitchEngine(source="synthetic")
        fallback.start()
        return fallback, str(exc)


def draw_calibration(surface, calibration, state):
    draw_background(surface)
    draw_text(surface, "CALIBRATION", title_font, DARK, 90, 16)
    lines = []
    line = ""
    for word in calibration.message.split():
        candidate = f"{line} {word}".strip()
        if small_font.size(candidate)[0] > 275 and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    for index, line in enumerate(lines):
        draw_text(surface, line, small_font, DARK, 22, 58 + index * 13)
    note = state.get("note") or "--"
    draw_text(surface, f"Current note: {note}   Gate: "
              f"{'OPEN' if state.get('gate_open') else 'CLOSED'}",
              small_font, DARK_GREEN, 22, 100)
    draw_text(surface, "SPACE: skip and use manual control", small_font, DARK, 22, 135)
    draw_text(surface, "ESC: return to menu", small_font, DARK, 22, 150)


def draw_game_sidebar(surface):
    """Frame the game with the most important controls."""
    surface.blit(ui_title_font.render("HUMMING BIRD  /  FLIGHT", True, (225, 235, 239)),
                 (game_rect.left, 16))
    pygame.draw.rect(surface, (77, 218, 225), game_rect.inflate(4, 4), 2)
    y = game_rect.bottom + 20
    lines = (
        "Hum higher to rise; lower to descend.",
        "Land on a flower to rest; hum higher to leave.",
        "SPACE: manual flap    P: pitch control",
        "C: recalibrate    ENTER: start / retry    ESC: menu",
    )
    for line in lines:
        surface.blit(ui_font.render(line, True, (190, 210, 220)), (game_rect.left, y))
        y += 23
    y += 8
    surface.blit(ui_font.render("HOW TO READ THE PLOTS", True, (77, 218, 225)),
                 (game_rect.left, y))
    y += 22
    for line in (
        "Waveform: compare the input and filtered signal.",
        "Spectrum: peaks are frequencies; white marks f0.",
        "Spectrogram: brighter bands mean stronger tones.",
        "Pitch track: orange smooths the cyan estimate.",
    ):
        surface.blit(ui_font.render(line, True, (190, 210, 220)), (game_rect.left, y))
        y += 21


def main(argv=None):
    global game_state, screen, game_rect, dashboard_rect
    from app.dashboard import Dashboard, handle_engine_key

    parser = argparse.ArgumentParser(description="Pitch-controlled Humming Bird")
    parser.add_argument("--source", default="mic", help="mic, synthetic, or a WAV file")
    parser.add_argument("--device", type=int, default=None, help="microphone device index")
    args = parser.parse_args(argv)
    initialize_display()
    engine = None
    try:
        engine, fallback_reason = start_audio(args.source, args.device)
        dashboard = Dashboard()
        calibration = None
        pitch_range = default_range(engine.mode)
        calibrated = engine.actual_source != "mic"
        pitch_control = True
        dashboard_surface = pygame.Surface(dashboard_rect.size)
        dashboard_elapsed = 1.0  # Draw immediately, then refresh at 15 Hz.
        after_calibration = "game"
        last_frame = None
        last_progress = time.perf_counter()
        running = True
        game_state = "menu"
        reset_game()

        def begin_calibration(destination):
            global game_state
            nonlocal calibration, after_calibration
            stop_menu_music()
            after_calibration = destination
            calibration = Calibration(engine)
            calibration.start()
            game_state = "calibration"

        def begin_game():
            global game_state
            reset_game()
            if engine.actual_source == "mic" and not calibrated:
                begin_calibration("game")
            else:
                game_state = "game"

        while running:
            elapsed = clock.tick(60) / 1000.0
            dashboard_elapsed += elapsed
            if game_state in ("menu", "how_to_play"):
                play_menu_music()
            else:
                stop_menu_music()

            state = engine.get_state()
            frame = state.get("frame_id")
            if frame is not None and frame != last_frame:
                last_frame, last_progress = frame, time.perf_counter()
            if engine.actual_source == "mic" and time.perf_counter() - last_progress > 3.0:
                fallback_reason = engine.last_error or "Microphone stopped producing audio."
                old = engine
                engine = PitchEngine(source="synthetic", method=old.method,
                                     window=old.window, mode=old.mode, N=old.N)
                engine.set_bypass(old.bypass)
                engine.set_noise_snr(old.noise_snr_db)
                old.stop()
                engine.start()
                state = engine.get_state()
                pitch_range = default_range(engine.mode)
                calibrated = True
                if game_state == "calibration":
                    calibration = None
                    game_state = after_calibration
                last_frame, last_progress = None, time.perf_counter()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if event.type == pygame.VIDEORESIZE:
                    size = (max(MIN_WINDOW_WIDTH, event.w), max(MIN_WINDOW_HEIGHT, event.h))
                    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
                    game_rect, dashboard_rect = layout_for_window(screen.get_size())
                    dashboard_surface = pygame.Surface(dashboard_rect.size)
                    dashboard_elapsed = 1.0
                    continue
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = screen_to_game(event.pos)
                    if pos is None:
                        continue
                    if game_state == "menu":
                        if start_button.collidepoint(pos):
                            begin_game()
                        elif how_button.collidepoint(pos):
                            game_state = "how_to_play"
                        elif quit_button.collidepoint(pos):
                            running = False
                    elif game_state == "how_to_play" and back_button.collidepoint(pos):
                        game_state = "menu"
                    elif game_state == "game_over":
                        if retry_button.collidepoint(pos):
                            begin_game()
                        elif game_over_menu_button.collidepoint(pos):
                            game_state = "menu"
                if event.type != pygame.KEYDOWN:
                    continue
                key = event.key
                if key == pygame.K_ESCAPE:
                    if game_state in ("game", "how_to_play", "calibration", "game_over"):
                        calibration = None
                        game_state = "menu"
                    else:
                        running = False
                    continue
                if game_state == "calibration":
                    if key == pygame.K_RETURN:
                        calibration.record_note(state)
                    elif key == pygame.K_SPACE:
                        calibration = None
                        pitch_range = default_range(engine.mode)
                        calibrated = False
                        pitch_control = False
                        game_state = after_calibration
                    continue
                if key == pygame.K_c and engine.actual_source == "mic":
                    begin_calibration("game" if game_state == "game" else "menu")
                elif key == pygame.K_p:
                    pitch_control = True
                elif key == pygame.K_SPACE and game_state == "game":
                    pitch_control = False
                    flap()
                elif key == pygame.K_RETURN:
                    if game_state in ("menu", "game_over"):
                        begin_game()
                    elif game_state == "how_to_play":
                        game_state = "menu"
                else:
                    changed = handle_engine_key(engine, key)
                    if changed == "mode":
                        pitch_range = default_range(engine.mode)
                        calibrated = engine.actual_source != "mic"
                        if not calibrated and game_state == "game":
                            begin_calibration("game")

            if game_state == "calibration" and calibration is not None:
                if calibration.update(state) == "done":
                    pitch_range = calibration.pitch_range
                    calibrated = True
                    pitch_control = True
                    calibration = None
                    game_state = after_calibration

            if game_state == "menu":
                draw_menu(canvas)
                if fallback_reason:
                    draw_text(canvas, "MIC UNAVAILABLE - SYNTHETIC DEMO", small_font,
                              DARK_GREEN, 65, 168)
            elif game_state == "how_to_play":
                draw_how_to_play(canvas)
            elif game_state == "calibration":
                draw_calibration(canvas, calibration, state)
            elif game_state == "game":
                update_game(state, pitch_range, pitch_control, dt=elapsed)
                draw_game(canvas, pitch_control, state)
            elif game_state == "game_over":
                draw_game_over(canvas)

            # Rendering DSP plots less often keeps audio and flight responsive.
            if dashboard_elapsed >= 1.0 / 15.0:
                dashboard.draw(dashboard_surface, engine, state, pitch_range, pitch_control,
                               fallback_reason, game_state)
                dashboard_elapsed %= 1.0 / 15.0
            screen.fill((14, 22, 34))
            draw_game_sidebar(screen)
            screen.blit(pygame.transform.scale(canvas, game_rect.size), game_rect)
            screen.blit(dashboard_surface, dashboard_rect)
            pygame.display.flip()
    finally:
        if engine is not None:
            engine.stop()
        stop_menu_music()
        pygame.quit()


if __name__ == "__main__":
    main()





