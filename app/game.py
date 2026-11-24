import pygame
import random
import sys
import math
import os
import wave
import struct

pygame.init()

# --------------------------------------------------
# GAME WINDOW
# --------------------------------------------------

WIDTH = 960
HEIGHT = 540

GAME_WIDTH = 320
GAME_HEIGHT = 180
SCALE = 3

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Humming Bird - Flower Adventure")

canvas = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))

clock = pygame.time.Clock()

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

# Pixel-style fonts
title_font = pygame.font.Font(None, 30)
medium_font = pygame.font.Font(None, 18)
small_font = pygame.font.Font(None, 12)
how_to_font = pygame.font.Font(None, 16)

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

bird_velocity = 0

GRAVITY = 0.09
FLAP_STRENGTH = -2.1

bird_width = 9
bird_height = 8

bird_animation = 0

# --------------------------------------------------
# GAME SETTINGS
# --------------------------------------------------

score = 0

game_speed = 0.65

ground_y = 166

camera_x = 0

landing = False
landing_timer = 0

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

    flowers.append(create_flower(100, 140))
    flowers.append(create_flower(190, 115))
    flowers.append(create_flower(280, 135))
    flowers.append(create_flower(370, 105))
    flowers.append(create_flower(460, 130))
    flowers.append(create_flower(550, 95))
    flowers.append(create_flower(640, 125))


# --------------------------------------------------
# BEES
# --------------------------------------------------

bees = []

def create_bees():

    global bees

    bees = []

    for i in range(4):

        bees.append({
            "x": 220 + i * 140,
            "y": random.randint(45, 110),
            "start_y": random.randint(45, 110),
            "direction": -1,
            "wing": 0
        })


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
# DRAW FLOWERS
# --------------------------------------------------

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
    wing_offset = 2 if bird_animation % 2 == 0 else -2

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

    global bird_y
    global bird_velocity
    global score
    global camera_x
    global landing
    global landing_timer
    global game_speed

    bird_y = 100
    bird_velocity = 0

    score = 0

    camera_x = 0

    landing = False
    landing_timer = 0

    game_speed = 0.65

    create_flowers()
    create_bees()



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

    mouse_pos = pygame.mouse.get_pos()

    # Convert mouse position to our small pixel screen
    mouse_pos = (
        mouse_pos[0] // SCALE,
        mouse_pos[1] // SCALE
    )

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
            "x": x,
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
        "1. Help your bird fly!",
        "2. Press SPACE to flap.",
        "3. Land on colourful flowers.",
        "4. Avoid the buzzing bees!",
        "5. Collect points by reaching flowers."
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

    mouse_pos = pygame.mouse.get_pos()

    mouse_pos = (
        mouse_pos[0] // SCALE,
        mouse_pos[1] // SCALE
    )

    draw_menu_button(
        surface,
        back_button,
        "BACK TO MENU",
        mouse_pos
    )

# --------------------------------------------------
# COLLISION DETECTION
# --------------------------------------------------

def check_bee_collision(bee):

    bird_rect = pygame.Rect(
        bird_x,
        bird_y,
        bird_width + 7,
        bird_height
    )

    bee_rect = pygame.Rect(
        bee["x"] - camera_x,
        bee["y"],
        16,
        8
    )

    return bird_rect.colliderect(bee_rect)


# --------------------------------------------------
# GAME UPDATE
# --------------------------------------------------

def update_game():

    global bird_y
    global bird_velocity
    global camera_x
    global score
    global landing
    global landing_timer
    global game_state
    global bird_animation
    global game_speed

    bird_animation += 1

    # --------------------------------------------------
    # LANDING ON A FLOWER
    # --------------------------------------------------

    if landing:

        landing_timer += 1

        bird_velocity = 0

        if landing_timer > 35:

            landing = False
            landing_timer = 0

            bird_velocity = FLAP_STRENGTH

        return

    # --------------------------------------------------
    # BIRD MOVEMENT
    # --------------------------------------------------

    bird_velocity += GRAVITY

    bird_y += bird_velocity

    # --------------------------------------------------
    # WORLD MOVEMENT
    # --------------------------------------------------

    camera_x += game_speed

    # --------------------------------------------------
    # FLOWER COLLISION
    # --------------------------------------------------

    bird_world_x = bird_x + camera_x

    for flower in flowers:

        flower_left = flower["x"] + 3
        flower_right = flower["x"] + 26

        bird_left = bird_world_x
        bird_right = bird_world_x + bird_width

        flower_top = flower["y"] - 1

        bird_bottom = bird_y + bird_height

        if (
            bird_right > flower_left
            and bird_left < flower_right
            and bird_velocity > 0
            and bird_bottom >= flower_top
            and bird_bottom <= flower_top + 5
        ):

            bird_y = flower_top - bird_height

            bird_velocity = 0

            landing = True
            landing_timer = 0

            score += 1

            break

    # --------------------------------------------------
    # BEE MOVEMENT
    # --------------------------------------------------

    for bee in bees:

        bee["x"] += bee["direction"] * 0.7

        bee["y"] = bee["start_y"] + math.sin(
            pygame.time.get_ticks() / 250 + bee["x"]
        ) * 7

        bee["wing"] += 1


        if check_bee_collision(bee):

            game_state = "game_over"

    # --------------------------------------------------
    # ADD NEW FLOWERS
    # --------------------------------------------------

    if flowers[-1]["x"] - camera_x < 300:

        last_x = flowers[-1]["x"]

        new_y = random.randint(85, 140)

        flowers.append(
            create_flower(last_x + random.randint(75, 100), new_y)
        )

    # --------------------------------------------------
    # ADD NEW BEES
    # --------------------------------------------------

    if bees[-1]["x"] - camera_x < 280:

        bees.append({
            "x": camera_x + 350,
            "y": random.randint(45, 110),
            "start_y": random.randint(45, 110),
            "direction": -1,
            "wing": 0
        })

    # Remove old objects
    flowers[:] = [
        flower for flower in flowers
        if flower["x"] - camera_x > -50
    ]

    bees[:] = [
        bee for bee in bees
        if bee["x"] - camera_x > -30
    ]

    # --------------------------------------------------
    # GAME OVER CONDITIONS
    # --------------------------------------------------

    if bird_y > ground_y - bird_height or bird_y < 0:

        game_state = "game_over"


# --------------------------------------------------
# DRAW GAME
# --------------------------------------------------

def draw_game(surface):

    draw_background(surface)

    # Flowers
    for flower in flowers:
        draw_flower(surface, flower)

    # Bees
    for bee in bees:
        game_bee = bee.copy()
        game_bee["x"] = bee["x"] - camera_x
        draw_bee(surface, game_bee)

    # Bird
    draw_bird(surface)

    # Ground
    draw_ground(surface)

    # Score
    draw_text(
        surface,
        "FLOWERS: " + str(score),
        medium_font,
        DARK,
        8,
        8
    )

    # Instructions
    draw_text(
        surface,
        "SPACE: FLAP",
        small_font,
        DARK,
        235,
        10
    )

    if landing:

        draw_text(
            surface,
            "RESTING...",
            small_font,
            DARK_GREEN,
            130,
            30
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
        "FLOWERS REACHED: " + str(score),
        medium_font,
        DARK_GREEN,
        90,
        70
    )

    mouse_pos = pygame.mouse.get_pos()

    mouse_pos = (
        mouse_pos[0] // SCALE,
        mouse_pos[1] // SCALE
    )

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

running = True

reset_game()

while running:

    clock.tick(60)

    # Play music on menu and how-to screens
    if game_state in ("menu", "how_to_play"):

        play_menu_music()

    else:

        stop_menu_music()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        # ------------------------------------------
        # MOUSE BUTTONS
        # ------------------------------------------

        if event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:

                mouse_pos = (
                    event.pos[0] // SCALE,
                    event.pos[1] // SCALE
                )

                if game_state == "menu":

                    if start_button.collidepoint(mouse_pos):

                        reset_game()
                        game_state = "game"

                    elif how_button.collidepoint(mouse_pos):

                        game_state = "how_to_play"

                    elif quit_button.collidepoint(mouse_pos):

                        running = False

                elif game_state == "how_to_play":

                    if back_button.collidepoint(mouse_pos):

                        game_state = "menu"

                elif game_state == "game_over":

                    if retry_button.collidepoint(mouse_pos):

                        reset_game()
                        game_state = "game"

                    elif game_over_menu_button.collidepoint(mouse_pos):

                        game_state = "menu"

        # ------------------------------------------
        # KEYBOARD CONTROLS
        # ------------------------------------------

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                if game_state == "game":

                    game_state = "menu"

                elif game_state == "how_to_play":

                    game_state = "menu"

                else:

                    running = False

            if event.key == pygame.K_RETURN:

                if game_state == "menu":

                    reset_game()
                    game_state = "game"

                elif game_state == "game_over":

                    reset_game()
                    game_state = "game"

                elif game_state == "how_to_play":

                    game_state = "menu"

            if event.key == pygame.K_SPACE:

                if game_state == "game" and not landing:

                    bird_velocity = FLAP_STRENGTH

    # ----------------------------------------------
    # UPDATE AND DRAW
    # ----------------------------------------------

    if game_state == "menu":

        draw_menu(canvas)

    elif game_state == "how_to_play":

        draw_how_to_play(canvas)

    elif game_state == "game":

        update_game()
        draw_game(canvas)

    elif game_state == "game_over":

        draw_game_over(canvas)

    # Enlarge pixel-art screen
    enlarged = pygame.transform.scale(
        canvas,
        (WIDTH, HEIGHT)
    )

    screen.blit(enlarged, (0, 0))

    pygame.display.flip()


stop_menu_music()

pygame.quit()
sys.exit()