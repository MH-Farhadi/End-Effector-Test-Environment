import pygame
import random
import math
import time
import numpy as np

pygame.init()
pygame.joystick.init()  # Initialize the joystick module



NOISE_MAGNITUDE = 0.2  # Initial noise magnitude
MIN_NOISE = 0.0
MAX_NOISE = 2.0
NOISE_STEP = 0.1


# Original window size for scaling reference
OLD_WINDOW_SIZE = (600, 600)

# New window size
WINDOW_SIZE = (1200, 800)

# Scaling factors
SCALING_FACTOR_X = WINDOW_SIZE[0] / OLD_WINDOW_SIZE[0]
SCALING_FACTOR_Y = WINDOW_SIZE[1] / OLD_WINDOW_SIZE[1]
SCALING_FACTOR = (SCALING_FACTOR_X + SCALING_FACTOR_Y) / 2  # Average scaling factor

# Constants (adjusted with scaling factor)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

FONT_COLOR = (0, 0, 0)
FONT_SIZE = int(24 * SCALING_FACTOR)
ARROW_LENGTH = int(60 * SCALING_FACTOR)

NUM_GOALS = 3
OBSTACLE_RADIUS = int(20 * SCALING_FACTOR)
COLLISION_BUFFER = int(5 * SCALING_FACTOR)
ENABLE_OBSTACLES = True
MAX_SPEED = 3 * SCALING_FACTOR

DOT_RADIUS = int(30 * SCALING_FACTOR)
TARGET_RADIUS = int(10 * SCALING_FACTOR)
GOAL_DETECTION_RADIUS = DOT_RADIUS + TARGET_RADIUS

# ---------- Ghost Path Settings ---------
GHOST_TRAIL_DURATION = 3.0  # seconds of trail to show
recent_positions = []       # will store (x, y, timestamp)
# ----------------------------------------

# ---------- Additional Config -----------
RECENT_DIR_LOOKBACK = 1.0   # how many seconds of ghost path to use for direction
GOAL_SWITCH_THRESHOLD = 0.05 # how much better the new target must be to cause a switch
# ----------------------------------------

# Set up display
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("2D Environment: Incorporate Recent Movement + Ghost Path")

# Load font
font = pygame.font.Font(None, FONT_SIZE)

START_POS = [WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2]
dot_pos = START_POS.copy()
gamma = 0.2
reached_goal = False

targets = []
for _ in range(NUM_GOALS):
    targets.append([random.randint(0, WINDOW_SIZE[0]),
                    random.randint(0, WINDOW_SIZE[1])])
current_target_idx = 0

obstacles = []

# Joystick init
joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Joystick initialized:", joystick.get_name())
else:
    print("No joystick detected.")


# ---------- Helper Functions -----------
def distance(pos1, pos2):
    return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

def line_circle_intersection(start, end, circle_center, radius):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    cx = circle_center[0] - start[0]
    cy = circle_center[1] - start[1]
    l2 = dx*dx + dy*dy
    if l2 == 0:
        return distance(start, circle_center) <= radius
    t = max(0, min(1, (cx*dx + cy*dy) / l2))
    projection_x = start[0] + t * dx
    projection_y = start[1] + t * dy
    return distance([projection_x, projection_y], circle_center) <= radius

def check_collision(pos, new_pos):
    if not ENABLE_OBSTACLES:
        return False
    for obstacle_pos in obstacles:
        if line_circle_intersection(pos, new_pos, obstacle_pos, OBSTACLE_RADIUS + COLLISION_BUFFER):
            return True
    return False

def get_recent_direction():
    """
    Compute the average velocity direction from the ghost path
    for the last RECENT_DIR_LOOKBACK seconds.
    Returns a normalized 2D vector or [0,0] if insufficient data.
    """
    if len(recent_positions) < 2:
        return [0, 0]

    current_time = time.time()

    # Collect points within the last RECENT_DIR_LOOKBACK seconds
    valid_points = []
    for (x, y, t) in reversed(recent_positions):
        if (current_time - t) <= RECENT_DIR_LOOKBACK:
            valid_points.append((x, y, t))
        else:
            break

    if len(valid_points) < 2:
        return [0, 0]

    # Sort by time ascending (just in case)
    valid_points.sort(key=lambda p: p[2])
    # We'll approximate average velocity by connecting first and last
    x1, y1, t1 = valid_points[0]
    x2, y2, t2 = valid_points[-1]

    dt = t2 - t1
    if dt < 0.001:
        return [0, 0]  # Avoid division by zero

    vx = (x2 - x1) / dt
    vy = (y2 - y1) / dt
    mag = math.hypot(vx, vy)
    if mag > 0:
        return [vx / mag, vy / mag]
    else:
        return [0, 0]


def predict_human_target(human_input):
    """
    Combines:
      1) alignment with the immediate human input
      2) alignment with the *recent direction* from ghost path
      3) proximity to each target
    Then picks the best target, unless improvement is below GOAL_SWITCH_THRESHOLD.
    """
    global current_target_idx

    # If close enough to the current target, stick to it
    dist_to_current = distance(dot_pos, targets[current_target_idx])
    close_threshold = GOAL_DETECTION_RADIUS * 2
    if dist_to_current < close_threshold:
        return current_target_idx

    if human_input[0] == 0 and human_input[1] == 0:
        return current_target_idx

    # Get immediate movement direction
    h_mag = math.hypot(human_input[0], human_input[1])
    h_dir = [human_input[0] / h_mag, human_input[1] / h_mag] if h_mag > 0 else [0, 0]

    # Get recent direction from ghost path
    recent_dir = get_recent_direction()

    best_score = float('-inf')
    best_target_idx = current_target_idx

    for i, target in enumerate(targets):
        # Vector from dot to target
        to_target_x = target[0] - dot_pos[0]
        to_target_y = target[1] - dot_pos[1]
        to_targ_mag = math.hypot(to_target_x, to_target_y)
        if to_targ_mag == 0:
            continue

        # Normalized direction from dot to target
        to_targ_dir = [to_target_x / to_targ_mag, to_target_y / to_targ_mag]

        # Dist factor
        dist = distance(dot_pos, target)
        max_dist = math.hypot(WINDOW_SIZE[0], WINDOW_SIZE[1])
        dist_factor = 1 - (dist / max_dist)  # closer => bigger

        # Alignment with immediate user input
        align_human = (h_dir[0]*to_targ_dir[0] + h_dir[1]*to_targ_dir[1])

        # Alignment with recent direction from ghost path
        align_recent = (recent_dir[0]*to_targ_dir[0] + recent_dir[1]*to_targ_dir[1])

        # Weighted combination
        # - alignment with user input has some weighting
        # - alignment with recent direction has some weighting
        # - distance factor remains significant
        score = (align_human * 0.2) + (align_recent * 0.3) + (dist_factor * 0.5)

        if score > best_score:
            best_score = score
            best_target_idx = i

    # Hysteresis: if the best target is *different* from the current one,
    # check if improvement is at least GOAL_SWITCH_THRESHOLD.
    if best_target_idx != current_target_idx:
        # Recompute the score for the *current* target
        curr_score = float('-inf')
        i = current_target_idx

        to_target_x = targets[i][0] - dot_pos[0]
        to_target_y = targets[i][1] - dot_pos[1]
        to_targ_mag = math.hypot(to_target_x, to_target_y)
        if to_targ_mag > 0:
            to_targ_dir = [to_target_x / to_targ_mag, to_target_y / to_targ_mag]

            dist = distance(dot_pos, targets[i])
            max_dist = math.hypot(WINDOW_SIZE[0], WINDOW_SIZE[1])
            dist_factor = 1 - (dist / max_dist)

            align_human = (h_dir[0]*to_targ_dir[0] + h_dir[1]*to_targ_dir[1])
            align_recent = (recent_dir[0]*to_targ_dir[0] + recent_dir[1]*to_targ_dir[1])

            curr_score = (align_human * 0.2) + (align_recent * 0.3) + (dist_factor * 0.5)

        improvement = best_score - curr_score
        if improvement < GOAL_SWITCH_THRESHOLD:
            # Not enough improvement to switch
            return current_target_idx

    return best_target_idx

def generate_obstacles():
    obstacles.clear()
    if not ENABLE_OBSTACLES:
        return
    # If you want to generate obstacles, do it here
    pass

def generate_targets():
    targets.clear()
    for _ in range(NUM_GOALS):
        while True:
            pos = [random.randint(0, WINDOW_SIZE[0]),
                   random.randint(0, WINDOW_SIZE[1])]
            valid_position = True
            if ENABLE_OBSTACLES:
                for obstacle_pos in obstacles:
                    if distance(pos, obstacle_pos) < OBSTACLE_RADIUS * 1.5:
                        valid_position = False
                        break
            if valid_position:
                targets.append(pos)
                break

# -----------------------------------------
def move_dot(human_input):
    """
    Movement with Gaussian noise added to human portion.
    """
    global dot_pos, gamma, reached_goal, current_target_idx

    # --- Basic direction computations ---
    h_dx, h_dy = human_input
    h_mag = math.hypot(h_dx, h_dy)
    h_dir = [h_dx / h_mag, h_dy / h_mag] if h_mag > 0 else [0, 0]

    target_pos = targets[current_target_idx]
    w_dx = target_pos[0] - dot_pos[0]
    w_dy = target_pos[1] - dot_pos[1]
    w_mag = math.hypot(w_dx, w_dy)
    w_dir = [w_dx / w_mag, w_dy / w_mag] if w_mag > 0 else [0, 0]

    # Human input scaling: clamp to [0, 1]
    input_mag = min(max(h_mag / MAX_SPEED, 0), 1)
    step_size = MAX_SPEED * input_mag

    # W portion
    w_move_x = gamma * w_dir[0] * step_size
    w_move_y = gamma * w_dir[1] * step_size

    # H portion with noise
    if h_mag > 0:
        # Add Gaussian noise to direction
        noise_x = np.random.normal(0, NOISE_MAGNITUDE)
        noise_y = np.random.normal(0, NOISE_MAGNITUDE)
        
        # Combine original direction with noise
        noisy_dx = h_dir[0] + noise_x
        noisy_dy = h_dir[1] + noise_y
        
        # Renormalize
        noisy_mag = math.hypot(noisy_dx, noisy_dy)
        if noisy_mag > 0:
            noisy_dx /= noisy_mag
            noisy_dy /= noisy_mag
            
        h_move_x = (1 - gamma) * noisy_dx * step_size
        h_move_y = (1 - gamma) * noisy_dy * step_size
    else:
        h_move_x = 0
        h_move_y = 0

    # Final movement
    final_dx = w_move_x + h_move_x
    final_dy = w_move_y + h_move_y

    new_x = dot_pos[0] + final_dx
    new_y = dot_pos[1] + final_dy
    if not check_collision(dot_pos, [new_x, new_y]):
        dot_pos[0] = max(0, min(WINDOW_SIZE[0], new_x))
        dot_pos[1] = max(0, min(WINDOW_SIZE[1], new_y))

    # For arrow rendering
    final_mag = math.hypot(final_dx, final_dy)
    if final_mag > 0:
        x_dir = [final_dx / final_mag, final_dy / final_mag]
    else:
        x_dir = [0, 0]

    dist_to_goal = distance(dot_pos, target_pos)
    if dist_to_goal < GOAL_DETECTION_RADIUS:
        reached_goal = True
        pygame.time.set_timer(pygame.USEREVENT, 1000)

    return h_dir, w_dir, x_dir

def reset():
    global dot_pos, reached_goal, current_target_idx, gamma
    global recent_positions

    dot_pos = START_POS.copy()
    reached_goal = False
    current_target_idx = 0
    gamma = 0.2

    # Clear the ghost trail
    recent_positions = []

    generate_obstacles()
    generate_targets()
    pygame.time.set_timer(pygame.USEREVENT, 0)

# -----------------------------------------
def draw_arrow(surface, color, start_pos, direction, length=ARROW_LENGTH):
    dx, dy = direction
    if dx == 0 and dy == 0:
        return
    dir_length = math.hypot(dx, dy)
    dx /= dir_length
    dy /= dir_length

    end_x = start_pos[0] + dx * length
    end_y = start_pos[1] + dy * length

    pygame.draw.line(surface, color, start_pos, (end_x, end_y), int(2 * SCALING_FACTOR))

    arrow_size = 7 * SCALING_FACTOR
    angle = math.atan2(dy, dx)
    arrow1_x = end_x - arrow_size * math.cos(angle + math.pi/6)
    arrow1_y = end_y - arrow_size * math.sin(angle + math.pi/6)
    arrow2_x = end_x - arrow_size * math.cos(angle - math.pi/6)
    arrow2_y = end_y - arrow_size * math.sin(angle - math.pi/6)

    pygame.draw.line(surface, color, (end_x, end_y), (arrow1_x, arrow1_y), int(2 * SCALING_FACTOR))
    pygame.draw.line(surface, color, (end_x, end_y), (arrow2_x, arrow2_y), int(2 * SCALING_FACTOR))

# -----------------------------------------
def render(h_dir, w_dir, x_dir):
    screen.fill(WHITE)

    # Draw obstacles
    if ENABLE_OBSTACLES:
        for obstacle_pos in obstacles:
            pygame.draw.circle(screen, GRAY, (int(obstacle_pos[0]), int(obstacle_pos[1])), OBSTACLE_RADIUS)

    # Draw targets
    for i, target in enumerate(targets):
        pygame.draw.circle(screen, YELLOW, (int(target[0]), int(target[1])), TARGET_RADIUS)
        num_text = font.render(str(i + 1), True, BLACK)
        screen.blit(num_text, (target[0] - 5, target[1] - 12))

    # Highlight current target
    current_target = targets[current_target_idx]
    pygame.draw.circle(screen, BLACK, (int(current_target[0]), int(current_target[1])),
                       TARGET_RADIUS + 2, int(2 * SCALING_FACTOR))

    # Draw ghost path (trail)
    current_time = time.time()
    # 1. Remove old points
    while len(recent_positions) > 0 and (current_time - recent_positions[0][2]) > GHOST_TRAIL_DURATION:
        recent_positions.pop(0)

    # 2. Connect the points
    if len(recent_positions) > 1:
        for idx in range(len(recent_positions) - 1):
            x1, y1, t1 = recent_positions[idx]
            x2, y2, t2 = recent_positions[idx + 1]
            pygame.draw.line(screen, (200, 200, 200), (x1, y1), (x2, y2), 2)

    # Draw the dot
    pygame.draw.circle(screen, BLACK, (int(dot_pos[0]), int(dot_pos[1])),
                       DOT_RADIUS, int(2 * SCALING_FACTOR))

    # Draw directional arrows
    if h_dir != [0, 0]:
        draw_arrow(screen, BLUE, (int(dot_pos[0]), int(dot_pos[1])), h_dir, length=ARROW_LENGTH)
    if w_dir != [0, 0]:
        draw_arrow(screen, GREEN, (int(dot_pos[0]), int(dot_pos[1])), w_dir, length=ARROW_LENGTH)
    if x_dir != [0, 0]:
        draw_arrow(screen, RED, (int(dot_pos[0]), int(dot_pos[1])), x_dir, length=ARROW_LENGTH)

    # Info text
    gamma_text = font.render(f"Gamma: {gamma:.2f}", True, FONT_COLOR)
    screen.blit(gamma_text, (10, 10))

    formula_text = font.render(f"Movement = {gamma:.2f}W + {1 - gamma:.2f}H", True, FONT_COLOR)
    screen.blit(formula_text, (10, 40))
    
    noise_text = font.render(f"Noise σ: {NOISE_MAGNITUDE:.2f}", True, FONT_COLOR)
    screen.blit(noise_text, (10, 100))
    
    instructions_text = font.render("L2/R2: adjust gamma, [/]: adjust noise, R: reset", True, FONT_COLOR)
    screen.blit(instructions_text, (10, 70))

    # If goal reached
    if reached_goal:
        reset_text = font.render("Goal Reached! Auto-resetting...", True, FONT_COLOR)
        screen.blit(reset_text, (150, 110))

    # Legend
    legend_y = WINDOW_SIZE[1] - int(100 * SCALING_FACTOR)
    legend_spacing = int(30 * SCALING_FACTOR)
    legend_items = [
        ("Green Arrow: Perfect Path (W)", GREEN),
        ("Blue Arrow: Human Movement (H)", BLUE),
        ("Red Arrow: Dot's Movement", RED),
        ("Gray line: Ghost Path (recent trail)", (200, 200, 200))
    ]
    for i, (text, color) in enumerate(legend_items):
        label = font.render(text, True, color)
        screen.blit(label, (10, legend_y + i * legend_spacing))

    pygame.display.update()

# -----------------------------------------
running = True
clock = pygame.time.Clock()

generate_obstacles()
generate_targets()

# Adjust these based on your device
AXIS_L2 = 4
AXIS_R2 = 5

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFTBRACKET:  # [ key
                NOISE_MAGNITUDE = max(MIN_NOISE, NOISE_MAGNITUDE - NOISE_STEP)
            elif event.key == pygame.K_RIGHTBRACKET:  # ] key
                NOISE_MAGNITUDE = min(MAX_NOISE, NOISE_MAGNITUDE + NOISE_STEP)

        if joystick and event.type == pygame.JOYBUTTONDOWN:
            if event.button == 2:  # e.g. Square button
                reset()

        # If USEREVENT is triggered by timer (after goal reached)
        if event.type == pygame.USEREVENT:
            reset()

    if not reached_goal:
        # Basic keyboard input
        dx, dy = 0.0, 0.0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            dx -= 1.0
        if keys[pygame.K_RIGHT]:
            dx += 1.0
        if keys[pygame.K_UP]:
            dy -= 1.0
        if keys[pygame.K_DOWN]:
            dy += 1.0

        # Joystick input
        if joystick:
            axis_0 = joystick.get_axis(0)
            axis_1 = joystick.get_axis(1)
            deadzone = 0.1
            if abs(axis_0) > deadzone or abs(axis_1) > deadzone:
                dx = axis_0
                dy = axis_1
            else:
                dx = 0.0
                dy = 0.0

            # Adjust gamma with triggers
            l2_value = joystick.get_axis(AXIS_L2)
            r2_value = joystick.get_axis(AXIS_R2)
            if l2_value > 0.1:
                gamma = max(0.0, gamma - 0.01)
            if r2_value > 0.0:
                gamma = min(1.0, gamma + 0.01)

        # Keyboard deadzone
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            dx, dy = 0.0, 0.0

        # Scale input
        dx *= MAX_SPEED
        dy *= MAX_SPEED

        # Move the dot
        human_input = [dx, dy]

        # Use updated predict function
        proposed_target_idx = predict_human_target(human_input)
        current_target_idx = proposed_target_idx

        h_dir, w_dir, x_dir = move_dot(human_input)

        # Store ghost path positions (x, y, timestamp)
        recent_positions.append((dot_pos[0], dot_pos[1], time.time()))

    else:
        # Reached goal, no movement
        h_dir, w_dir, x_dir = [0, 0], [0, 0], [0, 0]

    render(h_dir, w_dir, x_dir)
    clock.tick(60)

pygame.quit()
