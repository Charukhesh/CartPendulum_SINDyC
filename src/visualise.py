import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import src.config as config
def animate_pendulum(t, x_states, save_path=None):
    """
    Creates an animation of the inverted pendulum system from simulation data
    """

    num_frames_target = 300
    total_frames = len(t)
    animation_step = max(1, total_frames // num_frames_target)

    t_anim = t[::animation_step]
    x_states_anim = x_states[::animation_step, :]

    # Extract state data for plotting
    cart_x = x_states_anim[:, 0]
    pendulum_theta = x_states_anim[:, 2]

    L = config.L
    cart_width = 0.4
    cart_height = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(np.min(cart_x) - L - 0.5, np.max(cart_x) + L + 0.5)
    ax.set_ylim(-L * 1.5, L * 1.5)
    ax.set_aspect('equal')
    ax.grid()
    ax.axhline(0, color='gray', lw=2)
    cart_patch = Rectangle((0, 0), cart_width, cart_height, fc='royalblue', ec='black')
    ax.add_patch(cart_patch)
    line, = ax.plot([], [], 'o-', lw=2, color='black', markersize=6)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        cart_patch.set_xy((-cart_width / 2, -cart_height / 2))
        line.set_data([], [])
        time_text.set_text('')
        return cart_patch, line, time_text

    def update(frame):
        x = cart_x[frame]
        theta = pendulum_theta[frame]
        cart_x_pos = x - cart_width / 2
        cart_y_pos = -cart_height / 2
        cart_patch.set_xy((cart_x_pos, cart_y_pos))
        pivot_x = x
        pivot_y = cart_height / 2
        bob_x = pivot_x + L * np.sin(theta)
        bob_y = pivot_y - L * np.cos(theta)
        line.set_data([pivot_x, bob_x], [pivot_y, bob_y])
        time_text.set_text(f'Time: {t_anim[frame]:.2f}s') # Use t_anim
        return cart_patch, line, time_text

    # Create the animation object
    ani = FuncAnimation(
        fig,
        update,
        frames=len(t_anim), # Use the length of the subsampled time
        init_func=init,
        blit=True,
        interval=config.DT * 1000 * animation_step, # Adjust interval for correct speed
        repeat=False
    )

    if save_path:
        print(f"Saving animation with {len(t_anim)} frames to {save_path}...")
        # Calculate a new FPS based on the subsampling
        fps = int(1 / (config.DT * animation_step))
        ani.save(save_path, writer='pillow', fps=fps)
        print("Save complete.")

    plt.title("Inverted Pendulum Controlled Stabilization")
    plt.xlabel("Horizontal Position (m)")
    plt.ylabel("Vertical Position (m)")
    plt.show()