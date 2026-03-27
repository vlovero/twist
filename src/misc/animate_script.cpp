#include <fstream>
#include <iterator>
#include <string_view>

static const std::string_view default_script = R"TWIST(
import os
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def frame_generator(path, interval=1e-2):
    if not os.path.exists(path):
        return None

    last_modified = os.path.getmtime(path)
    current_state = None
    same_count = 0
    while True:
        sleep(interval)
        current_modified = os.path.getmtime(path)
        if current_modified != last_modified:
            with open(path, "rb") as file:
                current_state = np.fromfile(file, dtype=np.float64, count=-1)
                shape = current_state[:2]
                Nx, node = map(int, shape)
                current_state = current_state[2:]
                current_state = (current_state[:Nx], current_state[Nx : Nx * (node + 1)].reshape(Nx, node))
                yield current_state
            last_modified = current_modified
            same_count = 0
        else:
            if current_state is not None:
                yield current_state
            same_count += 1

        if same_count > 3000:
            return None


def update(frame, line, fig, ax):
    x, y = frame
    line.set_data(x, y[:, 0])
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()
    return (line,)


if __name__ == "__main__":
    path = ".cache/animation.dat"

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    while (frames := frame_generator(path)) is None:
        pass

    x, y = next(frames)
    ax.plot(x, y[:, 0], "--k")
    (line,) = ax.plot(x, y[:, 0], "b")
    # ax.set_ylim(-0.2, 1.1)
    anim = FuncAnimation(fig, lambda frame: update(frame, line, fig, ax), frames, blit=False, cache_frame_data=False, interval=0, save_count=0, repeat=False)
    plt.show()

)TWIST";

static std::string plot_script;

namespace python::animate
{
    void set_script(const std::string &script_path)
    {
        std::ifstream inputFile(script_path);
        plot_script = std::string((std::istreambuf_iterator<char>(inputFile)), std::istreambuf_iterator<char>());
    }

    const std::string_view get_script()
    {
        return plot_script.empty() ? default_script : plot_script;
    }
} // namespace python::animate
