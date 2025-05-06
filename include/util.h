#pragma once

#include <vector>
#include <random>
#include <cmath>

// Used to generate random targets for benchmarking
inline std::vector<std::vector<float>> generate_random_targets(int num_targets) {
    std::vector<std::vector<float>> random_targets;
    std::mt19937 gen(1337); // Set seed for target generation

    // For deciding which interval to use
    std::uniform_real_distribution<> dist_choice(0.0, 1.0);

    // Distributions for the two intervals for x and y
    std::uniform_real_distribution<> left_dist(-1.0, -0.2);
    std::uniform_real_distribution<> right_dist(0.2, 1.0);

    // Distribution for z remains unchanged
    std::uniform_real_distribution<> dist_z(0.2, 1.0);

    // Ensure points are within the robot workspace (e.g., inside a unit sphere)
    for (int i = 0; i < num_targets; ++i) {
        float x, y, z;
        float qw, qx, qy, qz;
        qw = 0; qx = 0; qy = 0; qz = 0;
        do {
            // Choose interval for x
            if (dist_choice(gen) < 0.5) {
                x = left_dist(gen);
            }
            else {
                x = right_dist(gen);
            }

            // Choose interval for y
            if (dist_choice(gen) < 0.5) {
                y = left_dist(gen);
            }
            else {
                y = right_dist(gen);
            }

            // z is sampled uniformly from [0, 1]
            z = dist_z(gen);

        } while (std::sqrt(x * x + y * y + z * z) > 0.7);

        random_targets.push_back({ x, y, z , qw, qx, qy, qz});
    }

    return random_targets;
}
