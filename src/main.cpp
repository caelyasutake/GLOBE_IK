#define NO_GRID_CUH_IMPL
#include "include/globeik_kernel.h"
#include "include/util.h"

#include <iostream>
#include <fstream>
#include <limits>  // For std::numeric_limits
#include <chrono>
#include <thread>

#define NUM_RANDOM_TARGETS 1
#define NUM_SOLUTIONS 1

// Take best error from num_solutions = 1000, num_random_targets = 1000

int main() {
    
    // Create output/results file
    //std::ofstream result_file("globeik_iiwa.txt");
    std::ofstream result_file("globeik_iiwa_errors.csv");
    if (!result_file.is_open()) {
        std::cerr << "Failed to open result file!" << std::endl;
        return 1;
    }

    // Create random target positions
    std::vector<std::vector<double>> target_positions = generate_random_targets(NUM_RANDOM_TARGETS);

    // Variables to track overall metrics
    double total_time = 0.0;
    double total_err = 0.0;

    // Variables to track smallest and largest average errors (initialize appropriately)
    double min_error = std::numeric_limits<double>::max();
    double max_error = std::numeric_limits<double>::lowest();

    // Variables to track the associated test index for min and max errors
    int min_index = -1;
    int max_index = -1;

    grid::robotModel<double>* d_robotModel = grid::init_robotModel<double>();

    result_file << "Test, Solve Time (ms), Error (mm),\n";
    // Main loop over all random targets
    for (int i = 0; i < NUM_RANDOM_TARGETS; ++i) {
        std::cout << "Case [" << i + 1 << "] out of " << NUM_RANDOM_TARGETS << std::endl;

        // Set target position
        
        double* target_pos = new double[DIM];
        for (int j = 0; j < DIM; ++j) {
            target_pos[j] = target_positions[i][j];
        }
        
        //double* target_pos = new double[DIM];
        //target_pos[0] = -0.442223;
        //target_pos[1] = -0.273242;
        //target_pos[2] = 0.297876;

        Result<double> res;

        // Run on the current target
        //for (int i=0; i<10; i++) 
        res = generate_ik_solutions<double>(target_pos, d_robotModel, NUM_SOLUTIONS);
        
        // Calculate the average error for this target
        double avg_serr = 0.0;
        for (int j = 0; j < NUM_SOLUTIONS; ++j) {
            avg_serr += res.errors[j];
        }
        avg_serr /= NUM_SOLUTIONS;

        // Update the smallest and largest average error values and associated indices
        if (avg_serr < min_error) {
            min_error = avg_serr;
            min_index = i;
        }
        if (avg_serr > max_error) {
            max_error = avg_serr;
            max_index = i;
        }

        // Write details to the results file
        /*
        result_file << "Test " << i + 1 << ":\n";
        result_file << "Elapsed time: " << res.elapsed_time << " ms\n";
        result_file << "Error: " << avg_serr << "\n";
        result_file << "Target Pos: " << target_pos[0] << ", "
            << target_pos[1] << ", " << target_pos[2] << "\n";
        result_file << "EE Pos: " << res.ee_pos[0] << ", "
            << res.ee_pos[1] << ", " << res.ee_pos[2] << "\n\n";
        */
        //result_file << "Target Pos: " << target_pos[0] << ", "
            //<< target_pos[1] << ", " << target_pos[2] << "\n";
        /*
        for (int j = 0; j < NUM_SOLUTIONS; ++j) {
            result_file << "Solution: " << j + 1 << ":\n";
            result_file << "Time: " << res.elapsed_time << "\n";
            result_file << "Error: " << res.errors[j] << "\n";
            result_file << "EE Pos: " << res.ee_pos[j * DIM + 0] << ", "
                << res.ee_pos[j * DIM + 1] << ", " << res.ee_pos[j * DIM + 2] << "\n";
            result_file << "Joint Config: " << res.joint_config[j * DIM + 0] << ", "
                << res.joint_config[j * DIM + 1] << ", " << res.joint_config[j * DIM + 2] << ", "
                << res.joint_config[j * DIM + 3] << ", " << res.joint_config[j * DIM + 4] << ", "
                << res.joint_config[j * DIM + 5] << ", " << res.joint_config[j * DIM + 6] << "\n\n";
        }
        */

        double best_err = res.errors[0] * 1000.0;

        result_file << (i + 1) << ", "
            << (res.elapsed_time) << ", "
            << (best_err) << "\n";

        // Aggregate total metrics
        total_time += res.elapsed_time;
        total_err += avg_serr;

        // Free dynamically allocated memory
        delete[] target_pos;
        delete[] res.joint_config;
        delete[] res.ee_pos;
        delete[] res.errors;
    }

    // Calculate overall averages
    double avg_time = total_time / NUM_RANDOM_TARGETS;
    double avg_err = total_err / NUM_RANDOM_TARGETS;

    // Print out the average metrics
    std::cout << "Avg time: " << avg_time << " ms" << std::endl;
    std::cout << "Avg error: " << avg_err * 1000.0 << " mm" << std::endl;

    // Print the smallest and largest average errors along with the associated test number
    std::cout << "Smallest avg error: " << min_error * 1000.0 << " mm (Test " << min_index + 1 << ")" << std::endl;
    std::cout << "Largest avg error: " << max_error * 1000.0 << " mm (Test " << max_index + 1 << ")" << std::endl;

    result_file.close();
    
    return 0;
}
