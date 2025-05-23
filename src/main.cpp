#define NO_GRID_CUH_IMPL
#include "include/globeik_kernel.h"
#include "include/util.h"

#include <iostream>
#include <fstream>
#include <limits>  // For std::numeric_limits
#include <chrono>
#include <thread>

#include <yaml-cpp/yaml.h>

#define NUM_RANDOM_TARGETS 1

int main() {
    YAML::Node config;
    try {
        config = YAML::LoadFile("tests/mb_set.yaml");
    }
    catch (const YAML::Exception& e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
        return 1;
    }

    YAML::Node bookshelf = config["bookshelf_small_panda"];
    if (!bookshelf) {
        std::cerr << "Error: 'bookshelf_small_panda' section not found in YAML file" << std::endl;
        return 1;
    }

    std::vector<std::vector<float>> goalPoses;
    std::vector<std::vector<float>> startConfigs;

    std::cout << "Loading YAML File" << std::endl;
    for (std::size_t i = 0; i < bookshelf.size(); i++) {
        YAML::Node entry = bookshelf[i];

        YAML::Node goal_pose_node = entry["goal_pose"];
        if (!goal_pose_node) {
            std::cerr << "Warning: Entry " << i
                << " does not contain a 'goal_pose'." << std::endl;
            continue;
        }

        YAML::Node pos_node = goal_pose_node["position_xyz"];
        YAML::Node quat_node = goal_pose_node["quaternion_wxyz"];
        if (!pos_node || !quat_node) {
            std::cerr << "Warning: Entry " << i
                << " does not have the correct pose sub-structure." << std::endl;
            continue;
        }
        if (pos_node.size() != 3 || quat_node.size() != 4) {
            std::cerr << "Warning: Entry " << i
                << " pose dimensions are incorrect." << std::endl;
            continue;
        }

        std::vector<float> targetPose;
        for (std::size_t j = 0; j < pos_node.size(); j++) {
            targetPose.push_back(pos_node[j].as<float>());
        }
        for (std::size_t j = 0; j < quat_node.size(); j++) {
            targetPose.push_back(quat_node[j].as<float>());
        }
        goalPoses.push_back(targetPose);

        YAML::Node start_node = entry["start"];
        if (!start_node || start_node.size() != 7) {
            std::cerr << "Warning: Entry " << i
                << " does not contain a valid 7-element start config." << std::endl;
            continue;
        }

        std::vector<float> startConfig;
        for (std::size_t j = 0; j < start_node.size(); j++) {
            startConfig.push_back(start_node[j].as<float>());
        }
        startConfigs.push_back(startConfig);
    }
    std::cout << "Finished Loading YAML" << std::endl;

    bool printYAML = false;

    if (printYAML) {
        std::cout << "Extracted " << goalPoses.size() << " goal poses and "
            << startConfigs.size() << " start configurations:\n\n";

        for (std::size_t i = 0; i < goalPoses.size(); i++) {
            std::cout << "Entry " << i + 1 << " Goal Pose: ";
            for (float value : goalPoses[i]) {
                std::cout << value << " ";
            }
            std::cout << "\nEntry " << i + 1 << " Start Config: ";
            if (i < startConfigs.size()) {
                for (float value : startConfigs[i]) {
                    std::cout << value << " ";
                }
            }
            std::cout << "\n\n";
        }
    }

    std::ofstream result_file("globeik_iiwa_robometrics.csv");
    if (!result_file.is_open()) {
        std::cerr << "Failed to oepn result file" << std::endl;
        return 1;
    }

    grid::robotModel<float>* d_robotModel = grid::init_robotModel<float>();

    result_file << "Test, Solve Time (ms), Pos Error (mm), Ang Error (deg),\n";
    //for (int i = 0; i < goalPoses.size(); ++i) {
    for(int i=0; i<1; ++i) {
        float* target_pose = new float[7];
        for (int j = 0; j < 7; ++j) {
            target_pose[j] = goalPoses[i][j];
        }
        //target_pose[0] = -0.2;
        //target_pose[2] = 1.1;
        //target_pose[3] = 1.0;

        Result<float> res;
        res = generate_ik_solutions<float>(target_pose, d_robotModel, NUM_SOLUTIONS);

        std::cout << "target: ";
        for (int j = 0; j < 7; ++j) {
            std::cout << target_pose[j] << ", ";
        }
        std::cout << std::endl;

        std::cout << "pose: ";
        for (int j = 0; j < 7; ++j) {
            std::cout << res.pose[j] << ", ";
        }
        std::cout << std::endl;

        std::cout << "x: ";
        for (int j = 0; j < 7; ++j) {
            std::cout << res.joint_config[j] << ", ";
        }
        std::cout << std::endl;

        //float avg_serr = 0.0;
        //for (int j = 0; j < NUM_SOLUTIONS; ++j) {
        //    avg_serr += res.errors[j];
        //}
        //avg_serr /= NUM_SOLUTIONS;

        //float best_err = res.errors[0] * 1000.0;

        float avg_pos_err = 0.0f;
        float avg_ori_err = 0.0f;
        for (int j = 0; j < NUM_SOLUTIONS; ++j) {
            avg_pos_err += res.pos_errors[j];
            avg_ori_err += res.ori_errors[j];
        }
        avg_pos_err /= NUM_SOLUTIONS;
        avg_ori_err /= NUM_SOLUTIONS;

        float best_pos_err = res.pos_errors[0] * 1000.0;
        float best_ori_err = res.ori_errors[0];

        result_file << (i + 1) << ", "
            << (res.elapsed_time) << ", "
            //<< (best_err) << "\n";
            << (best_pos_err) << ", "
            << (best_ori_err) << "\n";

        delete[] target_pose;
        delete[] res.joint_config;
        delete[] res.pose;
        //delete[] res.errors;
        delete[] res.pos_errors;
        delete[] res.ori_errors;

        std::cout << "Case [" << i + 1 << "] out of " << goalPoses.size() << std::endl;
        std::cout << "Elapsed Time (ms): " << res.elapsed_time << std::endl;
        //std::cout << "Error (mm): " << best_err << std::endl;
        std::cout << "Pos Error (mm): " << best_pos_err << std::endl;
        std::cout << "Ang Error (deg): " << best_ori_err << std::endl;
    }

    return 0;
}

/*
int main() {
    
    // Create output/results file
    //std::ofstream result_file("globeik_iiwa.txt");
    std::ofstream result_file("globeik_iiwa_errors.csv");
    if (!result_file.is_open()) {
        std::cerr << "Failed to open result file!" << std::endl;
        return 1;
    }

    // Create random target positions
    std::vector<std::vector<float>> target_poses = generate_random_targets(NUM_RANDOM_TARGETS);

    // Variables to track overall metrics
    float total_time = 0.0;
    float total_err = 0.0;

    // Variables to track smallest and largest average errors (initialize appropriately)
    float min_error = std::numeric_limits<float>::max();
    float max_error = std::numeric_limits<float>::lowest();

    // Variables to track the associated test index for min and max errors
    int min_index = -1;
    int max_index = -1;

    grid::robotModel<float>* d_robotModel = grid::init_robotModel<float>();

    result_file << "Test, Solve Time (ms), Error (mm),\n";
    // Main loop over all random targets
    for (int i = 0; i < NUM_RANDOM_TARGETS; ++i) {
        std::cout << "Case [" << i + 1 << "] out of " << NUM_RANDOM_TARGETS << std::endl;

        // Set target position
        
        float* target_pose = new float[7];
        for (int j = 0; j < 7; ++j) {
            target_pose[j] = target_poses[i][j];
        }
        
        //float* target_pos = new float[DIM];
        //target_pos[0] = -0.442223;
        //target_pos[1] = -0.273242;
        //target_pos[2] = 0.297876;

        Result<float> res;

        // Run on the current target
        //for (int i=0; i<10; i++) 
        res = generate_ik_solutions<float>(target_pose, d_robotModel, NUM_SOLUTIONS);
        
        // Calculate the average error for this target
        float avg_serr = 0.0;
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
        
        //result_file << "Test " << i + 1 << ":\n";
        //result_file << "Elapsed time: " << res.elapsed_time << " ms\n";
        //result_file << "Error: " << avg_serr << "\n";
        //result_file << "Target Pos: " << target_pos[0] << ", "
        //    << target_pos[1] << ", " << target_pos[2] << "\n";
        //result_file << "EE Pos: " << res.ee_pos[0] << ", "
        //    << res.ee_pos[1] << ", " << res.ee_pos[2] << "\n\n";
        
        //result_file << "Target Pos: " << target_pos[0] << ", "
            //<< target_pos[1] << ", " << target_pos[2] << "\n";
        
        //for (int j = 0; j < NUM_SOLUTIONS; ++j) {
        //    result_file << "Solution: " << j + 1 << ":\n";
        //    result_file << "Time: " << res.elapsed_time << "\n";
        //    result_file << "Error: " << res.errors[j] << "\n";
        //    result_file << "EE Pos: " << res.ee_pos[j * DIM + 0] << ", "
        //        << res.ee_pos[j * DIM + 1] << ", " << res.ee_pos[j * DIM + 2] << "\n";
        //    result_file << "Joint Config: " << res.joint_config[j * DIM + 0] << ", "
        //        << res.joint_config[j * DIM + 1] << ", " << res.joint_config[j * DIM + 2] << ", "
        //        << res.joint_config[j * DIM + 3] << ", " << res.joint_config[j * DIM + 4] << ", "
        //        << res.joint_config[j * DIM + 5] << ", " << res.joint_config[j * DIM + 6] << "\n\n";
        //}
        

        float best_err = res.errors[0] * 1000.0;

        result_file << (i + 1) << ", "
            << (res.elapsed_time) << ", "
            << (best_err) << "\n";

        // Aggregate total metrics
        total_time += res.elapsed_time;
        total_err += avg_serr;

        // Free dynamically allocated memory
        delete[] target_pose;
        delete[] res.joint_config;
        delete[] res.pose;
        delete[] res.errors;
    }

    // Calculate overall averages
    float avg_time = total_time / NUM_RANDOM_TARGETS;
    float avg_err = total_err / NUM_RANDOM_TARGETS;

    // Print out the average metrics
    std::cout << "Avg time: " << avg_time << " ms" << std::endl;
    std::cout << "Avg error: " << avg_err * 1000.0 << " mm" << std::endl;

    // Print the smallest and largest average errors along with the associated test number
    std::cout << "Smallest avg error: " << min_error * 1000.0 << " mm (Test " << min_index + 1 << ")" << std::endl;
    std::cout << "Largest avg error: " << max_error * 1000.0 << " mm (Test " << max_index + 1 << ")" << std::endl;

    result_file.close();
    
    return 0;
}
*/