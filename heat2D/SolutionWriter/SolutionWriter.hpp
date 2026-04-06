#ifndef SOLUTIONWRITER_HPP
#define SOLUTIONWRITER_HPP

#include <Eigen/Dense>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Mesh2D.hpp"

class SolutionWriter
{
    private:
        std::ofstream filename_;

    public: 
        SolutionWriter(const std::string& filename) : filename_(filename)
        {
            if (!filename_.is_open()) throw std::runtime_error("Could not open file to write solution\n");

            // Write header for output file
            filename_ << "x,y,u,t\n";
        }

        ~SolutionWriter() = default;

        // Make uncopyable
        SolutionWriter(const SolutionWriter&) = delete;
        SolutionWriter& operator=(const SolutionWriter&) = delete;

        void write(const spatial::Mesh2D& mesh, const Eigen::VectorXd& solution,  double t)
        {
            std::vector<spatial::Node2D> nodes = mesh.getNodes();

            for(const auto& node : nodes)
            {
                filename_ << node.x_ << "," << node.y_ << "," << solution[node.nodeID_] << "," << t << "\n";
            }
        }
};

#endif