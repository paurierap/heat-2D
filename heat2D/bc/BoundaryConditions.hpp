// heat2d/bc/BoundaryConditions.hpp
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "BoundaryCondition.hpp"
#include "DirichletBoundaryCondition.hpp"
#include "NeumannBoundaryCondition.hpp"
#include "RobinBoundaryCondition.hpp"

namespace heat2d::bc {

using BoundaryConditions =
    std::unordered_map<std::string, std::shared_ptr<BoundaryCondition>>;

}  // namespace heat2d::bc
