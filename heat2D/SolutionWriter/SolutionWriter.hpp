#ifndef SOLUTIONWRITER_HPP
#define SOLUTIONWRITER_HPP

#include <zlib.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Mesh2D.hpp"

namespace heat2d {

class SolutionWriter {
 public:
  enum class FORMAT { CSV, VTU };

 private:
  FORMAT format_;
  std::ofstream filename_;
  std::filesystem::path directory_;
  std::string name_;
  std::size_t step_ = 0;

  static void appendInteger(std::string& data, std::uint64_t value,
                            unsigned width) {
    for (unsigned i = 0; i < width; ++i) {
      data.push_back(static_cast<char>(value & 0xff));
      value >>= 8;
    }
  }

  static void appendDouble(std::string& data, double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "VTU requires 64-bit doubles");
    std::memcpy(&bits, &value, sizeof(bits));
    appendInteger(data, bits, 8);
  }

  // Compressed VTK blocks start directly with the UInt32 compression header:
  // [block count][block size][partial size][compressed sizes...]. No byte count
  // prefix is used, unlike uncompressed appended data.
  static std::size_t appendCompressed(std::string& appended,
                                      const std::string& raw) {
    constexpr std::size_t block_size = 32768;
    const std::size_t offset = appended.size();
    const std::size_t count =
        raw.size() / block_size + (raw.size() % block_size != 0);
    if (count > std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("VTU compression header is too large");
    appendInteger(appended, count, 4);
    appendInteger(appended, block_size, 4);
    appendInteger(appended, raw.size() % block_size, 4);
    std::string body;
    for (std::size_t pos = 0; pos < raw.size(); pos += block_size) {
      const auto size =
          static_cast<uLong>(std::min(block_size, raw.size() - pos));
      uLongf compressed_size = compressBound(size);
      std::vector<Bytef> buffer(compressed_size);
      if (compress2(buffer.data(), &compressed_size,
                    reinterpret_cast<const Bytef*>(raw.data() + pos), size,
                    Z_DEFAULT_COMPRESSION) != Z_OK)
        throw std::runtime_error("VTU zlib compression failed");
      appendInteger(appended, compressed_size, 4);
      body.append(reinterpret_cast<const char*>(buffer.data()),
                  compressed_size);
    }
    appended += body;
    return offset;
  }

  static std::string escapeXML(const std::string& text) {
    std::string result;
    for (char c : text) {
      switch (c) {
        case '&':
          result += "&amp;";
          break;
        case '<':
          result += "&lt;";
          break;
        case '>':
          result += "&gt;";
          break;
        case '"':
          result += "&quot;";
          break;
        default:
          result += c;
      }
    }
    return result;
  }

  void writeVTU(const mesh::Mesh2D& mesh, const Eigen::VectorXd& sol,
                double t) {
    const auto& nodes = mesh.getNodes();
    const auto& connectivity = mesh.getElementConnectivity();
    const auto& offsets = mesh.getElementOffsets();
    const auto cells = mesh.getNumElements();
    if (offsets.size() != cells + 1 || offsets.front() != 0 ||
        offsets.back() != connectivity.size())
      throw std::runtime_error("Invalid VTU mesh offsets");
    for (std::size_t i = 0; i < cells; ++i) {
      if (offsets[i + 1] - offsets[i] != 3)
        throw std::runtime_error("VTU writer requires triangular cells");
    }

    std::string solution, times, points, indices, ends, types;
    for (const auto& node : nodes) {
      appendDouble(solution, sol[node.nodeID_]);
      appendDouble(points, node.x_);
      appendDouble(points, node.y_);
      appendDouble(points, 0.0);
    }
    for (auto index : connectivity) appendInteger(indices, index, 8);
    for (std::size_t i = 0; i < cells; ++i) {
      appendDouble(times, t);
      appendInteger(ends, offsets[i + 1], 8);
    }
    types.assign(cells, static_cast<char>(5));  // VTK_TRIANGLE
    std::string appended;
    const auto u_offset = appendCompressed(appended, solution);
    const auto time_offset = appendCompressed(appended, times);
    const auto points_offset = appendCompressed(appended, points);
    const auto indices_offset = appendCompressed(appended, indices);
    const auto ends_offset = appendCompressed(appended, ends);
    const auto types_offset = appendCompressed(appended, types);

    std::ostringstream reference;
    reference << name_ << '_' << std::setfill('0') << std::setw(5) << step_
              << ".vtu";
    std::ofstream file(directory_ / reference.str(), std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file << "<?xml version=\"1.0\"?>\n"
         << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" "
            "byte_order=\"LittleEndian\" header_type=\"UInt32\" "
            "compressor=\"vtkZLibDataCompressor\">\n"
         << "<UnstructuredGrid><Piece NumberOfPoints=\"" << nodes.size()
         << "\" NumberOfCells=\"" << cells << "\">\n";
    auto array = [&](const char* type, const char* name, unsigned components,
                     std::size_t offset) {
      file << "<DataArray type=\"" << type << "\" Name=\"" << name
           << "\" NumberOfComponents=\"" << components
           << "\" format=\"appended\" offset=\"" << offset << "\"/>\n";
    };
    file << "<PointData Scalars=\"u\">\n";
    array("Float64", "u", 1, u_offset);
    file << "</PointData><CellData>\n";
    array("Float64", "time", 1, time_offset);
    file << "</CellData><Points>\n";
    array("Float64", "points", 3, points_offset);
    file << "</Points><Cells>\n";
    array("Int64", "connectivity", 1, indices_offset);
    array("Int64", "offsets", 1, ends_offset);
    array("UInt8", "types", 1, types_offset);
    file << "</Cells></Piece></UnstructuredGrid>\n"
         << "<AppendedData encoding=\"raw\">\n_";
    file.write(appended.data(), static_cast<std::streamsize>(appended.size()));
    file << "\n</AppendedData></VTKFile>\n";
    file.close();
    filename_ << "<DataSet timestep=\"" << std::setprecision(17) << t
              << "\" file=\"" << escapeXML(reference.str()) << "\"/>\n";
    if (!filename_) throw std::runtime_error("Could not write PVD entry");
    ++step_;
  }

 public:
  SolutionWriter(const std::string& filename, FORMAT format = FORMAT::CSV)
      : format_(format) {
    if (format_ == FORMAT::VTU) {
      directory_ = filename;
      if (directory_.extension() == ".pvd") directory_.replace_extension();
      name_ = directory_.filename().string();
      if (name_.empty() || name_ == "." || name_ == "..")
        throw std::invalid_argument("Invalid VTU results name");
      std::filesystem::create_directories(directory_);
      filename_.open(directory_ / (name_ + ".pvd"));
      if (!filename_) throw std::runtime_error("Could not open PVD file");
      filename_ << "<?xml version=\"1.0\"?>\n"
                << "<VTKFile type=\"Collection\" version=\"0.1\">\n"
                << "<Collection>\n";
      return;
    }
    filename_.open(filename);
    if (!filename_.is_open())
      throw std::runtime_error("Could not open file to write solution\n");

    // Write header for output file
    filename_ << "x,y,u,t\n";
  }

  ~SolutionWriter() {
    if (format_ == FORMAT::VTU) filename_ << "</Collection>\n</VTKFile>\n";
  }

  // Make uncopyable
  SolutionWriter(const SolutionWriter&) = delete;
  SolutionWriter& operator=(const SolutionWriter&) = delete;

  void write(const mesh::Mesh2D& mesh, const Eigen::VectorXd& solution,
             double t) {
    if (solution.size() != static_cast<Eigen::Index>(mesh.getNodes().size()))
      throw std::invalid_argument("Solution size does not match mesh nodes");
    if (format_ == FORMAT::VTU) {
      writeVTU(mesh, solution, t);
      return;
    }
    const std::vector<mesh::Node2D>& nodes = mesh.getNodes();

    for (const auto& node : nodes) {
      filename_ << node.x_ << "," << node.y_ << "," << solution[node.nodeID_]
                << "," << t << "\n";
    }
  }
};

}  // namespace heat2d

#endif
