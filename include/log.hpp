#pragma once
#include <string>
#include <fstream>
#include <vector>

class CSVLogger {
private:
    std::ofstream file;

public:
    CSVLogger(const std::string& filename, const std::string& header);
    
    ~CSVLogger();

    void log_row(int step, const std::vector<float>& metrics);
};